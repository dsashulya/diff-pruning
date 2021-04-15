import numpy as np
from typing import Tuple, List, NoReturn
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

NO_DECAY = ["bias", "LayerNorm.weight"]


class ConcreteStretched:
    def __init__(self, l: float, r: float, device="cuda"):
        """
       Parameters:
           l: float
               constant used to stretch s (l < 0)
           r: float
               constant used to stretch s (r > 1)
       """
        self.device = device
        self.l = l
        self.r = r
        self.dz_dalpha = None

    def __call__(self, alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
           z: torch.Tensor
               binary mask vector relaxed into continuous space
       """
        u = torch.rand_like(alpha).clamp(1e-8, 1)
        s = (torch.sigmoid(u.log() - (1 - u).log() + alpha))
        s_ = s * (self.r - self.l) + self.l
        t = torch.max(torch.zeros_like(s_), s_)
        z = torch.min(torch.ones_like(s_), t)

        # dz/da = dz/dt * dt/ds_ * ds_/ds * ds/da
        dz_dt = (t < 1).float()
        dt_ds_ = (s_ > 0).float()
        ds__ds = self.r - self.l
        ds_dalpha = s * (1 - s)
        self.dz_dalpha = dz_dt * dt_ds_ * ds__ds * ds_dalpha

        return z, self.dz_dalpha

    def backward(self, grad: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters:
            grad: torch.Tensor
                gradient of loss w.r.t class output
        Returns:
            dz_dalpha: torch.Tensor
                derivative of z w.r.t parameters alpha
        """
        assert self.dz_dalpha is not None, "Do a forward pass first"
        if grad is not None:
            return self.dz_dalpha * grad
        return self.dz_dalpha


class DiffPruning:
    def __init__(self, model: nn.Module, modelname: str, concrete_lower: float, concrete_upper: float,
                 epochs: int, weight_decay: float,
                 warmup_steps: int, training_steps: int, gradient_accumulation_steps: int,
                 max_grad_norm: float,
                 sparsity_penalty: float, sparsity_penalty_per_layer: List[float] = None,
                 lr_params: float = 0.001, lr_alpha: float = 0.1, eps: float = 1e-8,
                 alpha_init: torch.Tensor = None, per_params_alpha: bool = False, per_layer_alpha: bool = False,
                 device: str = "cuda"):
        """
        Parameters:
            model: nn.Module
                transformers model with pretrained parameters
            concrete_lower: float
               constant used to stretch s (l < 0)
            concrete_upper: float
               constant used to stretch s (r > 1)
            epochs: int
                num of training epochs
            weight_decay: float
                weight decay for the params optimiser
            warmup_steps: int
                num steps for the lr to increase
            training_steps: int
                total num of training steps
            lr_params: float
                learning rate for diff params
            lr_alpha: float
                learning rate for alpha
            eps: float
                Adam eps
            alpha_init: torch.Tensor
                initial values of alpha
            device: str
                device used for training (cpu / cuda)
        """
        self.model = model.to(device)
        self.pp_alpha = per_params_alpha
        self.pl_alpha = per_layer_alpha
        self.device = device
        self.modelname = modelname.lower()
        self.total_layers = 14
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.alpha_init = alpha_init
        self.log_ratio = self._get_log_ratio(concrete_lower, concrete_upper)
        self.concrete_stretched = ConcreteStretched(concrete_lower, concrete_upper)

        self.sparsity_penalty = [sparsity_penalty] * self.total_layers \
            if sparsity_penalty_per_layer is None else sparsity_penalty_per_layer
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pretrained = []
        self.diff = []
        self.alpha = []
        self.per_params_alpha = None
        self.per_layer_alpha = None
        self.optimiser_grouped_parameters = []
        self.bert_params = {}
        self._get_params()

        self.optimizer_params = AdamW(self.optimiser_grouped_parameters, lr=lr_params, eps=eps)
        self.optimizer_alpha = AdamW(self.alpha, lr=lr_alpha, eps=eps)

        self.scheduler_params = get_linear_schedule_with_warmup(
            self.optimizer_params, num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
        self.scheduler_alpha = get_linear_schedule_with_warmup(
            self.optimizer_alpha, num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

    def _get_layer_ind(self, layer_name: str) -> int:
        ind = 0
        if f"{self.modelname}.embeddings" in layer_name:
            return ind
        elif f"{self.modelname}.encoder.layer" in layer_name:
            ind = int(layer_name.split(".")[3])
        else:
            ind = self.total_layers - 1
        return ind

    @staticmethod
    def _get_log_ratio(concrete_lower: float, concrete_upper: float) -> float:
        """
        Parameters:
            concrete_lower: float
               constant used to stretch s (l < 0)
            concrete_upper: float
               constant used to stretch s (r > 1)
        Returns:
            log_ratio: float
                log(-l / r)
        """
        if concrete_lower == 0:
            return 0
        return np.log(-concrete_lower / concrete_upper)

    def _get_params(self) -> NoReturn:
        decay, no_decay = {'params': [], 'weight_decay': self.weight_decay}, \
                          {'params': [], 'weight_decay': 0.}

        for name, param in self.model.named_parameters():
            pretrained = torch.zeros_like(param.data).copy_(param).to(self.device)
            diff = torch.zeros_like(param.data, requires_grad=True).to(self.device)
            alpha = torch.zeros_like(param.data, requires_grad=True).to(self.device)
            if self.alpha_init is not None:
                alpha += self.alpha_init

            diff.grad = torch.zeros_like(diff)
            alpha.grad = torch.zeros_like(alpha)

            if name in NO_DECAY:
                no_decay['params'].append(diff)
            else:
                decay['params'].append(diff)

            self.bert_params[name] = [pretrained, diff, alpha]  # all parameters
            self.diff.append(self.bert_params[name][1])  # only diff vector w
            self.alpha.append(self.bert_params[name][2])  # only diff vector alpha

        # PER PARAMS ALPHA
        if self.pp_alpha:
            self.per_params_alpha = {}
            for name, param in self.model.named_parameters():
                alpha = torch.zeros(1).to(self.device) + self.alpha_init
                alpha.requires_grad = True
                alpha.grad = torch.zeros_like(alpha)
                self.per_params_alpha[name] = alpha
                self.alpha.append(alpha)
        # PER LAYER ALPHA
        elif self.pl_alpha:
            self.per_layer_alpha = torch.zeros(self.total_layers, requires_grad=True).to(self.device) + self.alpha_init
            self.per_layer_alpha.grad = torch.zeros_like(self.per_layer_alpha)

        self.optimiser_grouped_parameters.extend([decay, no_decay])

    def evaluate(self, dataloader: DataLoader) -> float:
        total_loss = 0.
        self.model.eval()
        with torch.no_grad():
            epoch_iterator = tqdm(dataloader, desc="Val iteration")
            for _, batch in enumerate(epoch_iterator):
                for name, param in self.model.named_parameters():
                    if "classifier" in name:
                        param.data.copy_(self.bert_params[name][0].data + self.bert_params[name][1].data)
                    else:
                        layer_ind = self._get_layer_ind(name)
                        z, _ = self.concrete_stretched(self.bert_params[name][2])
                        if self.pp_alpha:  # per params alpha
                            z2, _ = self.concrete_stretched(self.per_params_alpha[name])
                        elif self.pl_alpha:  # per layer alpha
                            z2, _ = self.concrete_stretched(self.per_layer_alpha[layer_ind])
                        else:  # only base alpha
                            z2 = torch.ones_like(z.data)

                        param.data.copy_(self.bert_params[name][0].data +
                                         (z * z2).data * self.bert_params[name][1].data)

                input_ids, attention_mask, token_type_ids, labels = batch
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                total_loss += output.loss
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.model.zero_grad()

        train_losses = []
        train_iterator = trange(0, int(self.epochs), desc="Epoch")
        for epoch in train_iterator:
            # l0_penalty = [0] * self.total_layers
            l0_penalty_sum = torch.zeros(1, requires_grad=True).to(self.device)

            epoch_iterator = tqdm(train_dataloader, desc="Train iteration")
            for step, batch in enumerate(epoch_iterator):
                nonzero_params = 0
                grad_params = {}
                if self.pp_alpha:
                    per_params_z_grad = {}
                elif self.pl_alpha:
                    per_layer_z_grad = [torch.empty(1)] * self.total_layers

                prev_layer_ind = -1
                # iterate over params, calculate all model params (params + z * z2 * w) and l0 penalty
                for name, param in self.model.named_parameters():
                    if "classifier" in name:
                        nonzero_params += param.numel()
                        # classifier diff vec is added to pretrained w/o z (base params + w)
                        param.data.copy_(self.bert_params[name][0].data + self.bert_params[name][1].data)
                    else:
                        layer_ind = self._get_layer_ind(name)
                        # passing main alpha through concrete stretched
                        z, z_grad = self.concrete_stretched(self.bert_params[name][2])
                        if self.pp_alpha:  # per params alpha
                            z2, z2_grad = self.concrete_stretched(self.per_params_alpha[name])
                            per_params_z_grad[name] = z2_grad
                            l0_penalty_sum += torch.sigmoid(self.per_params_alpha[name] - self.log_ratio).sum()
                            # l0_penalty[layer_ind] += torch.sigmoid(self.per_params_alpha[name] - self.log_ratio).sum()

                        elif self.pl_alpha and layer_ind != prev_layer_ind:  # per layer alpha
                            # if layer number changed
                            z2, z2_grad = self.concrete_stretched(self.per_layer_alpha[layer_ind])
                            per_layer_z_grad[layer_ind] = z2_grad
                            l0_penalty_sum += torch.sigmoid(self.per_layer_alpha[layer_ind] - self.log_ratio).sum()
                            # l0_penalty[layer_ind] = torch.sigmoid(self.per_layer_alpha[layer_ind] - self.log_ratio).sum()

                        else:  # only base alpha
                            z2, z2_grad = torch.ones_like(z.data), torch.ones_like(z_grad.data)

                        grad_params[name] = [self.bert_params[name][1] * z2,
                                             z * z2,
                                             z_grad,
                                             self.bert_params[name][1] * z]

                        l0_penalty_sum += torch.sigmoid(self.bert_params[name][2] - self.log_ratio).sum()

                        param.data.copy_(self.bert_params[name][0].data +
                                         (z * z2).data * self.bert_params[name][1].data)
                        nonzero_params += ((z * z2) > 0).float().detach().sum().item()
                        prev_layer_ind = layer_ind

                # forward pass
                self.model.train()
                input_ids, attention_mask, token_type_ids, labels = batch
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                loss = output.loss
                train_losses.append(loss)

                if self.gradient_accumulation_steps > 1:
                    loss /= self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or \
                        (step + 1) == len(epoch_iterator):  # last batch in epoch
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        if "classifier" in name:
                            # copying updated gradient to the respective w vector
                            self.bert_params[name][1].grad.copy_(param.grad.data)
                        else:
                            # adding grad to w
                            self.bert_params[name][1].grad.copy_(param.grad.data *
                                                                 grad_params[name][1].data)
                            # adding grad to base alpha
                            self.bert_params[name][2].grad.copy_(param.grad.data *
                                                                 grad_params[name][0].data *
                                                                 grad_params[name][2].data)
                            # adding grad to per param / layer alpha
                            if self.pp_alpha:
                                self.per_params_alpha[name].grad.copy_(torch.sum(param.grad.data *
                                                                                 grad_params[name][3].data *
                                                                                 per_params_z_grad[name].data))
                            elif self.pl_alpha:
                                ind = self._get_layer_ind(name)
                                self.per_layer_alpha[ind] += torch.sum(param.grad.data *
                                                                       grad_params[name][3].data *
                                                                       per_layer_z_grad[ind].data)

                    l0_penalty_sum.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.diff, self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.alpha, self.max_grad_norm)

                    self.optimizer_params.step()
                    self.optimizer_alpha.step()
                    self.scheduler_params.step()
                    self.scheduler_alpha.step()

                    self.model.zero_grad()

                    avg_val_loss = self.evaluate(val_dataloader)
                    print(f"Epoch {epoch + 1}: train loss = {np.array(train_losses).mean():.3f} val loss = {avg_val_loss:.3f}")
                    # empty the list
                    train_losses = []
