import numpy as np
import logging
from typing import Tuple, List, Union, NoReturn
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

    def __call__(self, alpha: torch.Tensor, return_grad: bool = True) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Parameters:
            alpha: torch.Tensor
                probabilities used in uniform distribution
            return_grad: bool
                whether to compute and return dz_dalpha
        Returns:
           z: torch.Tensor
               binary mask vector relaxed into continuous space
            dz_dalpha: Optional[torch.Tensor]
                gradient of the output with respect to input
       """
        u = torch.rand_like(alpha).clamp(1e-8, 1)
        s = (torch.sigmoid(u.log() - (1 - u).log() + alpha))
        s_ = s * (self.r - self.l) + self.l
        t = torch.max(torch.zeros_like(s_), s_)
        z = torch.min(torch.ones_like(s_), t)

        if return_grad:
            # dz/da = dz/dt * dt/ds_ * ds_/ds * ds/da
            dz_dt = (t < 1).float()
            dt_ds_ = (s_ > 0).float()
            ds__ds = self.r - self.l
            ds_dalpha = s * (1 - s)
            dz_dalpha = dz_dt * dt_ds_ * ds__ds * ds_dalpha
            return z, dz_dalpha
        return z


class DiffPruning:
    def __init__(self, model: nn.Module, model_name: str, concrete_lower: float, concrete_upper: float,
                 total_layers: int, weight_decay: float,
                 warmup_steps: int, gradient_accumulation_steps: int,
                 max_grad_norm: float,
                 lr_params: float = 0.001, lr_alpha: float = 0.1, eps: float = 1e-8,
                 alpha_init: torch.Tensor = None, per_params_alpha: bool = False, per_layer_alpha: bool = False,
                 device: str = "cuda"):
        """
        Parameters:
            model: nn.Module
                transformers model with pretrained parameters
            model_name: str
                name of the base model
            concrete_lower: float
               constant used to stretch s (l < 0)
            concrete_upper: float
               constant used to stretch s (r > 1)
            total_layers: int
                number of layers in the model
            weight_decay: float
                weight decay for the params optimiser
            warmup_steps: int
                num steps for the lr to increase
            gradient_accumulation_steps: int
                num steps while gradients are being accumulated
            max_grad_norm: float
                maximum value to which all gradients are clipped
            lr_params: float
                learning rate for diff params
            lr_alpha: float
                learning rate for alpha
            eps: float
                Adam eps
            alpha_init: torch.Tensor
                initial values of alpha
            per_params_alpha: bool
                structured diff pruning with groups formed based on param name
            per_layer_alpha: bool
                structured diff pruning with groups formed based on layer number
            device: str
                device used for training (cpu / cuda)
        """
        self.model = model.to(device)
        assert not (per_params_alpha and per_layer_alpha), \
            "Structured diff pruning can only be used either per params or per layer"
        self.pp_alpha = per_params_alpha
        self.pl_alpha = per_layer_alpha
        self.device = device
        self.model_name = model_name.lower()
        self.total_layers = total_layers
        self.max_grad_norm = max_grad_norm
        self.alpha_init = alpha_init
        self.log_ratio = self._get_log_ratio(concrete_lower, concrete_upper)
        self.concrete_stretched = ConcreteStretched(concrete_lower, concrete_upper)

        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.bert_params = {}
        self.diff = []
        self.alpha = []
        self.optimiser_grouped_parameters = []
        self.per_params_alpha = None
        self.per_layer_alpha = None
        self.__get_params()

        self.optimizer_params = AdamW(self.optimiser_grouped_parameters, lr=lr_params, eps=eps)
        self.optimizer_alpha = AdamW(self.alpha, lr=lr_alpha, eps=eps)

        self.warmup_steps = warmup_steps
        self.scheduler_params = None
        self.scheduler_alpha = None

        # for training
        self.logger = logging.getLogger(__name__)
        self.grad_params = None
        self.per_params_z_grad = None
        self.per_layer_z_grad = None

    def __prepare_scheduler(self, train_dataloader: DataLoader, epochs: int) -> NoReturn:
        """ Creates schedulers for diff and alpha """
        training_steps = len(train_dataloader) // self.gradient_accumulation_steps * epochs
        if self.gradient_accumulation_steps > len(train_dataloader):
            training_steps = epochs

        self.scheduler_params = get_linear_schedule_with_warmup(
            self.optimizer_params, num_warmup_steps=self.warmup_steps,
            num_training_steps=training_steps
        )
        self.scheduler_alpha = get_linear_schedule_with_warmup(
            self.optimizer_alpha, num_warmup_steps=self.warmup_steps,
            num_training_steps=training_steps
        )

    def _get_layer_ind(self, layer_name: str) -> int:
        """
        Parameters:
            layer_name: str
                name of the model layer
        Returns:
            index: int
                index corresponding to the layer name
        """
        ind = 0
        if f"{self.model_name}.embeddings" in layer_name:
            return ind
        elif f"{self.model_name}.encoder.layer" in layer_name:
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

    def __get_params(self) -> NoReturn:
        """
        Extracts model parameters and initialises w and alpha
        """
        decay = {'params': [], 'weight_decay': self.weight_decay}
        no_decay = {'params': [], 'weight_decay': 0.}
        if self.pp_alpha:
            self.per_params_alpha = {}

        for name, param in self.model.named_parameters():
            pretrained = torch.zeros_like(param.data, requires_grad=False).copy_(param).to(self.device)
            diff = torch.zeros_like(param.data, requires_grad=True).to(self.device)
            alpha = torch.zeros_like(param.data, requires_grad=True).to(self.device)
            # diff = torch.randn_like(param.data, requires_grad=True).to(self.device)
            # alpha = torch.randn_like(param.data, requires_grad=True).to(self.device)
            if self.alpha_init is not None:
                alpha += self.alpha_init

            diff.grad = torch.zeros_like(diff)
            alpha.grad = torch.zeros_like(alpha)

            if any(no_decay_layer in name for no_decay_layer in NO_DECAY):
                no_decay['params'].append(diff)
            else:
                decay['params'].append(diff)

            self.bert_params[name] = [pretrained, diff, alpha]  # all parameters
            self.diff.append(self.bert_params[name][1])  # only diff vector w
            self.alpha.append(self.bert_params[name][2])  # only diff vector alpha

            # PER PARAMS ALPHA
            if self.pp_alpha:
                alpha = torch.zeros(1).to(self.device)
                if self.alpha_init is not None:
                    alpha += self.alpha_init
                alpha.requires_grad = True
                alpha.grad = torch.zeros_like(alpha)
                self.per_params_alpha[name] = alpha
                self.alpha.append(alpha)

        # PER LAYER ALPHA
        if self.pl_alpha:
            self.per_layer_alpha = torch.zeros(self.total_layers, requires_grad=True).to(self.device)
            if self.alpha_init is not None:
                self.per_layer_alpha += self.alpha_init
            self.per_layer_alpha.grad = torch.zeros_like(self.per_layer_alpha)

        self.optimiser_grouped_parameters.extend([decay, no_decay])
        assert len(self.bert_params) and len(self.diff) and len(self.alpha), "Parameters were not initialised"

    def __calculate_params(self) -> Tuple[torch.Tensor, int]:
        """
        Copies trained parameters into the model
        Returns:
             l0_penalty_sum: torch.Tensor
                total l0 penalty used in backprop
            nonzero_params: int
                total number of nonzero parameters in diff vector
         """
        # l0_penalty_sum = torch.tensor([0.], requires_grad=True).to(self.device)
        nonzero_params = 0
        l0_created = False
        self.grad_params = {}
        if self.pp_alpha:
            self.per_params_z_grad = {}
        elif self.pl_alpha:
            self.per_layer_z_grad = {}

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
                if not l0_created:
                    l0_penalty_sum = torch.sigmoid(self.bert_params[name][2] - self.log_ratio).sum()
                    l0_created = True
                else:
                    l0_penalty_sum += torch.sigmoid(self.bert_params[name][2] - self.log_ratio).sum()

                # PER PARAMS ALPHA
                if self.pp_alpha:
                    z2, z2_grad = self.concrete_stretched(self.per_params_alpha[name])
                    self.per_params_z_grad[name] = z2_grad
                    l0_penalty_sum += torch.sigmoid(self.per_params_alpha[name] - self.log_ratio).sum()

                # PER LAYER ALPHA
                elif self.pl_alpha and layer_ind != prev_layer_ind:  # if layer number changed
                    z2, z2_grad = self.concrete_stretched(self.per_layer_alpha[layer_ind])
                    self.per_layer_z_grad[layer_ind] = z2_grad
                    l0_penalty_sum += torch.sigmoid(self.per_layer_alpha[layer_ind] - self.log_ratio).sum()

                else:  # only base alpha
                    z2 = torch.ones_like(z.data)

                self.grad_params[name] = [self.bert_params[name][1] * z2,
                                          z * z2,
                                          z_grad,
                                          self.bert_params[name][1] * z]

                param.data.copy_(self.bert_params[name][0].data +
                                 (z * z2).data * self.bert_params[name][1].data)
                nonzero_params += ((z * z2) > 0).float().detach().sum().item()
                prev_layer_ind = layer_ind
        return l0_penalty_sum, nonzero_params

    def __backward(self, loss: torch.Tensor, l0_penalty: torch.Tensor) -> NoReturn:
        if self.gradient_accumulation_steps > 1:
            loss /= self.gradient_accumulation_steps
            l0_penalty /= self.gradient_accumulation_steps

        loss.backward()
        l0_penalty.backward()

    def __calculate_grads(self) -> NoReturn:
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if "classifier" in name:
                # copying updated gradient to the respective w vector
                self.bert_params[name][1].grad.copy_(param.grad.data)
            else:
                # adding grad to w
                self.bert_params[name][1].grad.copy_(param.grad.data *
                                                     self.grad_params[name][1].data)
                # adding grad to base alpha
                self.bert_params[name][2].grad.copy_(param.grad.data *
                                                     self.grad_params[name][0].data *
                                                     self.grad_params[name][2].data)
                # adding grad to per param / per layer alpha
                if self.pp_alpha:
                    self.per_params_alpha[name].grad.copy_(torch.sum(param.grad.data *
                                                                     self.grad_params[name][3].data *
                                                                     self.per_params_z_grad[name].data))
                elif self.pl_alpha:
                    layer_ind = self._get_layer_ind(name)
                    self.per_layer_alpha[layer_ind] += torch.sum(param.grad.data *
                                                                 self.grad_params[name][3].data *
                                                                 self.per_layer_z_grad[layer_ind].data)

    def __clip_grad_norms(self) -> NoReturn:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.diff, self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.alpha, self.max_grad_norm)

    def __step(self) -> NoReturn:
        self.optimizer_params.step()
        self.optimizer_alpha.step()
        self.scheduler_params.step()
        self.scheduler_alpha.step()

        self.model.zero_grad()

    def __log_epoch(self, epoch: int, update_steps: int, train_losses: list,
                    val_losses: list, nonzero_params: list,
                    l0_penalties: list, norm: bool = False) -> NoReturn:
        string = f"Epoch {epoch + 1} | Update step {update_steps} average values:" +\
            f"\n\ttrain loss = {np.array(train_losses).mean():.3f} " +\
            f"val loss = {np.array(val_losses).mean():.3f}" +\
            f"\n\tnonzero params = {np.array(nonzero_params).mean():,.0f}" +\
            f"\n\tL0 penalty = {np.array(l0_penalties).mean():,.3f}"
        if norm:
            norms = self._calculate_param_norms()
            string += f"\n\tnorms = {np.array(norms).mean(axis=0)}"
        self.logger.info(string)

    def _calculate_param_norms(self, order: int = 2) -> List[List[float]]:
        norms = []
        for name, _ in self.model.named_parameters():
            bert, diff, alpha = self.bert_params[name]
            norms.append([torch.linalg.norm(bert, ord=order).item(),
                          torch.linalg.norm(diff, ord=order).item(),
                          torch.linalg.norm(alpha, ord=order).item()
                          ])
        return norms

    def forward(self, batch: list):
        """
        Computes a forward pass through the tuned model
        Parameters:
            batch: list
                [input_ids, attention_mask, token_type_ids, labels]
        Returns:
            output
                transformers model output
        """
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
        print(type(output))
        return output

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Parameters:
            dataloader: torch.utils.data.Dataloader
                data used for model evaluation
        Returns:
            avg_loss: float
                average loss across the data
        """
        total_loss = 0.
        self.model.eval()
        with torch.no_grad():
            epoch_iterator = tqdm(dataloader, desc="Val iteration", position=0, leave=True)
            for _, batch in enumerate(epoch_iterator):
                for name, param in self.model.named_parameters():
                    if "classifier" in name:
                        param.data.copy_(self.bert_params[name][0].data + self.bert_params[name][1].data)
                    else:
                        layer_ind = self._get_layer_ind(name)
                        z = self.concrete_stretched(self.bert_params[name][2], return_grad=False)
                        if self.pp_alpha:  # per params alpha
                            z2 = self.concrete_stretched(self.per_params_alpha[name], return_grad=False)
                        elif self.pl_alpha:  # per layer alpha
                            z2 = self.concrete_stretched(self.per_layer_alpha[layer_ind], return_grad=False)
                        else:  # only base alpha
                            z2 = torch.ones_like(z.data)

                        param.data.copy_(self.bert_params[name][0].data +
                                         (z * z2).data * self.bert_params[name][1].data)

                input_ids, attention_mask, token_type_ids, labels = batch
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                total_loss += output.loss.item()
        return total_loss / len(dataloader)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int) -> NoReturn:
        """
        Parameters:
            train_dataloader: torch.utils.data.Dataloader
                data used for training
            val_dataloader: torch.utils.data.Dataloader
                data used for evaluation
            epochs: int
                number of training epochs
        """
        self.__prepare_scheduler(train_dataloader, epochs)
        self.model.zero_grad()
        train_losses = []
        l0_penalties = []
        val_losses = []
        nonzero = []
        update_steps = 0

        train_iterator = trange(0, epochs, desc="Epoch")
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Train iteration", position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                l0_penalty_sum, nonzero_params = self.__calculate_params()

                # forward pass
                self.model.train()
                output = self.forward(batch)
                loss = output.loss
                train_losses.append(loss.item())
                l0_penalties.append(l0_penalty_sum.item())

                self.__backward(loss, l0_penalty_sum)

                if (step + 1) % self.gradient_accumulation_steps == 0 or (
                        # if last step in epoch and num of batches is smaller than accum steps
                        (step + 1) == len(epoch_iterator)
                        and self.gradient_accumulation_steps > len(epoch_iterator)
                ):
                    self.__calculate_grads()
                    self.__clip_grad_norms()
                    self.__step()
                    update_steps += 1
                    nonzero.append(nonzero_params)

                    avg_val_loss = self.evaluate(val_dataloader)
                    val_losses.append(avg_val_loss)

            # logging epoch
            self.__log_epoch(epoch, update_steps, train_losses,
                             val_losses, nonzero, l0_penalties, norm=False)

            train_losses = []
            l0_penalties = []
            val_losses = []
