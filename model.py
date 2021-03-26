from copy import deepcopy
import numpy as np
from typing import Tuple, List, NoReturn
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm, trange

NO_DECAY = ["bias", "LayerNorm.weight"]


class ConcreteStretched:
    def __init__(self, l: float, r: float, device="cuda"):
        """
       Parameters:
           alpha: torch.Tensor
               Bernoulli distribution parameters (probabilities)
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
    def __init__(self, model: nn.Module, concrete_lower: float, concrete_upper: float,
                 epochs: int, weight_decay: float,
                 warmup_steps: int, training_steps: int, gradient_accumulation_steps: int,
                 lr_params: float = 0.001, lr_alpha: float = 0.1, eps: float = 1e-8,
                 alpha_init: torch.Tensor = None, device: str = "cuda"):
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
        self.device = device
        self.epochs = epochs
        self.model = model.to(device)  # BertForSequenceClassification
        self.alpha_init = alpha_init
        self.log_ratio = self._get_log_ratio(concrete_lower, concrete_upper)
        self.concrete_stretched = ConcreteStretched(concrete_lower, concrete_upper)

        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.pretrained = []
        self.diff = []
        self.alpha = []
        self.all_alpha = None
        self.all_diff = None
        self.optimiser_grouped_parameters = []
        self._get_params()

        self.optimiser_params = AdamW(self.optimiser_grouped_parameters, lr=lr_params, eps=eps)
        self.optimiser_alpha = AdamW(self.alpha, lr=lr_alpha, eps=eps)

        self.scheduler_params = get_linear_schedule_with_warmup(
            self.optimiser_params, num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
        self.scheduler_alpha = get_linear_schedule_with_warmup(
            self.optimiser_alpha, num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )

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
        # TODO: PER PARAMS ALPHA
        all_alpha, all_diff = [], []
        decay, no_decay = {'params': [], 'weight_decay': self.weight_decay}, \
                          {'params': [], 'weight_decay': 0.}
        for name, param in self.model.named_parameters():
            pretrained = deepcopy(param.data)
            diff = torch.zeros_like(param.data, requires_grad=True)
            alpha = torch.zeros_like(param.data, requires_grad=True)
            if self.alpha_init is not None:
                alpha += self.alpha_init

            diff.grad = torch.zeros_like(diff)  ###
            alpha.grad = torch.zeros_like(alpha)  ###

            if name in NO_DECAY:
                no_decay['params'].append(diff)
            else:
                decay['params'].append(diff)
            self.pretrained.append(pretrained)
            self.diff.append(diff)
            ### PER LAYER ALPHA ###
            self.alpha.append(alpha)

            if "classifier" not in name:
                all_alpha.append(alpha.reshape(-1)) ###
                all_diff.append(diff.reshape(-1)) ###

        self.all_alpha = torch.cat(all_alpha, dim=0)
        self.all_add = torch.cat(all_diff, dim=0)
        self.optimiser_grouped_parameters.extend([decay, no_decay])

    def train(self, train_dataloader: DataLoader):
        self.model.zero_grad()
        self.model.train()

        train_iterator = trange(0, int(self.epochs), desc="Epoch")
        for _ in train_iterator:
            l0_penalty = 0

            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                nonzero_params = 0
                l0_penalty += torch.sigmoid(self.all_alpha - self.log_ratio).sum()
                # TODO: PER PARAMS ALPHA

                input_ids, attention_mask, token_type_ids, labels = batch
                output = self.model(input_ids, attention_mask, token_type_ids, labels)
                loss = output.loss

                if self.gradient_accumulation_steps > 1:
                    loss /= self.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    pass

                # z, z_grad = self.concrete_stretched()
                # z, z_grad = one_pass_concrete_stretched(all_alpha, args.concrete_lower,
                #                                         args.concrete_upper)
                # z2_, z2_grad_ = one_pass_concrete_stretched(all_pp_alpha, args.concrete_lower, args.concrete_upper)
