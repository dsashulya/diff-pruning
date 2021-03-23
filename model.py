from copy import deepcopy
import numpy as np
from typing import Tuple, List, NoReturn
import torch
from torch import nn
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer
)

NO_DECAY = ["bias", "LayerNorm.weight"]


class DiffPruning:
    def __init__(self, model: nn.Module, l: float, r: float, weight_decay: float,
                 lr_params: float, lr_alpha: float = 0.1, eps: float = 1e-8,
                 alpha_init: torch.Tensor = None, device: str = "cuda"):
        """
        Parameters:
            model: nn.Module
                transformers model with pretrained parameters
            l: float
               constant used to stretch s (l < 0)
           r: float
               constant used to stretch s (r > 1)
            alpha_init: torch.Tensor
                initial values of alpha
        """
        self.device = device
        self.model = model  # BertForSequenceClassification
        self.alpha_init = alpha_init
        self.weight_decay = weight_decay
        self.pretrained = []
        self.diff = []
        self.alpha = []
        self.optimiser_grouped_parameters = []
        self.get_params()

        self.optimiser_params = AdamW(self.optimiser_grouped_parameters, lr=lr_params, eps=eps)
        self.optimiser_alpha = AdamW(self.alpha, lr=lr_alpha, eps=eps)

        ### SCHEDULER ###

    def get_params(self) -> NoReturn:
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
            self.alpha.append(alpha)
        self.optimiser_grouped_parameters.extend([decay, no_decay])


class ConcreteStretched:
    def __init__(self, alpha: torch.Tensor, l: float, r: float, device="cuda"):
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
        self.alpha = alpha
        self.l = l
        self.r = r
        self.dz_dalpha = None

    def __call__(self) -> torch.Tensor:
        """
        Returns:
           z: torch.Tensor
               binary mask vector relaxed into continuous space
       """
        u = torch.rand_like(self.alpha).clamp(1e-8, 1)
        s = (torch.sigmoid(u.log() - (1 - u).log() + self.alpha))
        s_ = s * (self.r - self.l) + self.l
        t = torch.max(torch.zeros_like(s_), s_)
        z = torch.min(torch.ones_like(s_), t)

        # dz/da = dz/dt * dt/ds_ * ds_/ds * ds/da
        dz_dt = (t < 1).float()
        dt_ds_ = (s_ > 0).float()
        ds__ds = self.r - self.l
        ds_dalpha = s * (1 - s)
        self.dz_dalpha = dz_dt * dt_ds_ * ds__ds * ds_dalpha

        return z

    def backward(self, grad: torch.Tensor = None) -> torch.Tensor:
        assert self.dz_dalpha is not None, "Do a forward pass first"
        if grad is not None:
            return self.dz_dalpha * grad
        return self.dz_dalpha
