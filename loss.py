import torch
import torch.nn.functional as F
from torch import Tensor

from ner_utils import create_mask

def batch_cross_entropy_tensor(out: torch.Tensor, expected: torch.Tensor, exp_lens: torch.Tensor, mask_delta=0,
                               eps: float = 0.) -> torch.Tensor:
    loss = cross_entropy_label_smoothing(out, expected, eps)
    return apply_batch_mask_tensor(loss, exp_lens, mask_delta)


def apply_batch_mask_tensor(loss: torch.Tensor, exp_lens: torch.Tensor, mask_delta=0) -> torch.Tensor:
    mask = create_mask(loss, exp_lens + torch.tensor(mask_delta, device=exp_lens.device))
    mask.requires_grad = False
    mask = mask.to(dtype=bool)
    loss.masked_fill_(~mask, 0.0)
    return loss


def cross_entropy_label_smoothing(out: Tensor, expected: Tensor, eps: float = 0.) -> Tensor:
    n_classes = out.size(-1)
    if eps == 0.:
        loss = cross_entropy(out, expected)
    else:
        smoothed_targets = smooth_one_hot(expected.view(-1, 1), classes=n_classes, smoothing=eps)
        loss = soft_cross_entropy(out.view(-1, n_classes), smoothed_targets)
        loss = loss.view(expected.shape)
    return loss


def smooth_one_hot(target: torch.Tensor, classes: int, smoothing=0.0):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((target.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=target.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, target, confidence)
    return true_dist


def cross_entropy(out: Tensor, expected: Tensor) -> Tensor:
    return F.cross_entropy(
        out.view(-1, out.shape[-1]),
        expected.view(-1),
        reduction='none').view(expected.shape)


def soft_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         target: targets, can be soft

    Examples::

        pred = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        loss = cross_entropy(pred, target)
    """
    pred_log_soft = F.log_softmax(pred, dim=-1)
    loss_tensor = -target * pred_log_soft
    return torch.sum(loss_tensor, dim=-1)


def loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    scores = logits * mask.unsqueeze(2).float()
    lengths = mask.sum(1).int()
    loss_tensor = batch_cross_entropy_tensor(scores, labels, lengths, eps=0.1)
    return (loss_tensor.sum() / lengths.float().sum()).mean()