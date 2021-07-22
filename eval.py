from typing import Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from loss import loss as ner_loss
from ner_utils import masked_softmax
from tags import get_utils_tags_ids
from data import get_ner_model_inputs


def predict(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = masked_softmax(logits, mask)
    scores = scores * mask.unsqueeze(2).float()
    path = torch.max(scores, 2)[1]
    path = path * mask.long()
    return path


def update_confusion_matrix(conf_matrix: Tensor, actual: Tensor, predicted: Tensor, *, mask: Optional[Tensor] = None):
    if mask is None:
        mask = torch.ones_like(actual, dtype=torch.float32)

    index = predicted * conf_matrix.size(1) + actual
    conf_matrix.view(-1).index_add_(0, index, mask.float())


def f1_score_micro_precision_recall(conf_matrix: Tensor, *, start: int = 0, end: Optional[int] = None) \
        -> Tuple[Tensor, Tensor, Tensor]:
    score, precision, recall = 0., 0., 0.
    pr, p, r = 0., 0., 0.
    for tag in range(start, conf_matrix.size(0) if end is None else end):
        pr += conf_matrix[tag, tag].item()
        p += torch.sum(conf_matrix[tag, :]).item()
        r += torch.sum(conf_matrix[:, tag]).item()
    try:
        score = 2 * pr / (p + r)
    except ZeroDivisionError:
        pass

    try:
        precision = pr / r
    except ZeroDivisionError:
        pass

    try:
        recall = pr / p
    except ZeroDivisionError:
        pass
    return torch.tensor(score), torch.tensor(precision), torch.tensor(recall)


@torch.no_grad()
def evaluate_ner_metrics(model, dataloader, label_map, tokenizer):
    tags_vocab = {value: key for key, value in label_map.items()}
    tags_num = len(tags_vocab)
    skip_tags = len(get_utils_tags_ids(tags_vocab))
    conf_matrix = torch.zeros((tags_num, tags_num), dtype=torch.float)
    total_loss = 0.
    for batch in tqdm(dataloader, position=0, leave=True, desc="Validation"):
        batch = get_ner_model_inputs(batch, tokenizer, tags_vocab)
        outputs = model(**{key: value.to(model.device) for key, value in batch.items()})
        total_loss += ner_loss(outputs.logits.cpu(), batch['labels'],
                               batch['attention_mask'])
        predictions = predict(outputs.logits.cpu(), batch['attention_mask'])
        update_confusion_matrix(conf_matrix,
                                actual=batch['labels'].reshape(-1), predicted=predictions.reshape(-1))

    f1, pr, rec = f1_score_micro_precision_recall(conf_matrix, start=skip_tags)
    return total_loss / len(dataloader), {
        "precision": pr.item(),
        "recall": rec.item(),
        "f1": f1.item()
    }
