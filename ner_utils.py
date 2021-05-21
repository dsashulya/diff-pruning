from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from collections import defaultdict
from transformers import PreTrainedTokenizer
from data import tokens_to_dataloader
from seqeval.metrics import precision_score, recall_score, f1_score


@dataclass
class Example:
    guid: int
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


def get_labels(path: str) -> List[str]:
    labels = []
    with open(path, 'r') as file:
        for line in file:
            if line != '\n' or line != '':
                tokens = line.strip().split('\t')
                if tokens[-1] != '':
                    labels.append(tokens[-1])
    return np.unique(labels).tolist()


def read_data(path: str) -> List[Example]:
    examples = []
    with open(path, 'r') as file:
        ind = 0
        words = []
        labels = []
        for line in file:
            if line == '\n' or line == '':
                if words:
                    examples.append(Example(guid=ind, words=words, labels=labels))
                    ind += 1
                    words, labels = [], []
            else:
                tokens = line.strip().split('\t')
                words.append(tokens[0])
                if len(tokens) > 1 and tokens[1] != '':
                    labels.append(tokens[1])
                else:
                    labels.append("O")
        if words:
            examples.append(Example(guid=ind, words=words, labels=labels))
    return examples


def convert_to_features(examples: List[Example],
                        label_list: List[str],
                        tokenizer: PreTrainedTokenizer,
                        max_seq_length: int = 512,
                        pad_token_label_id: int = -100,
                        sep_token: str = "[SEP]",
                        sequence_a_segment_id: int = 0,
                        cls_token: str = "[CLS]",
                        cls_token_segment_id: int = 1,
                        pad_token: int = 0,
                        pad_token_segment_id: int = 0) -> dict:
    label_map = {label: i for i, label in enumerate(label_list)}
    features = defaultdict(list)
    for i, example in enumerate(tqdm(examples, position=0, leave=True, desc="Preparing data")):
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if "token_type_ids" not in tokenizer.model_input_names:
            token_type_ids = None

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(input_mask)
        features['token_type_ids'].append(token_type_ids)
        features['labels'].append(label_ids)

    return features


def get_bc2gm_train_data(args, tokenizer, return_val=True):
    train_data = read_data(args.path_to_train)
    labels = get_labels(args.path_to_train)

    train_features = convert_to_features(train_data, labels, tokenizer)
    train_dataloader = tokens_to_dataloader(train_features, args.batch_size, args.task_name)

    val_dataloader = None
    if return_val:
        val_data = read_data(args.path_to_val)
        val_features = convert_to_features(val_data, labels, tokenizer)
        val_dataloader = tokens_to_dataloader(val_features, args.batch_size, args.task_name)
    return train_dataloader, val_dataloader


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: dict):
    pred = np.argmax(predictions, axis=2)
    batch_size, seq_len = pred.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[pred[i][j]])
    return preds_list, out_label_list


@torch.no_grad()
def evaluate_ner_metrics(model, dataloader, label_map):
    pr, rec, f1 = 0., 0., 0.
    for batch in tqdm(dataloader, position=0, leave=True, desc="Evaluating metrics"):
        outputs = model.forward([item.to(model.device) for item in batch])
        pred, labels = align_predictions(outputs.logits.detach().cpu().numpy(), batch[-1].cpu().numpy(), label_map)
        pr += precision_score(labels, pred, zero_division=0)
        rec += recall_score(labels, pred, zero_division=0)
        f1 += f1_score(labels, pred, zero_division=0)

    return {
        "precision": pr / len(dataloader),
        "recall": rec / len(dataloader),
        "f1": f1 / len(dataloader)
    }
