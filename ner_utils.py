from pathlib import Path
from typing import List, NamedTuple, Dict, Any

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data import tokens_to_dataloader

PAD_TAG = 'PAD'
START_TAG = 'START'
END_TAG = 'END'
UNK_TAG = 'UNK'
OTHER_TAG = 'O'

UTIL_TAGS = [PAD_TAG, START_TAG, END_TAG, UNK_TAG, OTHER_TAG]
ALT_ENTITIES = {'PROTEIN': 'GENE', 'CL': 'CELL_TYPE', 'CHEBI': 'CHEMICAL', 'GGP': 'GENE', 'SPECIES': 'TAXON',
                'CELLLINE': 'CELL_LINE'}
TAB = '\t'


class NerDatasetItem(NamedTuple):
    input_ids: List[int]
    labels: List[int]


class RawNerItem(NamedTuple):
    words: List[str]
    tags: List[str]

    def ner_format(self, separator: str = ' ') -> str:
        res = ''
        for i, word in enumerate(self.words):
            res += f'{word}{separator}{self.tags[i]}\n'
        return res


def build_dict(tags: List[str], prefixes, map_to_same_id: bool = False):
    result = {}
    for tag in tags:
        next_id = max(result.values(), default=-1) + 1
        if tag in UTIL_TAGS:
            result[tag] = next_id
        else:
            for prefix in prefixes:
                result[prefix + tag] = next_id
                if not map_to_same_id:
                    next_id = max(result.values(), default=-1) + 1
    return result


def encode_words(words: List[str], tokenizer: PreTrainedTokenizer, cache=None):
    if cache is None:
        cache = {}
    encoded_words = []
    for word in words:
        word_to_tokenize = word.strip().lower()

        if word_to_tokenize in cache:
            word_ids = cache[word_to_tokenize]
        else:
            word_ids = tokenizer.encode(word_to_tokenize, add_special_tokens=False)
            cache[word_to_tokenize] = word_ids

        encoded_words.append(word_ids)
    return encoded_words


def encode_tags(tags: List[str], tags_vocab):
    other_tag_id = tags_vocab[OTHER_TAG]

    encoded_tags = []
    for tag in tags:
        tag = tag.upper()
        if tag in tags_vocab:
            encoded_tags.append(tags_vocab[tag])
        else:
            encoded_tags.append(other_tag_id)
    return encoded_tags


def align_word_ids_with_tag_ids(encoded_words, encoded_tags, unk_tag_id: int,
                                alignment='SAME'):
    assert len(encoded_words) == len(encoded_tags)

    encoded_text = []
    aligned_tags = []
    for i, word_ids in enumerate(encoded_words):
        cur_tag = encoded_tags[i]

        if alignment == 'SAME':
            cur_tags = [cur_tag] * len(word_ids)
        elif alignment == 'UNK':
            cur_tags = [cur_tag] + [unk_tag_id] * (len(word_ids) - 1)
        else:
            raise Exception(f'Unsupported NerTagAlignment: {alignment}')

        encoded_text.extend(word_ids)
        aligned_tags.extend(cur_tags)
    return encoded_text, aligned_tags


def encode_raw_ner_item(
        raw_ner_item, tokenizer: PreTrainedTokenizer, cutoff: int,
        tags_vocab, alignment='SAME'):
    pad_tag_id, unk_tag_id = tags_vocab[PAD_TAG], tags_vocab[UNK_TAG]
    encoded_words = encode_words(raw_ner_item.words, tokenizer)
    encoded_tags = encode_tags(raw_ner_item.tags, tags_vocab)

    text_ids, aligned_tags_ids = \
        align_word_ids_with_tag_ids(encoded_words, encoded_tags, unk_tag_id=unk_tag_id, alignment=alignment)

    text_ids = text_ids[:cutoff]
    aligned_tags_ids = aligned_tags_ids[:cutoff]

    result_len = len(text_ids)
    if result_len < cutoff:
        text_ids.extend([tokenizer.pad_token_id] * (cutoff - result_len))
        aligned_tags_ids.extend([pad_tag_id] * (cutoff - result_len))

    return NerDatasetItem(input_ids=text_ids, labels=aligned_tags_ids)


class NerDataset(Dataset):
    items: List[NerDatasetItem]

    def __init__(self, items: List[NerDatasetItem]):
        super(NerDataset, self).__init__()
        self.items = items

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            'input_ids': torch.tensor(item.input_ids, dtype=torch.long),
            'labels': torch.tensor(item.labels, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.items)

    def class_probs(self, num_classes: int) -> List[float]:
        counts = [0] * num_classes
        for i in range(len(self)):
            item = self[i]
            for tag in item['labels']:
                counts[tag] += 1

        counts[0] = 0
        counts[1] = len(self)
        counts[2] = len(self)

        token_count = sum(counts)
        return [c / token_count for c in counts]


class BioNerFileFormat:
    @staticmethod
    def deserialize(lines: List[str]) -> List[RawNerItem]:
        result: List[RawNerItem] = []

        words = []
        tags = []
        for line in lines:
            pair = line.strip().split('\t')
            if len(pair) == 2:
                word, tag = pair
                words.append(word)
                if tag[2:].upper() in ALT_ENTITIES:
                    tag = tag[0:2] + ALT_ENTITIES[tag[2:].upper()]
                tags.append(tag)
            else:
                if words:
                    result.append(RawNerItem(words=words, tags=tags))
                    words = []
                    tags = []

        if words:
            result.append(RawNerItem(words=words, tags=tags))
        return result

    @staticmethod
    def serialize(items: List[RawNerItem]) -> List[str]:
        result: List[str] = []

        for item in items:
            for w, t in zip(item.words, item.tags):
                result.append(f'{w}{TAB}{t}')
            result.append('')
        return result


def load_dataset(path, tokenizer: PreTrainedTokenizer, tags_vocab,
                 cutoff: int = 200) -> NerDataset:
    raw_items = BioNerFileFormat.deserialize(path.read_text().splitlines())
    encoded_items: List[NerDatasetItem] = []
    for item in tqdm(raw_items):
        encoded_item = encode_raw_ner_item(item, tokenizer=tokenizer, cutoff=cutoff,
                                           tags_vocab=tags_vocab)
        encoded_items.append(encoded_item)
    return NerDataset(encoded_items)

def get_bc2gm_train_data(args, tokenizer, tags_vocab, return_val=True, return_train=True):
    train_dataloader, val_dataloader = None, None
    if return_train:
        train_data = load_dataset(Path(args.path_to_train), tokenizer=tokenizer, tags_vocab=tags_vocab)
        train_dataloader = tokens_to_dataloader(train_data, args.batch_size, 'ner')
    if return_val:
        val_data = load_dataset(Path(args.path_to_val), tokenizer=tokenizer, tags_vocab=tags_vocab)
        val_dataloader = tokens_to_dataloader(val_data, args.batch_size, 'ner')
    return train_dataloader, val_dataloader


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: dict):
    pred = np.argmax(predictions, axis=2)
    batch_size, seq_len = pred.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index and label_map[label_ids[i, j]] != 'PAD':
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[pred[i][j]])
    return preds_list, out_label_list


@torch.no_grad()
def evaluate_ner_metrics(model, dataloader, label_map):
    pr, rec, f1 = 0., 0., 0.
    for batch in tqdm(dataloader, position=0, leave=True, desc="Evaluating metrics"):
        if isinstance(batch, list):
            outputs = model.forward([item.to(model.model.device) for item in batch])
            labels = batch[-1]
        else:
            outputs = model.forward({key: value.to(model.model.device) for key, value in batch.items()})
            labels = batch['labels']
        pred, labels = align_predictions(outputs.logits.detach().cpu().numpy(), labels.cpu().numpy(), label_map)

        pr += precision_score(labels, pred, zero_division=0)
        rec += recall_score(labels, pred, zero_division=0)
        f1 += f1_score(labels, pred, zero_division=0)

    return {
        "precision": pr / len(dataloader),
        "recall": rec / len(dataloader),
        "f1": f1 / len(dataloader)
    }
