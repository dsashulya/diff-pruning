from typing import NamedTuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from tags import *


class NerDatasetItem(NamedTuple):
    input_ids: List[int]
    labels: List[int]
    len: int


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
        tags_vocab, cache=None, alignment='SAME'):
    pad_tag_id, unk_tag_id = tags_vocab[PAD_TAG], tags_vocab[UNK_TAG]
    encoded_words = encode_words(raw_ner_item.words, tokenizer, cache=cache)
    encoded_tags = encode_tags(raw_ner_item.tags, tags_vocab)

    text_ids, aligned_tags_ids = \
        align_word_ids_with_tag_ids(encoded_words, encoded_tags, unk_tag_id=unk_tag_id, alignment=alignment)

    text_ids = text_ids[:cutoff]
    aligned_tags_ids = aligned_tags_ids[:cutoff]

    result_len = len(text_ids)
    if result_len < cutoff:
        text_ids.extend([tokenizer.pad_token_id] * (cutoff - result_len))
        aligned_tags_ids.extend([pad_tag_id] * (cutoff - result_len))

    return NerDatasetItem(input_ids=text_ids, labels=aligned_tags_ids, len=result_len)


class NerDataset(Dataset):
    items: List[NerDatasetItem]

    def __init__(self, items: List[NerDatasetItem]):
        super(NerDataset, self).__init__()
        self.items = items

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        return {
            'input_ids': torch.tensor(item.input_ids, dtype=torch.long),
            'labels': torch.tensor(item.labels, dtype=torch.long),
            'len': item.len
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


def masked_softmax(inp: Tensor, mask: Tensor) -> Tensor:
    masked_input = inp * mask.unsqueeze(2).float()
    scores = F.softmax(masked_input, dim=2)
    return scores


def create_mask(target: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(target.size(1), dtype=lens.dtype, device=lens.device).repeat(target.size(0),
                                                                                     1) < lens.unsqueeze(1)
    for i in range(len(target.size()) - 2):
        mask = mask.unsqueeze(-1)
    return mask.to(dtype=torch.bool)
