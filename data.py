import torch
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
import torch.distributed as distrib
from transformers import PreTrainedTokenizer
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from tags import START_TAG, END_TAG
from ner_utils import create_mask, NerDataset, BioNerFileFormat, NerDatasetItem, encode_raw_ner_item
from tqdm import tqdm


def load_dataset(path, tokenizer: PreTrainedTokenizer, tags_vocab,
                 cutoff: int = 200) -> NerDataset:
    raw_items = BioNerFileFormat.deserialize(path.read_text().splitlines())
    encoded_items: List[NerDatasetItem] = []
    cache = {}
    for i, item in enumerate(tqdm(raw_items)):
        encoded_item = encode_raw_ner_item(item, tokenizer=tokenizer, cutoff=cutoff,
                                           tags_vocab=tags_vocab, cache=cache)
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


def tokens_to_dataloader(tokenized_examples, batch_size, task_name='squad', shuffle=False):
    if task_name == 'squad':
        dataset = TensorDataset(
            torch.tensor(tokenized_examples["input_ids"], dtype=torch.long),
            torch.tensor(tokenized_examples["attention_mask"], dtype=torch.long),
            torch.tensor(tokenized_examples["token_type_ids"], dtype=torch.long),
            torch.tensor(tokenized_examples["start_positions"], dtype=torch.long),
            torch.tensor(tokenized_examples["end_positions"], dtype=torch.long)
        )
    else:
        dataset = tokenized_examples

    distributed = distrib.is_available() and distrib.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=distrib.get_world_size(), rank=distrib.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None and shuffle)


def insert_bounds(seqs: torch.Tensor, lens: torch.Tensor,
                  start_code: Optional[int] = None, end_code: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    change = 0
    start = 0
    if start_code is not None:
        change += 1
        start = 1
    if end_code is not None:
        change += 1

    new_shape = (seqs.shape[0], seqs.shape[1] + change)
    new_seqs = torch.zeros(new_shape, dtype=seqs.dtype, device=seqs.device)
    new_lens = lens.clone()

    new_seqs[:, start:start + seqs.shape[1]] = seqs
    new_lens[:] += change

    if start_code is not None:
        new_seqs[:, 0] = start_code
    if end_code is not None:
        new_seqs[range(lens.shape[0]), lens + change - 1] = end_code

    return new_seqs, new_lens


class SequenceBatch:
    seqs: torch.Tensor
    lens: torch.Tensor

    def __init__(self, seqs: torch.Tensor, lens: torch.Tensor) -> None:
        if torch.numel(lens) > 0:
            max_len = lens.max().item()
            seqs = seqs[:, :max_len]
        self.seqs = seqs
        self.lens = lens

    def insert_bounds(self, start_code: Optional[int] = None, end_code: Optional[int] = None) -> 'SequenceBatch':
        new_seqs, new_lens = insert_bounds(self.seqs, self.lens, start_code=start_code, end_code=end_code)
        return SequenceBatch(new_seqs, new_lens)

    def to(self, device: torch.device) -> 'SequenceBatch':
        return SequenceBatch(self.seqs.to(device), self.lens.to(device))

    def __len__(self):
        return self.lens.shape[0]

    @property
    def device(self):
        return self.seqs.device


def parse_ner_batch(batch: Dict[str, Any], tokenizer: PreTrainedTokenizer,
                    tags_dict: Dict[str, int]) -> Tuple[SequenceBatch, SequenceBatch]:
    tensor = SequenceBatch(batch['input_ids'], batch['len'])
    targets = SequenceBatch(batch['labels'], batch['len'])
    inp_seq = tensor.insert_bounds(start_code=tokenizer.cls_token_id, end_code=tokenizer.sep_token_id)
    out_seq = targets.insert_bounds(start_code=tags_dict[START_TAG], end_code=tags_dict[END_TAG])
    return inp_seq, out_seq


def get_ner_model_inputs(batch: Dict[str, Any], tokenizer: PreTrainedTokenizer,
                         tags_vocab: Dict) -> Dict[str, Any]:
    in_, out_ = parse_ner_batch(batch, tokenizer, tags_vocab)
    return {
        'input_ids': in_.seqs,
        'labels': out_.seqs,
        'attention_mask': create_mask(in_.seqs, in_.lens)
    }
