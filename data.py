import torch
from torch.utils.data import TensorDataset, DistributedSampler, DataLoader
import torch.distributed as dist


def tokens_to_dataloader(tokenized_examples, batch_size, task_name='squad', shuffle=True):
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

    distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None and shuffle)
