from model import DiffPruning
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch import distributed as distrib
import logging


logging.basicConfig(filename="log.txt", filemode='a', level=20)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
DISTRIBUTED = True
LOCAL_RANK = 0 if DISTRIBUTED else -1
OPT_LEVEL = 'O1'


def prepare_dataloader(data, labels, tokenizer, batch_size):
    encodings = tokenizer(
        data,
        return_token_type_ids=True,
        max_length=None,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    ).to(DEVICE)
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["token_type_ids"],
        torch.tensor(labels, dtype=torch.long).to(DEVICE)
    )
    distributed = distrib.is_available() and distrib.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(data, num_replicas=distrib.get_world_size(), rank=distrib.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None)


def main():
    if DISTRIBUTED:
        torch.cuda.set_device(0)
        distrib.init_process_group("nccl")

    bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = DiffPruning(bert, "bert", concrete_lower=-1.5, concrete_upper=1.5, total_layers=14,
                        sparsity_penalty=1.25e-2, weight_decay=0.0001, alpha_init=5.,
                        lr_params=5e-5, lr_alpha=1e-2, per_layer_alpha=False,
                        warmup_steps=0, gradient_accumulation_steps=1, max_grad_norm=1, device=DEVICE)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_data = ["This is the first sentence.",
                  "This is the second sentence.",
                  "And this is the final, third sentence."]
    train_labels = [0, 0, 1]
    val_data = ["This is a sentence for validation.",
                "And another one just in case!"]
    val_labels = [0, 1]

    train_dataloader = prepare_dataloader(train_data, train_labels, tokenizer, BATCH_SIZE)
    val_dataloader = prepare_dataloader(val_data, val_labels, tokenizer, BATCH_SIZE)

    model.train(train_dataloader, val_dataloader, 30, write=False)


if __name__ == "__main__":
    main()
