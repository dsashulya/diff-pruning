from model import DiffPruning
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
import logging

logging.basicConfig(filename="log.txt", filemode='a', level=20)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_dataloader(data, labels, tokenizer):
    encodings = tokenizer(
        data,
        return_token_type_ids=True,
        max_length=None,
        padding='max_length',
        return_tensors="pt"
    ).to(DEVICE)
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["token_type_ids"],
        torch.tensor(labels, dtype=torch.long).to(DEVICE)
    )
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True)
    return dataloader


def main():
    bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = DiffPruning(bert, "bert", concrete_lower=-1.5, concrete_upper=1.5, total_layers=14,
                        sparsity_penalty=0.000000125, weight_decay=0.0001, alpha_init=5.,
                        lr_params=5e-5, lr_alpha=5e-5,
                        warmup_steps=100, gradient_accumulation_steps=1, max_grad_norm=1, device=DEVICE)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_data = ["This is the first sentence.",
                  "This is the second sentence.",
                  "And this is the final, third sentence."]
    train_labels = [0, 0, 1]
    val_data = ["This is a sentence for validation.",
                "And another one just in case!"]
    val_labels = [0, 1]

    train_dataloader = prepare_dataloader(train_data, train_labels, tokenizer)
    val_dataloader = prepare_dataloader(val_data, val_labels, tokenizer)

    model.train(train_dataloader, val_dataloader, 50)


if __name__ == "__main__":
    # args = parse_args()
    main()
