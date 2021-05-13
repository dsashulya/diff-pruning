import argparse
from model import DiffPruning
from transformers import (
    BertForQuestionAnswering, BertForSequenceClassification, BertTokenizer,
    DistilBertForQuestionAnswering, DistilBertForSequenceClassification, DistilBertTokenizer
)
import torch
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from torch import distributed as dist
import logging
import os
import random
import numpy as np

MODEL_CLASSES = {
    "bert": {"model": BertForSequenceClassification, "tokenizer": BertTokenizer},
    "distilbert": {"model": DistilBertForSequenceClassification, "tokenizer": DistilBertTokenizer}
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=10, type=int, required=False)
    parser.add_argument('--local_rank', default=-1, type=int, required=False)
    parser.add_argument('--world_size', default=1, type=int, required=False)
    parser.add_argument('--n_gpu', default=1, type=int, required=False)
    parser.add_argument('--logging_level', default=20, type=int, required=False)
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str, required=False,
                        help="used in model_class.from_pretrained()")

    # data params
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--tokenizer_name', default="bert-base-uncased", type=str, required=False)
    parser.add_argument('--do_lower_case', default=1, type=lambda x: bool(int(x)), required=False)

    # model params
    parser.add_argument('--model_name', default="bert", type=str, required=False)
    parser.add_argument('--concrete_lower', default=-1.5, type=float, required=False)
    parser.add_argument('--concrete_upper', default=1.5, type=float, required=False)
    parser.add_argument('--total_layers', default=14, type=int, required=False)
    parser.add_argument('--sparsity_penalty', default=1.25e-7, type=float, required=False)
    parser.add_argument('--weight_decay', default=0., type=float, required=False)
    parser.add_argument('--alpha_init', default=5., type=float, required=False)
    parser.add_argument('--lr_params', default=5e-5, type=float, required=False)
    parser.add_argument('--lr_alpha', default=5e-5, type=float, required=False)
    parser.add_argument('--per_layer_alpha', default=0, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--per_params_alpha', default=0, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--warmup_steps', default=0, type=int, required=False)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1., type=float, required=False)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--no_diff', default=0, type=lambda x: bool(int(x)), required=False)

    # train params
    parser.add_argument('--num_train_epochs', default=5, type=int, required=True)
    parser.add_argument('--logging_steps', default=5, type=int, required=False)
    parser.add_argument('--eval_steps', default=5, type=int, required=False)
    parser.add_argument('--write', default=1, type=lambda x: bool(int(x)), required=False,
                        help="Write logs to summary writer")
    return parser


def prepare_dataloader(data, labels, tokenizer, batch_size):
    encodings = tokenizer(
        data,
        return_token_type_ids=True,
        max_length=None,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["token_type_ids"],
        torch.tensor(labels, dtype=torch.long)
    )
    distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(data, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None)


def train(local_rank, args):
    logging.basicConfig(filename="log.txt", filemode='a', level=args.logging_level)

    if args.local_rank != -1:
        setup_distributed(local_rank, args.world_size)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model_class, tokenizer_class = MODEL_CLASSES[args.model_name].values()
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank == 0:
        torch.distributed.barrier()

    train_data = ["This is the first sentence.",
                  "This is the second sentence.",
                  "And this is the final, third sentence."]
    train_labels = [0, 0, 1]
    val_data = ["This is a sentence for validation.",
                "And another one just in case!"]
    val_labels = [0, 1]

    train_dataloader = prepare_dataloader(train_data, train_labels, tokenizer, args.batch_size)
    val_dataloader = prepare_dataloader(val_data, val_labels, tokenizer, args.batch_size)

    diff_model = DiffPruning(model=model,
                             model_name=args.model_name,
                             concrete_lower=args.concrete_lower,
                             concrete_upper=args.concrete_upper,
                             total_layers=args.total_layers,
                             sparsity_penalty=args.sparsity_penalty,
                             weight_decay=args.weight_decay,
                             alpha_init=args.alpha_init,
                             lr_params=args.lr_params,
                             lr_alpha=args.lr_alpha,
                             per_layer_alpha=args.per_layer_alpha,
                             per_params_alpha=args.per_params_alpha,
                             warmup_steps=args.warmup_steps,
                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                             max_grad_norm=args.max_grad_norm,
                             local_rank=args.local_rank,
                             no_diff=args.no_diff)
    set_seed(args)
    diff_model.train(local_rank=local_rank,
                     train_dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     epochs=args.num_train_epochs,
                     max_steps=args.max_steps,
                     logging_steps=args.logging_steps,
                     write=args.write)


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    if args.local_rank != -1:
        mp.spawn(train,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        train(args.local_rank, args)
