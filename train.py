import argparse
import logging

import numpy as np
import os
import random
import torch
import torch.multiprocessing as mp
from datasets import load_dataset, load_metric
from torch import distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer,
    BertForQuestionAnswering, BertTokenizer,
    DistilBertForQuestionAnswering, DistilBertTokenizer
)

from model import DiffPruning

MODEL_CLASSES = {
    "bert": {"model": BertForQuestionAnswering, "tokenizer": BertTokenizer},
    "distilbert": {"model": DistilBertForQuestionAnswering, "tokenizer": DistilBertTokenizer}
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
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
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
    parser.add_argument('--max_length', default=384, type=int, required=False)
    parser.add_argument('--doc_stride', default=128, type=int, required=False)
    parser.add_argument('--tokenizer_name', default="bert-base-uncased", type=str, required=False)
    parser.add_argument('--do_lower_case', default=1, type=lambda x: bool(int(x)), required=False)

    # model params
    parser.add_argument("--task_name", default=None, type=str, required=True)
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
    parser.add_argument('--squad_v2', default=0, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--num_train_epochs', default=5, type=int, required=True)
    parser.add_argument('--logging_steps', default=5, type=int, required=False)
    parser.add_argument('--eval_steps', default=5, type=int, required=False)
    parser.add_argument('--write', default=1, type=lambda x: bool(int(x)), required=False,
                        help="Write logs to summary writer")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    return parser


def prepare_dataloader(data, tokenizer, max_length, doc_stride, batch_size):
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        data["question" if pad_on_right else "context"],
        data["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_token_type_ids=True
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    offset_iterator = tqdm(offset_mapping, desc="Preparing data", position=0, leave=True)
    for i, offsets in enumerate(offset_iterator):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = data["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append([cls_index])
                tokenized_examples["end_positions"].append([cls_index])
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append([token_start_index - 1])
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append([token_end_index + 1])

    dataset = TensorDataset(
        torch.tensor(tokenized_examples["input_ids"], dtype=torch.long),
        torch.tensor(tokenized_examples["attention_mask"], dtype=torch.long),
        torch.tensor(tokenized_examples["token_type_ids"], dtype=torch.long),
        torch.tensor(tokenized_examples["start_positions"], dtype=torch.long),
        torch.tensor(tokenized_examples["end_positions"], dtype=torch.long)
    )
    distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None)


def get_data(args, tokenizer, return_val=True):
    datasets = load_dataset("squad_v2" if args.squad_v2 else "squad")
    train_dataloader = prepare_dataloader(data=datasets["train"][:87_599],
                                          tokenizer=tokenizer,
                                          max_length=args.max_length,
                                          doc_stride=args.doc_stride,
                                          batch_size=args.batch_size)
    val_dataloader = None
    if return_val:
        val_dataloader = prepare_dataloader(data=datasets["validation"][:10_570],
                                            tokenizer=tokenizer,
                                            max_length=args.max_length,
                                            doc_stride=args.doc_stride,
                                            batch_size=args.batch_size)
    return train_dataloader, val_dataloader


def train(local_rank, args):
    filename = f"log_{args.task_name}_" + "diff.txt" if not args.no_diff else "no_diff.txt"
    logging.basicConfig(filename=filename, filemode='a', level=args.logging_level)

    if args.local_rank != -1:
        setup_distributed(local_rank, args.world_size)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # model_class, tokenizer_class = MODEL_CLASSES[args.model_name].values()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    train_dataloader, val_dataloader = get_data(args=args,
                                                tokenizer=tokenizer,
                                                return_val=True)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank == 0:
        torch.distributed.barrier()

    diff_model = DiffPruning(model=model,
                             task_name=args.task_name,
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
                     save_steps=args.save_steps,
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
