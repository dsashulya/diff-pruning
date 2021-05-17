import argparse
import logging
import collections

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


def init_model(args, model):
    return DiffPruning(model=model,
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
    parser.add_argument('--do_train', default=True, type=lambda x: bool(int(x)), required=True)
    parser.add_argument('--do_eval', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--local_rank', default=-1, type=int, required=False)
    parser.add_argument('--world_size', default=1, type=int, required=False)
    parser.add_argument('--n_gpu', default=1, type=int, required=False)
    parser.add_argument('--logging_level', default=20, type=int, required=False)
    parser.add_argument('--model_name_or_path', default='bert-base-uncased', type=str, required=True,
                        help="used in model_class.from_pretrained()")
    parser.add_argument('--model_checkpoint', default='', type=str, required=False,
                        help="checkpoint to load the model from")

    # data params
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--max_length', default=384, type=int, required=False)
    parser.add_argument('--doc_stride', default=128, type=int, required=False)
    parser.add_argument('--tokenizer_name', default="bert-base-uncased", type=str, required=True)
    parser.add_argument('--do_lower_case', default=True, type=lambda x: bool(int(x)), required=False)

    # model params
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument('--model_name', default="bert", type=str, required=True)
    parser.add_argument('--concrete_lower', default=-1.5, type=float, required=False)
    parser.add_argument('--concrete_upper', default=1.5, type=float, required=False)
    parser.add_argument('--total_layers', default=14, type=int, required=False)
    parser.add_argument('--sparsity_penalty', default=1.25e-7, type=float, required=False)
    parser.add_argument('--weight_decay', default=0., type=float, required=False)
    parser.add_argument('--alpha_init', default=5., type=float, required=False)
    parser.add_argument('--lr_params', default=5e-5, type=float, required=False)
    parser.add_argument('--lr_alpha', default=5e-5, type=float, required=False)
    parser.add_argument('--per_layer_alpha', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--per_params_alpha', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--warmup_steps', default=0, type=int, required=False)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1., type=float, required=False)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--no_diff', default=False, type=lambda x: bool(int(x)), required=False)

    # train params
    parser.add_argument('--squad_v2', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--num_train_epochs', default=5, type=int, required=False)
    parser.add_argument('--logging_steps', default=5, type=int, required=False)
    parser.add_argument('--eval_steps', default=5, type=int, required=False)
    parser.add_argument('--write', default=True, type=lambda x: bool(int(x)), required=False,
                        help="Write logs to summary writer")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    return parser


def prepare_train_features(data, tokenizer, max_length, doc_stride):
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
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = data["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append([cls_index])
                tokenized_examples["end_positions"].append([cls_index])
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append([token_start_index - 1])
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append([token_end_index + 1])
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length, doc_stride):
    pad_on_right = tokenizer.padding_side == "right"
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        padding="max_length"
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def tokens_to_dataloader(tokenized_examples, batch_size, shuffle=True):
    try:
        dataset = TensorDataset(
            torch.tensor(tokenized_examples["input_ids"], dtype=torch.long),
            torch.tensor(tokenized_examples["attention_mask"], dtype=torch.long),
            torch.tensor(tokenized_examples["token_type_ids"], dtype=torch.long),
            torch.tensor(tokenized_examples["start_positions"], dtype=torch.long),
            torch.tensor(tokenized_examples["end_positions"], dtype=torch.long)
        )
    except KeyError:
        dataset = TensorDataset(
            torch.tensor(tokenized_examples["input_ids"], dtype=torch.long),
            torch.tensor(tokenized_examples["attention_mask"], dtype=torch.long),
            torch.tensor(tokenized_examples["token_type_ids"], dtype=torch.long)
        )

    distributed = dist.is_available() and dist.is_initialized()
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    return DataLoader(dataset, batch_size=batch_size, num_workers=1 if distributed else 0, sampler=sampler,
                      shuffle=sampler is None and shuffle)


def get_train_data(args, tokenizer, return_val=True):
    datasets = load_dataset("squad_v2" if args.squad_v2 else "squad")
    train_tokens = prepare_train_features(data=datasets["train"][:87_599],
                                          tokenizer=tokenizer,
                                          max_length=args.max_length,
                                          doc_stride=args.doc_stride)
    train_dataloader = tokens_to_dataloader(train_tokens, args.batch_size)
    val_dataloader = None
    if return_val:
        val_tokens = prepare_train_features(data=datasets["validation"][:10_570],
                                            tokenizer=tokenizer,
                                            max_length=args.max_length,
                                            doc_stride=args.doc_stride)
        val_dataloader = tokens_to_dataloader(val_tokens, args.batch_size)
    return train_dataloader, val_dataloader


def get_validation_data(args, tokenizer):
    datasets = load_dataset("squad_v2" if args.squad_v2 else "squad")
    val_tokens = prepare_validation_features(examples=datasets["validation"][:10_570],
                                             tokenizer=tokenizer,
                                             max_length=args.max_length,
                                             doc_stride=args.doc_stride)

    val_dataloader = tokens_to_dataloader(val_tokens, args.batch_size, shuffle=False)
    return val_dataloader, val_tokens


@torch.no_grad()
def postprocess_qa_predictions(args, datasets, tokenized_examples, all_start_logits, all_end_logits, tokenizer,
                               n_best_size=20, max_answer_length=30):
    example_id_to_index = {k: i for i, k in enumerate(datasets['validation']["id"])}
    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(tokenized_examples["example_id"]):
        features_per_example[example_id_to_index[feature]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(datasets['validation'])):
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = tokenized_examples["offset_mapping"][feature_index]

            cls_index = tokenized_examples["input_ids"][feature_index].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        if not args.squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


@torch.no_grad()
def eval(args):
    filename = f"log_{args.task_name}_" + ("diff.txt" if not args.no_diff else "no_diff.txt")
    logging.basicConfig(filename=filename, filemode='a', level=args.logging_level)
    logger = logging.getLogger(__name__)
    logger.info(f"  **** Getting metrics for {args.model_name} ****  ")

    metric = load_metric('squad')
    datasets = load_dataset('squad')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path) \
        .to(torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"))
    set_seed(args)
    diff_model = init_model(args, model)
    if args.model_checkpoint:
        diff_model.load(args.model_checkpoint)
        logger.info(f"     Model loaded from checkpoint {args.model_checkpoint}")

    dataloader, tokenized_examples = get_validation_data(args, tokenizer)
    all_start_logits, all_end_logits = None, None
    # getting logits
    for batch in tqdm(dataloader, position=0, leave=True):
        batch = [element.to(model.device) for element in batch]
        output = diff_model.forward(batch)
        if all_start_logits is None:
            all_start_logits = output.start_logits.cpu().numpy()
            all_end_logits = output.end_logits.cpu().numpy()
        else:
            all_start_logits = np.concatenate((all_start_logits, output.start_logits.cpu().numpy()))
            all_end_logits = np.concatenate((all_end_logits, output.end_logits.cpu().numpy()))

    final_predictions = postprocess_qa_predictions(args, datasets, tokenized_examples, all_start_logits, all_end_logits,
                                                   tokenizer)

    if args.squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                                 final_predictions.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

    metrics = metric.compute(predictions=formatted_predictions, references=references)
    logger.info(f'METRICS: {metrics}')


def train(local_rank, args):
    filename = f"log_{args.task_name}_" + ("diff.txt" if not args.no_diff else "no_diff.txt")
    logging.basicConfig(filename=filename, filemode='a', level=args.logging_level)

    if args.local_rank != -1:
        setup_distributed(local_rank, args.world_size)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # model_class, tokenizer_class = MODEL_CLASSES[args.model_name].values()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    train_dataloader, val_dataloader = get_train_data(args=args,
                                                      tokenizer=tokenizer,
                                                      return_val=True)

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank == 0:
        torch.distributed.barrier()

    diff_model = init_model(args, model)
    set_seed(args)
    diff_model.train(local_rank=local_rank,
                     train_dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     epochs=args.num_train_epochs,
                     max_steps=args.max_steps,
                     logging_steps=args.logging_steps,
                     save_steps=args.save_steps,
                     eval_steps=args.eval_steps,
                     write=args.write)


if __name__ == "__main__":
    args = setup_argparser().parse_args()
    if args.do_train:
        if args.local_rank != -1:
            mp.spawn(train,
                     args=(args,),
                     nprocs=args.world_size,
                     join=True)
        else:
            train(args.local_rank, args)
    elif args.do_eval:
        eval(args)
