import argparse
import logging
from typing import Dict

import numpy as np
import os
import random
import torch
import torch.multiprocessing as mp
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertForQuestionAnswering,
    BertForTokenClassification,
    BertConfig,
    BertTokenizer,
    DistilBertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer
)

from model import DiffPruning
from ner_utils import build_dict
from data import get_bc2gm_train_data
from tags import UTIL_TAGS
from eval import evaluate_ner_metrics
from squad_utils import get_train_data, get_validation_data, postprocess_qa_predictions

MODEL_CLASSES = {'qa': {
    "bert": {"model": BertForQuestionAnswering,
             "config": BertConfig,
             "tokenizer": BertTokenizer},
    "distilbert": {"model": DistilBertForQuestionAnswering,
                   "config": DistilBertConfig,
                   "tokenizer": DistilBertTokenizer}
},
    'ner': {
        "bert": {"model": BertForTokenClassification,
                 "config": BertConfig,
                 "tokenizer": BertTokenizer}
    }
}


def init_model(args, model):
    return DiffPruning(model=model,
                       task_name=args.task_name,
                       model_name=args.model_name,
                       concrete_lower=args.concrete_lower,
                       concrete_upper=args.concrete_upper,
                       total_layers=args.total_layers,
                       sparsity_penalty=args.sparsity_penalty,
                       lambda_increase_steps=args.lambda_increase_steps,
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
    parser.add_argument('--do_train', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--do_eval', default=False, type=lambda x: bool(int(x)), required=False)
    parser.add_argument('--local_rank', default=-1, type=int, required=False)
    parser.add_argument('--world_size', default=1, type=int, required=False)
    parser.add_argument('--n_gpu', default=1, type=int, required=False)
    parser.add_argument('--logging_level', default=20, type=int, required=False)
    parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                        help="used in model_class.from_pretrained()")
    parser.add_argument('--model_checkpoint', default=None, type=str, required=False,
                        help="checkpoint to load the model from")
    parser.add_argument('--load_diff_checkpoint', default=False, type=lambda x: bool(int(x)), required=False,
                        help="whether the checkpoint that is being loaded was trained in diff mode")
    parser.add_argument('--magnitude_pruning', default=False, type=lambda x: bool(int(x)), required=False,
                        help="whether to apply magnitude pruning at the end of training")
    parser.add_argument('--threshold', default=-1, type=float, required=False,
                        help='magnitude pruning threshold')
    parser.add_argument('--target_sparsity', default=0.005, type=float, required=False,
                        help='target sparsity for magnitude pruning')
    parser.add_argument('--fixmask_finetune', default=False, type=lambda x: bool(int(x)), required=False,
                        help='whether to apply fixed mask finetuning (after magnitude pruning)')

    # data params
    parser.add_argument('--path_to_train', default=None, type=str, required=False)
    parser.add_argument('--path_to_val', default=None, type=str, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--max_length', default=384, type=int, required=False)
    parser.add_argument('--doc_stride', default=128, type=int, required=False)
    parser.add_argument('--tokenizer_name', default=None, type=str, required=False)
    parser.add_argument('--use_fast', default=True, type=lambda x: bool(int(x)), required=False,
                        help="whether to use fast tokenizer")
    parser.add_argument('--do_lower_case', default=True, type=lambda x: bool(int(x)), required=False)

    # model params
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument('--model_name', default=None, type=str, required=True)
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
    parser.add_argument('--scheduler', default='const', type=str, required=False)
    parser.add_argument('--warmup_steps', default=0, type=int, required=False)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False)
    parser.add_argument('--lambda_increase_steps', default=0, type=int, required=False,
                        help="number of training steps for the sparsity rate to reach its target value")
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
    parser.add_argument('--save_steps', type=int, default=10,
                        help="Save last checkpoint every X update steps")
    parser.add_argument('--update_steps_start', type=int, default=0,
                        help="when using pretrained model enter how many update steps it already underwent")
    return parser


@torch.no_grad()
def eval_squad(args):
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

    dataloader, tokenized_examples = get_validation_data(args, tokenizer)
    all_start_logits, all_end_logits = None, None

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


@torch.no_grad()
def eval_ner(args):
    model, tokenizer = load_model(args.local_rank, args)
    _, dataloader = get_bc2gm_train_data(args, tokenizer, return_train=False, return_val=True, tags_vocab=args.label2id)
    diff_model = init_model(args, model)
    if args.model_checkpoint:
        if args.load_diff_checkpoint:
            diff_model.load(args.model_checkpoint, train=False)
        else:
            diff_model.load(args.model_checkpoint, no_diff_load=True)
    set_seed(args)

    _, nonzero = diff_model._calculate_params(no_grads=True)
    _, metrics = evaluate_ner_metrics(diff_model, dataloader, args.label_map, tokenizer)

    for metric, value in metrics.items():
        print(f"{metric} = {value:.4f}")
    print(f"nonzero count = {nonzero}")
    print(f"nonzero frac = {diff_model.get_sparsity(threshold=0.):.4f}")


def load_model(local_rank, args):
    # Make sure only the first process in distributed training will download model & vocab
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model_class, config_class, tokenizer_class = MODEL_CLASSES[args.task_name][args.model_name].values()

    if args.task_name == 'ner':
        label2id: Dict[str, int] = build_dict(UTIL_TAGS + ['GENE'], ['B-', 'I-', 'E-', 'S-'])
        label_map = {value: key for key, value in label2id.items()}
        num_labels = len(label_map)
        args.label_map = label_map
        args.label2id = label2id

        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id=label2id,
        )
    else:
        config = config_class.from_pretrained(
            args.model_name_or_path,
        )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return model, tokenizer


def magnitude_pruning(args):
    model, tokenizer = load_model(args.local_rank, args)
    diff_model = init_model(args, model)
    if args.model_checkpoint:
        if args.load_diff_checkpoint:
            diff_model.load(args.model_checkpoint, train=False)
        else:
            diff_model.load(args.model_checkpoint, no_diff_load=True)
    if args.threshold < 0:  # DEFAULT -1 USED
        assert 0 <= args.target_sparsity <= 1, "Target sparsity must be between 0 and 1"
        l, r = 0., 1.
        # SEARCHING FOR A THRESHOLD THAT'LL PROVIDE THE TARGET SPARSITY BY ZEROING OUT EVERYTHING BELOW IT
        # THRESHOLD CAN'T BE GREATER THAN 1
        while l + 1e-6 < r:
            mid = (l + r) / 2.0
            if diff_model.get_sparsity(mid) <= args.target_sparsity:
                r = mid
            else:
                l = mid
        threshold = r
    else:
        threshold = args.threshold

    diff_model.apply_magnitude_pruning(threshold)

    save_name = args.model_checkpoint.replace('.pt', '') + f'_mag_prune_{args.target_sparsity}.pt'
    diff_model.save(path_bert_params=save_name)

    _, dataloader = get_bc2gm_train_data(args, tokenizer, return_train=False, return_val=True, tags_vocab=args.label2id)
    set_seed(args)
    _, metrics = evaluate_ner_metrics(diff_model, dataloader, args.label_map, tokenizer)
    for metric, value in metrics.items():
        print(f"{metric} = {value:.4f}")
    print(f"% nonzero = {diff_model.get_sparsity(threshold=0.)}")


def train(local_rank, args):
    filename = f"log_{args.task_name}_" + ("diff_.txt" if not args.no_diff else "no_diff.txt")
    logging.basicConfig(filename=filename, filemode='a', level=args.logging_level)

    if args.local_rank != -1:
        setup_distributed(local_rank, args.world_size)

    model, tokenizer = load_model(local_rank, args)

    if args.task_name == 'squad':
        train_dataloader, val_dataloader = get_train_data(args=args,
                                                          tokenizer=tokenizer,
                                                          return_val=True)
    elif args.task_name == 'ner':
        train_dataloader, val_dataloader = get_bc2gm_train_data(args, tokenizer, tags_vocab=args.label2id,
                                                                return_val=True)

    else:
        train_dataloader, val_dataloader = None, None

    # Make sure only the first process in distributed training will download model & vocab
    if local_rank == 0:
        torch.distributed.barrier()

    diff_model = init_model(args, model)
    eval_func = evaluate_ner_metrics if args.task_name == 'ner' else None

    if args.model_checkpoint:
        if args.load_diff_checkpoint:
            diff_model.load(args.model_checkpoint, train=True)
        else:
            diff_model.load(args.model_checkpoint, no_diff_load=True)

    if args.fixmask_finetune:
        diff_model.fixmask_finetune()

    set_seed(args)
    diff_model.train(local_rank=local_rank,
                     args=args,
                     train_dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     eval_func=eval_func,
                     label_map=args.label_map,
                     tokenizer=tokenizer)


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
        if args.task_name == 'squad':
            eval_squad(args)
        else:
            eval_ner(args)
    elif args.magnitude_pruning:
        magnitude_pruning(args)
