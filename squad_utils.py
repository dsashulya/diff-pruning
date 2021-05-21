import collections
import torch
from datasets import load_dataset
import numpy as np
from data import tokens_to_dataloader
from tqdm import tqdm


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
