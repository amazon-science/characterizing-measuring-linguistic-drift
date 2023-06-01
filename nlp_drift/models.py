"""
Utilities for the Hugging Face models, including fine-tuning and evaluation.
"""

import os
import numpy as np
import codecs
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_metric
import transformers
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EvalPrediction, Trainer,
)

from .custom_settings import (CUSTOM_TASK_NAME, CUSTOM_N_LABELS, CUSTOM_TEXT_FIELDS)
from .dataset_utils import get_dataset


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to the domain drift experiments.

    Using `HfArgumentParser` we can turn this class into argparse arguments to
    be able to specify them on the command line.
    """
    experiment_output_dir: str = field(default="experiment_output")
    dataset_dir: str = field(default="datasets",
        metadata={"help": "In this directory, datasets should be in files: [task]/[domain]_[train/eval].tsv"})
    task: str = field(default="sentiment_amazon_categories")
    train_domains: str = field(default="") # Empty defaults to all train domains.
    eval_domains: str = field(default="") # Empty defaults to all eval domains.
    # In the output path, append this suffix after the task name.
    # This allows multiple fine-tuning runs for the same task.
    dir_suffix: str = field(default="")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="hf_model_cache", metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False, metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


# Fine-tune a Transformer model on the train domain for a sequence classification
# task. Tested on roberta-base. Training arguments are defined by Hugging Face.
# Outputs the model to experiment_output_dir/task/train_domain.
def finetune_model(model_args, experiment_args, training_args, domain):
    print("Fine-tuning domain: {}".format(domain))
    # Output directory.
    model_output_dir = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix, domain)
    if training_args.do_train and not training_args.overwrite_output_dir:
        if os.path.isfile(os.path.join(model_output_dir, "pytorch_model.bin")):
            print("ERROR: output model already exists.")
            return False
    training_args.output_dir = model_output_dir # Overrides any previous setting.
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    # Sequence classification tasks.
    model_class = AutoModelForSequenceClassification
    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        n_labels = 2
    elif experiment_args.task == "mnli":
        n_labels = 3
    elif experiment_args.task == CUSTOM_TASK_NAME:
        n_labels = CUSTOM_N_LABELS
    else:
        print("Unrecognized task: {}".format(experiment_args.task))
        return False
    eval_metric = load_metric("accuracy")
    metric_names = ["accuracy"]
    # Custom compute_metrics function should return a dictionary of string
    # to float.
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        # Returns {"accuracy": [acc]}.
        return eval_metric.compute(predictions=predictions, references=labels)

    # Load dataset. Ignore test dataset.
    train_dataset, eval_dataset, _ = get_dataset(
        experiment_args.dataset_dir, experiment_args.task, domain)

    # Load config, tokenizer, and model.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=n_labels,
        finetuning_task=experiment_args.task,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        text_fields = ["review_body"]
    elif experiment_args.task in ["mnli"]:
        text_fields = ["premise", "hypothesis"]
    elif experiment_args.task == CUSTOM_TASK_NAME:
        text_fields = CUSTOM_TEXT_FIELDS
    def preprocess_function(examples):
        inputs = tuple([examples[text_field] for text_field in text_fields])
        return tokenizer(
            *inputs,
            padding=False,
            max_length=model_args.max_seq_length,
            truncation=True,
        )

    # Preprocess (tokenize) examples.
    if training_args.do_train:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on validation dataset",
            )

    # Initialize Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )

    # Train.
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()  # Saves the tokenizer too for easy upload.
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation.
        if training_args.do_eval:
            print("Evaluating...")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    print("Completed fine-tuning for domain: {}".format(domain))
    return True


# Evaluate a fine-tuned Transformer model on the eval_datasets (a dictionary
# from eval domain names to eval datasets). Tested on roberta-base.
# Outputs the tsv of accuracies to experiment_output_dir/task/train_domain/domain_eval_results.tsv.
# Outputs the predictions to experiment_output_dir/task/train_domain/domain_eval_predictions.npy,
# with shape: n_examples, n_classes+1. The first columns correspond to class logits,
# and the last column is the true label.
def evaluate_model(model_args, experiment_args, training_args, train_domain, eval_datasets):
    print("Evaluating model trained on domain: {}".format(train_domain))
    # Output directory.
    model_path = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix, train_domain)
    output_path = os.path.join(model_path, "domain_eval_results.tsv")
    if os.path.isfile(output_path):
        print("ERROR: output file already exists.")
        return False
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    # Load model, same as in finetune_model(), but the model should already exist in
    # the output directory.
    # Sequence classification tasks.
    model_class = AutoModelForSequenceClassification
    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        n_labels = 2
    elif experiment_args.task == "mnli":
        n_labels = 3
    elif experiment_args.task == CUSTOM_TASK_NAME:
        n_labels = CUSTOM_N_LABELS
    else:
        print("Unrecognized task: {}".format(experiment_args.task))
        return False
    eval_metric = load_metric("accuracy")
    metric_names = ["accuracy"]
    # Custom compute_metrics function should return a dictionary of string
    # to float.
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        # Returns {"accuracy": [acc]}.
        return eval_metric.compute(predictions=predictions, references=labels)
    # Load config, tokenizer, and model.
    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=n_labels,
        finetuning_task=experiment_args.task,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = model_class.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        text_fields = ["review_body"]
    elif experiment_args.task in ["mnli"]:
        text_fields = ["premise", "hypothesis"]
    elif experiment_args.task == CUSTOM_TASK_NAME:
        text_fields = CUSTOM_TEXT_FIELDS
    def preprocess_function(examples):
        inputs = tuple([examples[text_field] for text_field in text_fields])
        return tokenizer(
            *inputs,
            padding=False,
            max_length=model_args.max_seq_length,
            truncation=True,
        )
    # End code section copied from finetune_model().

    outfile = codecs.open(output_path, 'w', encoding='utf-8')
    outfile.write("TrainDomain\tEvalDomain\tEvalExamples\t{0}\n".format("\t".join(metric_names)))
    all_predictions = []
    for eval_domain, eval_dataset in eval_datasets.items():
        print("Evaluating train/eval domain pair: {0}, {1}".format(train_domain, eval_domain))
        # Preprocess (tokenize) examples.
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            processed_eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                desc="Running tokenizer on validation dataset",
            )
        # Initialize Trainer.
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        )
        # Evaluate.
        # Use predict() instead of evaluate() to get predictions.
        # predictions shape: n_examples, n_classes
        # label_ids shape: n_examples
        predictions, label_ids, metrics = trainer.predict(test_dataset=processed_eval_dataset)
        # Predictions, then true labels.
        predictions = np.concatenate([predictions, label_ids.reshape(-1, 1)], axis=-1)
        all_predictions.append(predictions)
        print(metrics)
        # Write outputs.
        outfile.write("{0}\t{1}\t{2}".format(train_domain, eval_domain, len(eval_dataset)))
        for metric_name in metric_names:
            outfile.write("\t{}".format(metrics["test_" + metric_name]))
        outfile.write("\n")
    outfile.close()
    # Shape: n_examples, n_classes+1.
    # First columns are the logits for individual classes, and the last
    # column is the true label.
    all_predictions = np.concatenate(all_predictions, axis=0)
    np.save(os.path.join(model_path, "domain_eval_predictions.npy"), all_predictions, allow_pickle=False)
    print("Completed evaluation for training domain: {}".format(train_domain))
    return True
