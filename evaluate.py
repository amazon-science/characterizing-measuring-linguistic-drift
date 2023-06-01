"""
Evaluate fine-tuned Transformer models on all eval domains.
Outputs the tsv of accuracies to:
experiment_output_dir/task/all_domain_eval_results.tsv.

Outputs the raw predictions for each model to:
experiment_output_dir/task/train_domain/domain_eval_predictions.npy
With shape: n_examples, n_classes+1. The first columns correspond to class logits,
and the last column is the true label.

Sample usage:

python3 evaluate.py \
--max_seq_length=512 \
--per_device_eval_batch_size=16 --output_dir="placeholder" \
--task="sentiment_amazon_categories" --dataset_dir="datasets" \
--experiment_output_dir="experiment_output"

"""

import os
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from nlp_drift.models import ExperimentArguments, ModelArguments, evaluate_model
from nlp_drift.dataset_utils import get_all_domains, get_eval_datasets
from nlp_drift.drift_evaluation_utils import combine_output_data


def main():
    parser = HfArgumentParser((ModelArguments, ExperimentArguments, TrainingArguments))
    model_args, experiment_args, training_args = parser.parse_args_into_dataclasses()

    if len(experiment_args.train_domains) == 0:
        # By default, evaluate all training domains.
        train_domains = get_all_domains(experiment_args.task)
    else:
        train_domains = experiment_args.train_domains.split(",")
    if len(experiment_args.eval_domains) == 0:
        # By default, evaluate on all eval domains.
        eval_domains = get_all_domains(experiment_args.task, eval=True)
    else:
        eval_domains = experiment_args.eval_domains.split(",")

    eval_datasets = get_eval_datasets(
        experiment_args.dataset_dir, experiment_args.task, eval_domains)

    # Run evaluation for each training domain.
    # Each model will be evaluated on all eval domains.
    for train_domain in train_domains:
        evaluate_model(model_args, experiment_args, training_args, train_domain, eval_datasets)

    # Compile results into one file.
    combine_output_data(experiment_args.experiment_output_dir, experiment_args.task,
                        "domain_eval_results.tsv", dir_suffix=experiment_args.dir_suffix)
    print("Done.")


if __name__ == "__main__":
    main()
