"""
Fine-tune a language model on each training domain for a sequence classification
task. Outputs each model to experiment_output_dir/task/train_domain.
Sample usage:

python3 finetune.py \
--model_name_or_path="roberta-base" --max_seq_length=512 \
--output_dir="placeholder" --save_strategy="no" \
--do_train --do_eval --evaluation_strategy="steps" --eval_steps=1000 \
--per_device_train_batch_size=8 --gradient_accumulation_steps=4 \
--lr_scheduler_type="linear" --learning_rate=0.00002 --warmup_ratio=0.10 \
--num_train_epochs=4 --per_device_eval_batch_size=8 \
--task="sentiment_amazon_categories" --dataset_dir="datasets" \
--experiment_output_dir="experiment_output" --seed=42

"""

import os
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from nlp_drift.models import ExperimentArguments, ModelArguments, finetune_model
from nlp_drift.dataset_utils import get_all_domains


def main():
    parser = HfArgumentParser((ModelArguments, ExperimentArguments, TrainingArguments))
    model_args, experiment_args, training_args = parser.parse_args_into_dataclasses()

    if len(experiment_args.train_domains) == 0:
        # By default, fine-tune on each domain.
        train_domains = get_all_domains(experiment_args.task)
    else:
        train_domains = experiment_args.train_domains.split(",")

    # Run fine-tuning for each domain.
    task_output_dir = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix)
    os.makedirs(task_output_dir, exist_ok=True)
    for train_domain in train_domains:
        finetune_model(model_args, experiment_args, training_args, train_domain)
    print("Done.")


if __name__ == "__main__":
    main()
