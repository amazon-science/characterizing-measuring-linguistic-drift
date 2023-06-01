"""
Extracts contextualized representations and computes cosine similarities between
datasets, using fine-tuned or pre-trained models. If model_name_or_path is set,
then uses the specified pre-trained model. Otherwise, uses the fine-tuned model
for each training domain, from finetune.py.

Outputs the tsv of mean cosine similarities between training/eval domains to:
experiment_output_dir/task/all_[finetuned/pretrained]_cosine_metrics.tsv

For example-level metrics (the mean cosine similarity between each example
embedding and all training example embeddings), we output for each train domain:
experiment_output_dir/task/train_domain/eval_[finetuned/pretrained]_cosine_annotations.npy
With shape: (n_eval_examples).

If using a pre-trained model (i.e. model_name_or_path is set), then computes
the lexical semantic similarity metrics too, at both the dataset level and example level.
Example-level lexical semantic similarities are outputted to:
experiment_output_dir/task/train_domain/eval_pretrained_semantic_annotations.npy
With shape: (n_eval_examples, 2).
Columns correspond to lexical semantic similarities when unfiltered or filtered
to content tokens. When computing lexical semantic similarities, pos_dict_path
defines the content words (see get_content_word_mask() in nlp_drift/cosine_metrics.py).
We provide a default POS file (English_pos_dict.tsv) based on the Universal
Dependencies English corpus.

Sample usage (pre-trained and fine-tuned):

python3 compute_cosine_metrics.py \
--max_seq_length 512 \
--per_device_eval_batch_size 16 --output_dir="placeholder" \
--task="sentiment_amazon_categories" --dataset_dir="datasets" \
--experiment_output_dir="experiment_output" \
--pos_dict_path="English_pos_dict.tsv" \
--model_name_or_path="roberta-base"

python3 compute_cosine_metrics.py \
--max_seq_length 512 \
--per_device_eval_batch_size 16 --output_dir="placeholder" \
--task="sentiment_amazon_categories" --dataset_dir="datasets" \
--experiment_output_dir="experiment_output"

"""

import os
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from compute_drift_metrics import MetricArguments
from nlp_drift.dataset_utils import get_eval_datasets, get_all_domains
from nlp_drift.models import ExperimentArguments, ModelArguments
from nlp_drift.cosine_metrics import compute_cosine_metrics
from nlp_drift.drift_evaluation_utils import combine_output_data


def main():
    parser = HfArgumentParser((ModelArguments, ExperimentArguments, TrainingArguments, MetricArguments))
    model_args, experiment_args, training_args, metric_args = parser.parse_args_into_dataclasses()

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

    # Run for each training domain.
    # Each model will be evaluated on all eval domains.
    # As in AWS Model Monitor, uses the last two hidden layers for representations
    # (https://aws.amazon.com/blogs/machine-learning/detect-nlp-data-drift-using-custom-amazon-sagemaker-model-monitor/).
    for train_domain in train_domains:
        if model_args.model_name_or_path is None:
            # Fine-tuned model representations.
            compute_cosine_metrics(model_args, experiment_args, training_args,
                                   train_domain, eval_datasets, layers=[-1,-2])
        else:
            # Pre-trained model representations.
            # In this case, compute lexical semantic similarities.
            compute_cosine_metrics(model_args, experiment_args, training_args,
                                   train_domain, eval_datasets, use_pretrained=model_args.model_name_or_path, layers=[-1,-2],
                                   include_lexical_semantic_similarity=True, content_words_path=metric_args.pos_dict_path)

    # Compile results into one file.
    if model_args.model_name_or_path is None:
        filename = "finetuned_cosine_metrics.tsv"
    else:
        filename = "pretrained_cosine_metrics.tsv"
    combine_output_data(experiment_args.experiment_output_dir, experiment_args.task,
                        filename, dir_suffix=experiment_args.dir_suffix)
    print("Done.")


if __name__ == "__main__":
    main()
