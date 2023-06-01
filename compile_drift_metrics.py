"""
Compiles any computed example-level drift metrics into one file.
Outputs to: experiment_output_dir/task/eval_example_drift_metrics.tsv
Each row corresponds to a single evaluation example relative to one training domain.
Depending on the metrics computed, columns can include the training and evaluation
domains, the various drift metrics, whether the trained model correctly predicted
the example, the task-specific text fields, and the ground truth label.

Sample usage:

python3 compile_drift_metrics.py \
--task="sentiment_amazon_categories" --dataset_dir="datasets" \
--experiment_output_dir="experiment_output"

"""

import argparse
import os
import itertools
from datasets import concatenate_datasets

from nlp_drift.custom_settings import CUSTOM_TASK_NAME, CUSTOM_TEXT_FIELDS
from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.drift_evaluation_utils import get_all_example_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--dir_suffix', default="")
    return parser


def main(args):
    # Include all possible metrics.
    columns_to_keep = ["labels", "structural_xent", "vocab_xent", "frequency_js_div", "frequency_xent", "sequence_length",
                       "finetuned_cosine", "pretrained_cosine", "semantic_content"]
    if args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        columns_to_keep.extend(["review_body"])
    elif args.task in ["mnli"]:
        columns_to_keep.extend(["premise", "hypothesis"])
    elif args.task == CUSTOM_TASK_NAME:
        columns_to_keep.extend(CUSTOM_TEXT_FIELDS)

    example_data = get_all_example_data(args.experiment_output_dir, args.dataset_dir,
                        args.task, columns_to_keep=columns_to_keep, dir_suffix=args.dir_suffix, as_dict="train,eval")
    # Replace integer train/eval domains with string domain names.
    final_dataset = []
    train_domains = get_all_domains(args.task)
    eval_domains = get_all_domains(args.task, eval=True)
    for train_domain, eval_domain in itertools.product(train_domains, eval_domains):
        dataset = example_data.pop(train_domain + "," + eval_domain, None)
        dataset = dataset.add_column(name="TrainDomain", column=[train_domain]*len(dataset))
        dataset = dataset.add_column(name="EvalDomain", column=[eval_domain]*len(dataset))
        dataset = dataset.remove_columns(["train_domain", "eval_domain"])
        final_dataset.append(dataset)
    final_dataset = concatenate_datasets(final_dataset)

    # Save.
    os.makedirs(os.path.join(args.experiment_output_dir, args.task + args.dir_suffix), exist_ok=True)
    outpath = os.path.join(args.experiment_output_dir, args.task + args.dir_suffix, "eval_example_drift_metrics.tsv")
    final_dataset.to_csv(outpath, sep="\t", index=False)
    print("Saved to: {}".format(outpath))
    print("Columns saved: {}".format(final_dataset.column_names))
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
