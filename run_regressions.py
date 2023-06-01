"""
Run a logistic regression for each model, and each drift metric(s), predicting
whether the model will get each example correct.
Outputs predicted accuracies to:
experiment_output_dir/task/eval_performance_predictions.tsv
Outputs individual example probability predictions to a subdirectory in each
train domain directory:
experiment_output_dir/task/train_domain/individual_performance_predictions
This contains one npy file for each regression, containing the predicted
probabilities of getting each evaluation example correct (using the model trained
on train_domain) based on the drift metric(s) for that regression.

Sample usage:

python3 run_regressions.py \
--experiment_output_dir="experiment_output" \
--task="sentiment_amazon_categories" \
--dataset_dir="datasets" \
--outfile_name="eval_performance_predictions.tsv"

"""

import argparse
import os

from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.drift_evaluation_utils import (get_all_example_data,
    run_logistic_regressions, run_constant_regressions, run_ground_truth_regressions)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--outfile_name', default="eval_performance_predictions.tsv")
    parser.add_argument('--k_fold', default=5)
    # Appended after the task name in the experiment output path. Corresponds
    # to different fine-tuning runs. If non-empty, should be "0", "1", "2", etc.
    parser.add_argument('--dir_suffix', default="")
    return parser


def main(args):
    # Example level drift metrics.
    # Run a regression for each of these metric(s) strings and for each model (trained on some train domain).
    independent_vars_strings = ["frequency_js_div", "frequency_xent", "pretrained_cosine",
                                "finetuned_cosine", "structural_xent", "vocab_xent",
                                "semantic_all", "semantic_content",
                                "structural_xent,vocab_xent", "structural_xent,semantic_content",
                                "vocab_xent,semantic_content",
                                "structural_xent,vocab_xent,semantic_content",
                                "structural_xent,vocab_xent,semantic_content,finetuned_cosine",
                                "structural_xent,vocab_xent,semantic_content,structural_xent*vocab_xent,structural_xent*semantic_content,vocab_xent*semantic_content",
                                "frequency_js_div,frequency_xent,pretrained_cosine,finetuned_cosine",
                                "frequency_js_div,frequency_xent,pretrained_cosine"]
    save_predictions = [True, True, True, True, True, True, True, True,
                        False, False, False, True, True, True, True, True]
    assert len(independent_vars_strings) == len(save_predictions)

    run_logistic_regressions(independent_vars_strings, args.experiment_output_dir, args.dataset_dir,
                             args.task, dir_suffix=args.dir_suffix, k_fold=args.k_fold, outfile_name=args.outfile_name,
                             save_predictions=save_predictions)
    run_constant_regressions(args.experiment_output_dir, args.dataset_dir, args.task,
                             dir_suffix=args.dir_suffix, k_fold=args.k_fold, outfile_name=args.outfile_name) # Intercept only.
    run_ground_truth_regressions(args.experiment_output_dir, args.dataset_dir, args.task,
                                 dir_suffix=args.dir_suffix, k_fold=args.k_fold, outfile_name=args.outfile_name) # Ground truth accuracy.
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
