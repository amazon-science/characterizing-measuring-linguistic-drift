"""
Computes the RMSEs for different drift metrics for a given task.
Outputs to: experiment_output_dir/task/rmses.tsv
Sample usage:

python3 get_rmse_scores.py \
--experiment_output_dir="experiment_output" \
--task="sentiment_amazon_categories"

"""

import argparse
import os

from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.drift_evaluation_utils import get_rmses


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--task', required=True)
    # If -1, only consider one run with no dir_suffix (default).
    # If >0, consider n_runs fine-tuning runs with dir_suffix equal to each
    # training run index.
    parser.add_argument('--n_runs', type=int, default=-1)
    return parser


def main(args):
    train_domains = get_all_domains(args.task)
    eval_domains = get_all_domains(args.task, eval=True)

    # Comma-separated list of drift metrics inputted to each regression, in the
    # order they were included in run_regressions.py.
    metric_strings = ["constant", "frequency_js_div", "frequency_xent", "pretrained_cosine",
        "finetuned_cosine", "structural_xent", "vocab_xent", "semantic_content",
        "structural_xent,vocab_xent,semantic_content",
        "frequency_js_div,frequency_xent,pretrained_cosine",
        "frequency_js_div,frequency_xent,pretrained_cosine,finetuned_cosine"]

    rmses = get_rmses(args.experiment_output_dir, args.task, train_domains, eval_domains, metric_strings, n_runs=args.n_runs)
    os.makedirs(os.path.join(args.experiment_output_dir, args.task), exist_ok=True)
    outpath = os.path.join(args.experiment_output_dir, args.task, "rmses.tsv")
    rmses.to_csv(outpath, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
