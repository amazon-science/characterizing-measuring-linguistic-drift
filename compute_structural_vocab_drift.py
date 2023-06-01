"""
Annotate structural and vocabulary cross-entropies for eval examples.
To align with the output predictions from evaluate.py, scripts should be run
with a consistent set of eval domains (e.g. all eval domains, by default).
Outputs annotations in eval_structural_vocab_xent_annotations.npy in each train
domain directory, with shape (n_eval_examples, 2). Columns correspond to
structural and vocabulary cross-entropy.

Sample usage:

python3 compute_structural_vocab_drift.py \
--experiment_output_dir="experiment_output" --task="sentiment_amazon_categories" \
--dataset_dir="datasets" --max_seq_length=512

"""

import os
import argparse

from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.spacy_metrics import annotate_structural_vocab_xent


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--train_domains', default="")
    parser.add_argument('--eval_domains', default="")
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--ngram_n', type=int, default=5)
    # Minimum POS sequence length when computing POS n-grams. Counting the SEP
    # token after each sentence. Minimum 3 means that examples must contain more
    # than one non-SEP token (at least three tokens counting the SEP token).
    parser.add_argument('--min_pos_seq_length', type=int, default=3)
    parser.add_argument('--dir_suffix', default="")
    return parser


def main(args):
    if len(args.train_domains) == 0:
        train_domains = get_all_domains(args.task)
    else:
        train_domains = args.train_domains.split(",")
    if len(args.eval_domains) == 0:
        eval_domains = get_all_domains(args.task, eval=True)
    else:
        eval_domains = args.eval_domains.split(",")

    annotate_structural_vocab_xent(args.experiment_output_dir, args.dataset_dir, args.task,
            train_domains, eval_domains, dir_suffix=args.dir_suffix, max_seq_length=args.max_seq_length,
            min_pos_seq_length=args.min_pos_seq_length, ngram_n=args.ngram_n)
    print("Done.")
    return True

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
