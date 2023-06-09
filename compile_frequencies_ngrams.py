"""
For debugging, outputs the content word frequencies and POS n-gram frequencies
in each eval domain, in decreasing frequency order. Assumes compute_drift_metrics.py
has already been run for spaCy metrics, and loads from the spaCy cache.
Outputs to:
experiment_output_dir/task/eval_domain_data

Sample usage:
python3 compile_frequencies_ngrams.py \
--task="sentiment_amazon_categories" \
--experiment_output_dir="experiment_output"

"""

import codecs
import argparse
import os
import itertools

from nlp_drift.spacy_metrics import get_spacy_distributions
from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.constants import SPACY_POS_TAGS


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--dir_suffix', default="")
    parser.add_argument('--ngram_n', type=int, default=5)
    return parser


def main(args):
    eval_domains = get_all_domains(args.task, eval=True)
    spacy_cache = os.path.join(args.experiment_output_dir, args.task + args.dir_suffix, "spacy_cache")
    print("Note: spaCy cache should be saved at: {} (generated by compute_drift_metrics.py)".format(spacy_cache))
    output_dir = os.path.join(args.experiment_output_dir, args.task + args.dir_suffix, "eval_domain_data")
    os.makedirs(output_dir, exist_ok=True)
    for eval_domain in eval_domains:
        print("Running domain {}.".format(eval_domain))
        # Note: ngram_probs are n-gram distributions instead of conditional n-gram distributions.
        # Assume pickled, so set dataset and task to None.
        collected_frequencies, all_ngram_probs = get_spacy_distributions(None, None,
                                            spacy_cache, eval_domain+"_eval")
        content_frequencies = collected_frequencies[0]
        del collected_frequencies
        ngram_probs = all_ngram_probs[args.ngram_n-1].copy()
        del all_ngram_probs

        # Output content word frequencies in decreasing sorted order.
        sorted_freqs = sorted(content_frequencies.items(), key=lambda x: x[1], reverse=True)
        outfile = codecs.open(os.path.join(output_dir, "{0}-content_word_frequencies.tsv".format(eval_domain)), 'w', encoding='utf-8')
        outfile.write("ContentWord\tFrequency\n")
        for content_word, freq in sorted_freqs:
        	outfile.write("{0}\t{1}\n".format(content_word, freq))
        outfile.close()

        # Output n-gram frequencies in decreasing sorted order.
        sequence_freq_dict = dict()
        sequences = list(itertools.product(range(len(SPACY_POS_TAGS)), repeat=args.ngram_n))
        for sequence in sequences:
            readable_sequence = " ".join([SPACY_POS_TAGS[i] for i in sequence])
            freq = ngram_probs[sequence]
            if freq > 0.0:
                sequence_freq_dict[readable_sequence] = freq
        # Output.
        sorted_freqs = sorted(sequence_freq_dict.items(), key=lambda x: x[1], reverse=True)
        outfile = codecs.open(os.path.join(output_dir, "{0}-pos_sequence_frequencies.tsv".format(eval_domain)), 'w', encoding='utf-8')
        outfile.write("PosSequence\tFrequency\n")
        for readable_sequence, freq in sorted_freqs:
        	outfile.write("{0}\t{1}\n".format(readable_sequence, freq))
        outfile.close()
    print("Saved to: {}".format(output_dir))
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
