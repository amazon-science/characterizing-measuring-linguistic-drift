"""
Computes model-agnostic dataset-level drift metrics between datasets.
Outputs a tsv of the drift metric between domain pairs, from train
set A to eval set B.

If computing frequency metrics, the example-level frequency metrics are outputted
into each train domain output directory:
experiment_output_dir/task/train_domain/eval_frequency_annotations.npy
With shape (n_eval_examples, 3), concatenating all evaluation domains.
Columns correspond to JS-distance, cross-entropy, and raw example sequence length.
To align with the output predictions from evaluate.py, scripts should
be run with a consistent set of eval domains (e.g. by default, all scripts use
all eval domains).

Sample usage (frequency and spaCy metrics):

python3 compute_drift_metrics.py \
--tokenizer_name="roberta-base" --metrics="frequency" \
--experiment_output_dir="experiment_output" --task="sentiment_amazon_categories" \
--dataset_dir="datasets" \
--max_seq_length=512

python3 compute_drift_metrics.py \
--metrics="spacy" \
--experiment_output_dir="experiment_output" --task="sentiment_amazon_categories" \
--dataset_dir="datasets" \
--max_seq_length=512

"""

import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from nlp_drift.models import ExperimentArguments, ModelArguments
from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.frequency_metrics import compute_all_frequency_metrics
from nlp_drift.spacy_metrics import compute_all_spacy_metrics


@dataclass
class MetricArguments:
    """
    Arguments pertaining to the drift metrics.
    """
    # Either frequency (frequency_js_distance, frequency_xent) or spacy
    # (pos_unigram_distance, pos_bigram_distance, pos_trigram_distance,
    # pos_4gram_distance, pos_5gram_distance, content_word_distance,
    # spacy_token_distance, adjective_distance, adverb_distance, noun_distance,
    # pronoun_distance, verb_distance).
    metrics: str = field(default="frequency")
    # Minimum POS sequence length when computing POS sequence probabilities for
    # spaCy metrics. Counting the SEP token after each sentence.
    # Minimum 3 means that examples must contain more than one non-SEP token
    # (at least three tokens counting the SEP token).
    min_pos_seq_length: int = field(default=3)
    # Only used for the lexical semantic similarity metric, to identify content
    # tokens in the RoBERTa tokenizer (in compute_cosine_metrics.py).
    pos_dict_path: str = field(default="")


def main():
    parser = HfArgumentParser((ModelArguments, ExperimentArguments, MetricArguments))
    model_args, experiment_args, metric_args = parser.parse_args_into_dataclasses()

    if len(experiment_args.train_domains) == 0:
        train_domains = get_all_domains(experiment_args.task)
    else:
        train_domains = experiment_args.train_domains.split(",")
    if len(experiment_args.eval_domains) == 0:
        eval_domains = get_all_domains(experiment_args.task, eval=True)
    else:
        eval_domains = experiment_args.eval_domains.split(",")

    task_output_dir = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix)
    os.makedirs(task_output_dir, exist_ok=True)
    if metric_args.metrics == "frequency":
        outpath = os.path.join(task_output_dir, "frequency_metrics.tsv")
        if os.path.isfile(outpath):
            print("ERROR: frequency_metrics.tsv already exists.")
            return
        compute_all_frequency_metrics(model_args, experiment_args, train_domains, eval_domains, outpath)
    if metric_args.metrics == "spacy":
        outpath = os.path.join(task_output_dir, "spacy_metrics.tsv")
        if os.path.isfile(outpath):
            print("ERROR: spacy_metrics.tsv already exists.")
            return
        compute_all_spacy_metrics(model_args, experiment_args, metric_args, train_domains, eval_domains, outpath)
    print("Done.")


if __name__ == "__main__":
    main()
