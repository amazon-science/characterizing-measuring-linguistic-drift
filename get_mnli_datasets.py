"""
Pulls MNLI datasets from Hugging Face, with domain splits.

Note: if the Hugging Face dataset download throws an error, you may need
to delete the Hugging Face dataset cache.
Sample usage:

python3 get_mnli_datasets.py --output_dir="datasets" \
--max_per_domain=-1

"""

from datasets import load_dataset, Dataset
import argparse
import os

from nlp_drift.constants import MNLI_TRAIN_GENRES, MNLI_EVAL_GENRES, HF_DATASET_CACHE_DIR


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_per_domain', type=int, required=True)
    return parser


# Take a subset of a dataset.
def subset_dataset(dataset, n_to_take):
    # Default to random subset.
    if n_to_take > len(dataset):
        print("WARNING: dataset size {0}, attempting to take {1}.".format(len(dataset), n_to_take))
        n_to_take = len(dataset)
    return Dataset.from_dict(dataset.shuffle(seed=42)[:n_to_take])

def main(args):
    # All data is in one subset but with train/eval splits, domains correspond to
    # a particular field.
    hf_dataset_name = "multi_nli"
    hf_subset = "default"
    hf_train_eval_splits = ["train", "validation_matched+validation_mismatched"]
    train_eval_domains = [MNLI_TRAIN_GENRES, MNLI_EVAL_GENRES]
    domain_field = "genre"

    # Load dataset.
    mnli_outdir = os.path.join(args.output_dir, "mnli")
    os.makedirs(mnli_outdir, exist_ok=True)
    for train_or_eval in range(2):
        # 0 = train; 1 = eval.
        # Note: train and eval datasets saved separately for each domain.
        # Load entire dataset, then filter for each domain.
        full_dataset = load_dataset(hf_dataset_name, hf_subset, split=hf_train_eval_splits[train_or_eval],
                                    use_auth_token=False, cache_dir=HF_DATASET_CACHE_DIR,
                                    streaming=False)
        for domain_name in train_eval_domains[train_or_eval]:
            print("Pulling domain: {}".format(domain_name))
            domain_examples = full_dataset.filter(lambda x: x[domain_field] == domain_name)
            if args.max_per_domain > 0 and args.max_per_domain < len(domain_examples):
                print("Subsetting domain: {}".format(domain_name))
                domain_examples = subset_dataset(domain_examples, args.max_per_domain)
            outpath = os.path.join(mnli_outdir, "{0}_{1}.tsv".format(
                        domain_name, "train" if train_or_eval==0 else "eval"))
            domain_examples = domain_examples.shuffle(seed=42)
            domain_examples.to_csv(outpath, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
