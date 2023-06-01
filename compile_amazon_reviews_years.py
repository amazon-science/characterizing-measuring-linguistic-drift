"""
Run get_amazon_reviews_datasets.py first.
Generates a product category-balanced and polarity-balanced dataset for each
year between 2001 and 2015 (inclusive). Outputs tsv files into:
dataset_dir/sentiment_amazon_years.
Sample usage:

python3 compile_amazon_reviews_years.py --dataset_dir="datasets" \
--n_eval=5000 --n_test=5000

"""

import argparse
import numpy as np
import os
import datasets
from datasets import Dataset, concatenate_datasets
import codecs

from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.constants import HF_DATASET_CACHE_DIR, AMAZON_REVIEWS_YEARS
from get_amazon_reviews_datasets import subset_dataset_balanced


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--n_eval', type=int, required=True)
    parser.add_argument('--n_test', type=int, required=True)
    parser.add_argument('--min_per_domain', type=int, default=100)
    return parser


def main(args):
    train_domains = get_all_domains("sentiment_amazon_categories", eval=False)
    year_count_dict = dict()
    domain_dataset_dict = dict()
    for train_i, train_domain in enumerate(train_domains):
        print("Getting dataset for domain: {}".format(train_domain))
        dataset_path = os.path.join(args.dataset_dir, "sentiment_amazon_categories_years", "{}.tsv".format(train_domain))
        try:
            domain_dataset = Dataset.from_csv(dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
        except:
            print("Dataset not found: {}".format(dataset_path))
            continue
        # Add to dict.
        domain_dataset_dict[train_domain] = domain_dataset
        # Count examples in each year.
        example_years = np.array(domain_dataset["year"], dtype=int)
        years = set(example_years)
        for year in years:
            domain_year_count = np.sum(example_years == year)
            if year not in year_count_dict:
                year_count_dict[year] = np.zeros(len(train_domains), dtype=int)
            year_count_dict[year][train_i] = domain_year_count

    # Write year_count_dict to tsv.
    year_count_outpath = os.path.join(args.dataset_dir, "sentiment_amazon_categories_year_counts.tsv")
    outfile = codecs.open(year_count_outpath, 'w', encoding='utf-8')
    outfile.write("Year\t" + "\t".join(train_domains) + "\n")
    years = list(year_count_dict.keys())
    years.sort()
    for year in years:
        outfile.write("{}\t".format(year))
        year_counts = [str(count) for count in year_count_dict[year]]
        outfile.write("\t".join(year_counts) + "\n")
    outfile.close()

    # For each domain, take the minimum number of reviews of any year in the range.
    # Remove domains with less than min_per_domain minimum examples per year.
    # Note that these subsets were already polarity-balanced, so polarity-balanced
    # subsets exist for each year.
    target_years = AMAZON_REVIEWS_YEARS

    # Create datasets.
    count_array = np.zeros((len(target_years), len(train_domains)), dtype=int)
    for year_i, year in enumerate(target_years):
        count_array[year_i] = year_count_dict[year]
    # Shape: len(train_domains).
    total_target_counts = np.amin(count_array, axis=0)
    total_target_counts[total_target_counts < args.min_per_domain] = 0
    # Get target count per domain for training, eval, and test.
    target_proportions = total_target_counts / np.sum(total_target_counts)
    eval_target_counts = (target_proportions * args.n_eval).astype(int)
    test_target_counts = (target_proportions * args.n_test).astype(int)
    train_target_counts = total_target_counts - eval_target_counts - test_target_counts

    print("Total examples per year: {}".format(np.sum(total_target_counts)))
    total_target_count_dict = dict(zip(train_domains, list(total_target_counts)))
    print("Target example counts per year: {}".format(total_target_count_dict))

    # Check if datasets already exist.
    outdir = os.path.join(args.dataset_dir, "sentiment_amazon_years")
    if os.path.isdir(outdir):
        print("ERROR: output directory already exists.")
        return

    # Create year datasets.
    os.mkdir(outdir)
    def split_dataset(dataset, counts_tuple):
        subset_dataset = subset_dataset_balanced(dataset, np.sum(counts_tuple))
        subset_dataset = subset_dataset.shuffle(seed=42)
        end_idx = 0
        start_idx = 0
        final_subsets = []
        for count in counts_tuple:
            start_idx = end_idx
            end_idx += count
            final_subset = Dataset.from_dict(subset_dataset[start_idx:end_idx])
            final_subsets.append(final_subset)
        assert end_idx == np.sum(counts_tuple)
        assert np.sum([len(final_subset) for final_subset in final_subsets]) == np.sum(counts_tuple)
        return tuple(final_subsets)
    for year in target_years:
        train_dataset = []
        eval_dataset = []
        test_dataset = []
        # Collect year dataset for each domain.
        for train_i, train_domain in enumerate(train_domains):
            domain_dataset = domain_dataset_dict[train_domain]
            # Filter to target year.
            example_years = np.array(domain_dataset["year"], dtype=int)
            domain_dataset = domain_dataset.select(np.nonzero(example_years == year)[0])
            domain_dataset = domain_dataset.add_column(name="domain", column=[train_domain]*len(domain_dataset))
            # Target counts for this domain.
            counts_tuple = tuple([train_target_counts[train_i], eval_target_counts[train_i], test_target_counts[train_i]])
            # print("Domain dataset size {0}, pulling {1}.".format(len(domain_dataset), np.sum(counts_tuple)))
            domain_train, domain_eval, domain_test = split_dataset(domain_dataset, counts_tuple)
            train_dataset.append(domain_train)
            eval_dataset.append(domain_eval)
            test_dataset.append(domain_test)
        train_dataset = concatenate_datasets(train_dataset).shuffle(seed=42)
        eval_dataset = concatenate_datasets(eval_dataset).shuffle(seed=42)
        test_dataset = concatenate_datasets(test_dataset).shuffle(seed=42)
        # Save train/eval/test for this year.
        outpath = os.path.join(outdir, "year{}_train.tsv".format(year))
        train_dataset.to_csv(outpath, sep="\t", index=False)
        outpath = os.path.join(outdir, "year{}_eval.tsv".format(year))
        eval_dataset.to_csv(outpath, sep="\t", index=False)
        outpath = os.path.join(outdir, "year{}_test.tsv".format(year))
        test_dataset.to_csv(outpath, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
