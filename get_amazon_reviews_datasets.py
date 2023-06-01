"""
Pulls Amazon reviews datasets from Hugging Face.

Outputs up to max_per_category randomly sampled reviews per product category
into the output subdirectory: sentiment_amazon_categories.

Outputs up to max_per_category_year randomly sampled reviews per year and per
product category into the output subdirectory: sentiment_amazon_categories_years.

All random samples are polarity balanced. To compile the domain-balanced year
datasets, run compile_amazon_reviews_years.py after running this script. Here,
note that the max_per_category_year examples (examples per category per year)
are pulled independently of the max_per_category size.

Note: if the Hugging Face dataset download throws an error, you may need
to delete the Hugging Face dataset cache.
Sample usage:

python3 get_amazon_reviews_datasets.py --output_dir="datasets" \
--max_per_category=100000 --max_per_category_year=5000

"""

from datasets import load_dataset, concatenate_datasets
import argparse
import os
import shutil
import numpy as np

from nlp_drift.constants import AMAZON_REVIEWS_CATEGORY_SUBSETS, HF_DATASET_CACHE_DIR


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_per_category', type=int, required=True)
    parser.add_argument('--max_per_category_year', type=int, required=True)
    return parser


# Consider 1-2 star reviews as negative, 4-5 star reviews as positive,
# and take a balanced random subset of positive and negative examples.
def subset_dataset_balanced(dataset, n_to_take):
    # Balance positive and negative polarity.
    star_ratings = np.array(dataset["star_rating"], dtype=np.int8)
    negative_indices = np.nonzero(star_ratings < 3)[0]
    positive_indices = np.nonzero(star_ratings > 3)[0]
    n_to_take_each = n_to_take // 2
    # Must be enough negative and positive examples.
    if n_to_take_each > negative_indices.shape[0]:
        print("WARNING: negative dataset size {0}, attempting to take {1}.".format(negative_indices.shape[0], n_to_take_each))
        n_to_take_each = negative_indices.shape[0]
    if n_to_take_each > positive_indices.shape[0]:
        print("WARNING: positive dataset size {0}, attempting to take {1}.".format(positive_indices.shape[0], n_to_take_each))
        n_to_take_each = positive_indices.shape[0]
    # Randomly sample.
    negative_indices = np.random.choice(negative_indices, size=n_to_take_each, replace=False)
    positive_indices = np.random.choice(positive_indices, size=n_to_take_each, replace=False)
    negative = dataset.select(negative_indices)
    positive = dataset.select(positive_indices)
    balanced_subset = concatenate_datasets([negative, positive]).shuffle(seed=42)
    if "polarity" not in balanced_subset.column_names:
        def add_polarity(example):
            # Note: examples with rating exactly 3 were discarded above.
            example["polarity"] = int(example["star_rating"] > 3)
            return example
        balanced_subset = balanced_subset.map(add_polarity)
    return balanced_subset


# Return a dictionary mapping years to year subsets.
def get_year_samples(dataset, n_per_year):
    def add_year(example):
        try:
            # Assume year is the first four characters.
            example["year"] = int(example["review_date"][:4])
        except:
            print("Error extracting year from review date: {}".format(example["review_date"]))
        return example
    year_annotated_dataset = dataset.map(add_year)
    year_subsets = dict()
    # Balance positive and negative polarity.
    year_annotations = np.array(year_annotated_dataset["year"], dtype=np.int32)
    years = set(year_annotations)
    for year in years:
        n_this_year = n_per_year
        year_indices = np.nonzero(year_annotations == year)[0]
        year_filtered = year_annotated_dataset.select(year_indices)
        year_filtered = subset_dataset_balanced(year_filtered, n_per_year)
        year_subsets[year] = year_filtered
    return year_subsets


def main(args):
    # All data is in one split, product categories correspond to subsets.
    # https://huggingface.co/datasets/amazon_us_reviews
    hf_dataset_name = "amazon_us_reviews"
    hf_category_subsets = AMAZON_REVIEWS_CATEGORY_SUBSETS
    hf_dataset_split = "train" # All data is in the train split for this dataset.
    categories_outdir = os.path.join(args.output_dir, "sentiment_amazon_categories")
    os.makedirs(categories_outdir, exist_ok=True)
    categories_years_outdir = os.path.join(args.output_dir, "sentiment_amazon_categories_years")
    os.makedirs(categories_years_outdir, exist_ok=True)

    # Load each category.
    for category_name, subset_names in hf_category_subsets.items():
        category_examples = None
        category_year_samples = dict()
        # Load each subset of the category.
        for subset_i, subset_name in enumerate(subset_names):
            print("Pulling category, subset: {0}, {1}".format(category_name, subset_name))
            data_subset = load_dataset(hf_dataset_name, subset_name, split=hf_dataset_split,
                                  use_auth_token=False, cache_dir=HF_DATASET_CACHE_DIR,
                                  streaming=False)
            random_sample = subset_dataset_balanced(data_subset, args.max_per_category)
            category_examples = random_sample if category_examples is None else concatenate_datasets([category_examples, random_sample])
            year_samples = get_year_samples(data_subset, args.max_per_category_year)
            for year, year_sample in year_samples.items():
                if year in category_year_samples:
                    category_year_samples[year] = concatenate_datasets([category_year_samples[year], year_sample])
                else:
                    category_year_samples[year] = year_sample
        # Save the category examples.
        outpath = os.path.join(categories_outdir, "{0}.tsv".format(category_name))
        if len(category_examples) > args.max_per_category:
            category_examples = subset_dataset_balanced(category_examples, args.max_per_category)
        category_examples.to_csv(outpath, sep="\t", index=False)
        # Save the year-balanced category examples.
        for year, year_sample in category_year_samples.items():
            if len(year_sample) > args.max_per_category_year:
                category_year_samples[year] = subset_dataset_balanced(year_sample, args.max_per_category_year)
        category_year_balanced_examples = concatenate_datasets(list(category_year_samples.values()))
        outpath = os.path.join(categories_years_outdir, "{0}.tsv".format(category_name))
        category_year_balanced_examples.to_csv(outpath, sep="\t", index=False)
        # To save disk space, delete Hugging Face cache after each category.
        shutil.rmtree(HF_DATASET_CACHE_DIR)

    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
