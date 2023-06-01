"""
Utilities to handle domains and datasets for drift experiments.
"""

import os
import datasets
from datasets import Dataset, concatenate_datasets

from .custom_settings import (CUSTOM_TASK_NAME, CUSTOM_TEXT_FIELDS,
    CUSTOM_LABEL_NAME, CUSTOM_TRAIN_DOMAINS, CUSTOM_EVAL_DOMAINS)
from .constants import (MNLI_TRAIN_GENRES, MNLI_EVAL_GENRES, AMAZON_REVIEWS_CATEGORY_SUBSETS,
    AMAZON_REVIEWS_YEARS, HF_DATASET_CACHE_DIR)


# Filter a dataset for empty text fields.
def filter_dataset(dataset, text_fields):
    filtered_dataset = dataset
    for text_field in text_fields:
        filtered_dataset = filtered_dataset.filter(lambda x: x[text_field] != None)
    return filtered_dataset

# Returns train, eval, test dataset.
# Automatically selects the train/eval/test proportion based on the dataset (task and domain).
# Renames the label_name column to "labels" for use in Hugging Face models.
def get_dataset(dataset_dir, task, domain, columns_to_keep=[]):
    # Add the default columns to keep. The label_name will be renamed to "labels".
    if task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        columns_to_keep.extend(["review_body", "polarity"])
        text_fields = ["review_body"] # To filter empty examples.
        label_name = "polarity"
    elif task == "mnli":
        columns_to_keep.extend(["premise", "hypothesis", "label"])
        text_fields = ["premise", "hypothesis"] # To filter empty examples.
        label_name = "label"
    elif task == CUSTOM_TASK_NAME:
        columns_to_keep.extend(CUSTOM_TEXT_FIELDS)
        text_fields = CUSTOM_TEXT_FIELDS
        label_name = CUSTOM_LABEL_NAME
    # If split_type is "proportions", assume that the train/eval/test datasets
    # are all in one file, split by some train/eval/test proportion.
    # If split_type is "indices", assume fixed indices for train/eval/test
    # splits (e.g. test set is always the last 5000 examples).
    # If split_type is "files", assume that the train/eval/test datasets are
    # all saved separately (default).
    split_type = "files"
    if task == "mnli":
        split_type = "files"
    elif task in ["sentiment_amazon_categories"]:
        # 70/20/10 train/eval/test split.
        split_type = "proportions"
        train_proportion, eval_proportion, test_proportion = 0.70, 0.20, 0.10
    elif task in ["sentiment_amazon_categories_small"]:
        split_type = "indices"
        if domain in ["gift_card"]:
            # Domains with less than 20K total examples: 2K eval and test.
            train_end_idx = -4000
            eval_end_idx = -2000
        else:
            # Domains with >20K total examples: 5K eval and test.
            train_end_idx = -10000
            eval_end_idx = -5000
        test_end_idx = "max"
    elif task in ["sentiment_amazon_years"] or task == CUSTOM_TASK_NAME:
        split_type = "files"
    else:
        print("WARNING: unrecognized task/domain pair: {0}/{1}".format(task, domain))
        return None, None, None
    # Get the full train/eval/test dataset, and the train/eval/test indices.
    datasets.disable_progress_bar()
    if split_type == "proportions":
        dataset_path = os.path.join(dataset_dir, task, "{}.tsv".format(domain))
        full_dataset = Dataset.from_csv(dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
        train_end_idx = int(len(full_dataset) * train_proportion)
        eval_end_idx = train_end_idx + int(len(full_dataset) * eval_proportion)
        test_end_idx = eval_end_idx + int(len(full_dataset) * test_proportion)
        # Ensure end indices are in [0, n_examples].
        train_end_idx = max(0, min(train_end_idx, len(full_dataset)))
        eval_end_idx = max(0, min(eval_end_idx, len(full_dataset)))
        test_end_idx = max(0, min(test_end_idx, len(full_dataset)))
    elif split_type == "indices":
        # Assume train/eval/test end indices have already been set.
        dataset_path = os.path.join(dataset_dir, task, "{}.tsv".format(domain))
        full_dataset = Dataset.from_csv(dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
        train_end_idx = len(full_dataset) if train_end_idx == "max" else train_end_idx
        eval_end_idx = len(full_dataset) if eval_end_idx == "max" else eval_end_idx
        test_end_idx = len(full_dataset) if test_end_idx == "max" else test_end_idx
    elif split_type == "files":
        full_dataset = []
        train_end_idx = 0
        eval_end_idx = 0
        test_end_idx = 0
        train_dataset_path = os.path.join(dataset_dir, task, "{}_train.tsv".format(domain))
        if os.path.isfile(train_dataset_path):
            train_dataset = Dataset.from_csv(train_dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
            train_end_idx = len(train_dataset)
            full_dataset.append(train_dataset)
        eval_dataset_path = os.path.join(dataset_dir, task, "{}_eval.tsv".format(domain))
        if os.path.isfile(eval_dataset_path):
            eval_dataset = Dataset.from_csv(eval_dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
            eval_end_idx = train_end_idx + len(eval_dataset)
            full_dataset.append(eval_dataset)
        test_dataset_path = os.path.join(dataset_dir, task, "{}_test.tsv".format(domain))
        if os.path.isfile(test_dataset_path):
            test_dataset = Dataset.from_csv(test_dataset_path, sep="\t", cache_dir=HF_DATASET_CACHE_DIR)
            test_end_idx = eval_end_idx + len(test_dataset)
            full_dataset.append(test_dataset)
        full_dataset = concatenate_datasets(full_dataset)
    # Rename labels.
    columns_to_keep.append("labels")
    if label_name in full_dataset.column_names:
        full_dataset = full_dataset.rename_column(label_name, "labels")
        if label_name in columns_to_keep:
            columns_to_keep.remove(label_name)
    elif "labels" not in full_dataset.column_names:
        print("WARNING: neither {0} nor labels column found for domain {1}.".format(label_name, domain))
    # Remove extra columns.
    full_dataset = full_dataset.remove_columns(
        [col for col in full_dataset.column_names if col not in columns_to_keep])
    # Get train, eval, and test datasets.
    # Note: we wait until after indexing the train/eval/test set to filter
    # examples with empty text fields.
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if train_end_idx != 0:
        train_dataset = Dataset.from_dict(full_dataset[:train_end_idx])
        train_dataset = filter_dataset(train_dataset, text_fields)
    if train_end_idx != eval_end_idx:
        eval_dataset = Dataset.from_dict(full_dataset[train_end_idx:eval_end_idx])
        eval_dataset = filter_dataset(eval_dataset, text_fields)
    if eval_end_idx != test_end_idx:
        test_dataset = Dataset.from_dict(full_dataset[eval_end_idx:test_end_idx])
        test_dataset = filter_dataset(test_dataset, text_fields)
    datasets.enable_progress_bar()
    del full_dataset
    return train_dataset, eval_dataset, test_dataset


# Returns a dictionary mapping domains to eval datasets.
def get_eval_datasets(dataset_dir, task, domains, columns_to_keep=[]):
    eval_datasets = dict()
    for domain in domains:
        _, eval_dataset, _ = get_dataset(dataset_dir, task, domain, columns_to_keep=columns_to_keep)
        eval_datasets[domain] = eval_dataset
    return eval_datasets

# Returns a dictionary mapping domains to train datasets.
def get_train_datasets(dataset_dir, task, domains, columns_to_keep=[]):
    train_datasets = dict()
    for domain in domains:
        train_dataset, _, _ = get_dataset(dataset_dir, task, domain, columns_to_keep=columns_to_keep)
        train_datasets[domain] = train_dataset
    return train_datasets


# Returns a list of domains, either for training or evaluation.
def get_all_domains(task, eval=False):
    if task in ["sentiment_amazon_categories", "sentiment_amazon_categories_large"]:
        # Same domains for training and evaluation.
        amazon_categories = list(AMAZON_REVIEWS_CATEGORY_SUBSETS.keys())
        return amazon_categories
    elif task == "sentiment_amazon_years":
        # Only evaluating on future years will have to be handled manually by
        # the scripts. For now, same domains for training and evaluation.
        amazon_years = AMAZON_REVIEWS_YEARS
        return ["year" + str(year) for year in amazon_years]
    elif task == "mnli":
        mnli_domains = MNLI_EVAL_GENRES if eval else MNLI_TRAIN_GENRES
        return ["mnli_" + domain for domain in mnli_domains]
    elif task == CUSTOM_TASK_NAME:
        # Use settings from custom_settings.py.
        return CUSTOM_EVAL_DOMAINS if eval else CUSTOM_TRAIN_DOMAINS
    else:
        print("WARNING: unrecognized task name: {}".format(task))
        return []
