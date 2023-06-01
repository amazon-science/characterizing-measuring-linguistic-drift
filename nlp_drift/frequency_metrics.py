"""
Utilities for token frequency divergence metrics, using a Hugging Face model
tokenizer.
"""

import os
import numpy as np
from tqdm import tqdm
import codecs
from scipy.spatial import distance
from transformers import AutoTokenizer

from .custom_settings import CUSTOM_TASK_NAME, CUSTOM_TEXT_FIELDS
from .dataset_utils import get_eval_datasets, get_train_datasets


# Returns the vocab size, along with a dictionary from domain names to tokenized datasets.
# model_args and experiment_args are defined in models.py.
def get_tokenized_datasets(model_args, experiment_args, domains, eval=False):
    if eval:
        datasets = get_eval_datasets(experiment_args.dataset_dir, experiment_args.task, domains)
    else:
        datasets = get_train_datasets(experiment_args.dataset_dir, experiment_args.task, domains)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    vocab_size = len(tokenizer) # This includes special tokens.
    # Define preprocessing steps.
    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_large", "sentiment_amazon_years"]:
        text_fields = ["review_body"]
    elif experiment_args.task in ["mnli"]:
        text_fields = ["premise", "hypothesis"]
    elif experiment_args.task == CUSTOM_TASK_NAME:
        text_fields = CUSTOM_TEXT_FIELDS
    def preprocess_function(examples):
        inputs = tuple([examples[text_field] for text_field in text_fields])
        return tokenizer(
            *inputs,
            padding=False,
            max_length=model_args.max_seq_length,
            truncation=True,
        )
    # Process and tokenize datasets.
    tokenized_datasets = dict()
    for domain, dataset in datasets.items():
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )
        tokenized_datasets[domain] = processed_dataset
    return vocab_size, tokenized_datasets


# Input: a Hugging Face dataset with tokenized inputs in the "input_ids" field.
# Output: a tensor of token frequencies (shape: vocab_size).
def get_frequencies(tokenized_dataset, vocab_size):
    frequencies = np.zeros(vocab_size)
    token_count = 0
    for example in tokenized_dataset:
        token_count += len(example["input_ids"])
        for token_id in example["input_ids"]:
            frequencies[token_id] += 1
    frequencies = frequencies / token_count
    return frequencies


# Computes the JS-distance and cross-entropy from a train token frequency
# distribution to the tokenized eval set. Also computes the metrics at the
# example level. The tokenized eval set should have tokenized inputs in the
# "input_ids" field.
def compute_frequency_metrics(train_frequency_distr, tokenized_eval_dataset):
    min_train_freq = np.amin(train_frequency_distr[train_frequency_distr > 0.0])
    train_logprobs = np.where(train_frequency_distr > 0.0, train_frequency_distr, min_train_freq)
    train_logprobs = np.log(train_logprobs)
    eval_frequency_distr = np.zeros(train_frequency_distr.shape[-1]) # The overall eval frequency distribution.
    eval_annotations = np.zeros((len(tokenized_eval_dataset), 3)) # JS-distance, XEnt, and sequence length.
    token_count = 0 # Total tokens in the eval set.
    example_distr = np.zeros(train_frequency_distr.shape[-1]) # A frequency distribution, reset for each eval example.
    for example_i, example in enumerate(tokenized_eval_dataset):
        token_count += len(example["input_ids"])
        example_logprobs = []
        example_distr[:] = 0.0
        for token_id in example["input_ids"]:
            example_logprobs.append(train_logprobs[token_id])
            example_distr[token_id] += 1
        eval_frequency_distr += example_distr # Update overall counts.
        # Example JS-distance.
        example_distr = example_distr / len(example["input_ids"])
        example_js_dist = distance.jensenshannon(train_frequency_distr, example_distr)
        eval_annotations[example_i, 0] = example_js_dist
        # Example XEnt.
        example_xent = -1.0 * np.mean(example_logprobs)
        eval_annotations[example_i, 1] = example_xent
        # Sequence length. Assume no padding during tokenization.
        eval_annotations[example_i, 2] = len(example["input_ids"])
    eval_frequency_distr = eval_frequency_distr / token_count
    # Note: computes base e.
    js_distance = distance.jensenshannon(train_frequency_distr, eval_frequency_distr)
    xent = np.mean(eval_annotations[:, 1])
    return js_distance, xent, eval_annotations


# Computes dataset-level drift metrics based on token frequencies, outputting to a tsv.
# Also saves the token frequency JS-distance and XEnt annotations for all eval
# examples (concatenated for all eval domains), relative to each train dataset.
# Saved in eval_frequency_annotations.npy in each train domain directory.
def compute_all_frequency_metrics(model_args, experiment_args, train_domains, eval_domains, outpath):
    print("Tokenizing train datasets.")
    vocab_size, tokenized_train_datasets = get_tokenized_datasets(model_args, experiment_args, train_domains, eval=False)
    print("Computing train frequency distributions.")
    train_frequency_distributions = np.zeros((len(train_domains), vocab_size))
    for train_domain_i, tokenized_dataset in tqdm(enumerate(tokenized_train_datasets.values())):
        train_frequency_distributions[train_domain_i] = get_frequencies(tokenized_dataset, vocab_size)
    del tokenized_train_datasets
    print("Tokenizing eval datasets.")
    vocab_size, tokenized_eval_datasets = get_tokenized_datasets(model_args, experiment_args, eval_domains, eval=True)
    print("Computing frequency JS distances, cross-entropies, and example-level metrics.")
    outfile = codecs.open(outpath, 'w', encoding='utf-8')
    outfile.write("TrainDomain\tEvalDomain\tfrequency_js_distance\tfrequency_xent\n".format(""))
    for train_domain_i, train_domain in enumerate(train_domains):
        print("Running train domain: {}".format(train_domain))
        all_eval_annotations = []
        for eval_domain_i, eval_domain in tqdm(enumerate(eval_domains)):
            tokenized_eval_dataset = tokenized_eval_datasets[eval_domain]
            js_distance, xent, eval_annotations = compute_frequency_metrics(train_frequency_distributions[train_domain_i],
                                                                            tokenized_eval_dataset)
            all_eval_annotations.append(eval_annotations)
            outfile.write("{0}\t{1}\t{2}\t{3}\n".format(train_domain, eval_domain, js_distance, xent))
        all_eval_annotations = np.concatenate(all_eval_annotations, axis=0)
        train_domain_dir = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix, train_domain)
        annotations_outpath = os.path.join(train_domain_dir, "eval_frequency_annotations.npy")
        os.makedirs(train_domain_dir, exist_ok=True)
        np.save(annotations_outpath, all_eval_annotations, allow_pickle=False)
        print("Saved frequency annotations for training domain: {}".format(train_domain))
    outfile.close()
    return True
