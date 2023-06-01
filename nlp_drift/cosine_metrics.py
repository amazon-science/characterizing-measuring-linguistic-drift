"""
Functions for embedding cosine similarity metrics.
"""

import os
import codecs
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
)

from .custom_settings import CUSTOM_TASK_NAME, CUSTOM_TEXT_FIELDS
from .constants import CONTENT_POS_TAGS
from .dataset_utils import get_dataset
from .spacy_metrics import STOP_WORDS

# Outputs:
# (1) the mean normed representation for the dataset (where the representation for
# each example is mean-pooled over token representations).
# (2) the dot product between each normalized example representation and a given reference representation.
# If the reference representation is the mean of nomed training representations, then this is equal to the
# mean cosine similarity between the example and the set of training representations.
# Shape: n_examples.
# (3) the mean normed representation for each token.
# (4) the lexical semantic similarity between each example and the reference. This is computed
# as the mean dot product between each normed token representation in the example and
# the reference representation for that token. This can be filtered to only content word
# tokens using the content_word_mask.
# Shape: (n_examples, 2).
# Columns are lexical semantic similarities unfiltered or filtered to content tokens.
# (5) vector of token counts.
def get_representation_data(model, processed_dataset, eval_batch_size=16, layers=[-2, -1],
                            reference_representation=None, reference_token_reps=None, content_word_mask=None):
    hidden_size = model.config.hidden_size
    sum_normed_representations = None # Will aggregate the sum of normed representations.
    example_similarities = None # The dot product between each normed example and the reference.
    sum_normed_token_reps = np.zeros((model.config.vocab_size, hidden_size)) # Aggregates the sum of normed representations for each token.
    # Mean dot product between each token and the reference for that token, for each example.
    # The first column is the mean over all shared tokens in the training and eval domains;
    # the second column is the mean over all shared content tokens, as defined by content_word_mask.
    example_token_similarities = None
    token_counts = np.zeros(model.config.vocab_size, dtype=int)
    # Compute dot products from individual examples to a reference.
    if reference_representation is not None:
        example_similarities = np.zeros(len(processed_dataset))
        reshaped_reference_representation = reference_representation.reshape(1, -1)
    # Compute dot products for individual examples, matching tokens (lexical semantic similarity).
    if reference_token_reps is not None:
        example_token_similarities = np.zeros((len(processed_dataset), 2))
    # Run model.
    if torch.cuda.is_available():
        model = model.cuda()
    for batch_start_i in tqdm(range(0, len(processed_dataset), eval_batch_size)):
        batch = processed_dataset[batch_start_i:batch_start_i+eval_batch_size]
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"]).bool()
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True, return_dict=True)
        hidden_states = torch.zeros_like(outputs["hidden_states"][-1]) # Shape: batch_size, max_seq_length, hidden_size.
        for layer in layers:
            hidden_states += outputs["hidden_states"][layer].detach()
        hidden_states = (hidden_states / len(layers)).cpu() # Average over the selected layers.
        del outputs # Delete before the next batch runs.
        n_examples = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        example_reps = np.zeros((n_examples, hidden_size))
        # For each example in the batch.
        for example_i in range(hidden_states.shape[0]):
            # Shape: seq_length, hidden_size.
            example_hidden_states = np.array(hidden_states[example_i][attention_mask[example_i]])
            # Token representations: seq_length, hidden_size.
            normed_token_reps = example_hidden_states / np.linalg.norm(example_hidden_states, axis=-1, keepdims=True)
            example_token_similarity = 0.0 # Total similarity over all tokens in the example.
            example_shared_token_count = 0
            example_content_token_similarity = 0.0 # Content tokens only.
            example_shared_content_token_count = 0
            for token_i, token_id in enumerate(input_ids[example_i][attention_mask[example_i]]):
                sum_normed_token_reps[token_id] = sum_normed_token_reps[token_id] + normed_token_reps[token_i]
                token_counts[token_id] += 1
                # Compute the lexical semantic similarity.
                if reference_token_reps is not None:
                    # Token must have appeared in training domain.
                    if not np.isnan(reference_token_reps[token_id][0].item()):
                        token_similarity = np.dot(normed_token_reps[token_i], reference_token_reps[token_id])
                        example_token_similarity += token_similarity
                        example_shared_token_count += 1
                        if content_word_mask is None or content_word_mask[token_id]: # If content token, also increment these.
                            # If content_word_mask is None, treat all tokens as content tokens.
                            example_content_token_similarity += token_similarity
                            example_shared_content_token_count += 1
            # Take the mean for each example for lexical semantic similarity.
            if reference_token_reps is not None:
                # Over all tokens in the example.
                if example_shared_token_count > 0:
                    example_token_similarity = example_token_similarity / example_shared_token_count
                else:
                    example_token_similarity = np.nan
                example_token_similarities[batch_start_i + example_i, 0] = example_token_similarity
                # Over all content tokens in the example.
                if example_shared_content_token_count > 0:
                    example_content_token_similarity = example_content_token_similarity / example_shared_content_token_count
                else:
                    example_content_token_similarity = np.nan
                example_token_similarities[batch_start_i + example_i, 1] = example_content_token_similarity
            # Example representation: mean over token representations.
            example_representation = np.mean(example_hidden_states, axis=0)
            example_reps[example_i, :] = example_representation
        # Normed example representations. Shape: n_examples, hidden_size.
        normed_batch = example_reps / np.linalg.norm(example_reps, axis=-1, keepdims=True)
        # Dot with the reference vector. Shape: 1, n_examples.
        if reference_representation is not None:
            batch_similarities = np.matmul(reshaped_reference_representation, normed_batch.T).reshape(-1)
            example_similarities[batch_start_i:batch_start_i+n_examples] = batch_similarities
        # Sum the normed example representations.
        sum_normed_batch = np.sum(normed_batch, axis=0)
        sum_normed_representations = sum_normed_batch if sum_normed_representations is None else sum_normed_representations + sum_normed_batch
    mean_normed_representation = sum_normed_representations / len(processed_dataset)
    mean_normed_token_reps = sum_normed_token_reps / token_counts.reshape(-1, 1) # If token count is zero, sets that representation to np.nans.
    return mean_normed_representation, example_similarities, mean_normed_token_reps, example_token_similarities, token_counts


# Gets a mask for content words given an input file path.
# The input filepath should be a TSV of words and space-separated POS tags for
# each word (indicating the possible POS tags for that word).
# These words will be mapped to token ids, and any word/token that could have a content
# word POS tag will be set to True. Stop words are excluded.
# Returns a boolean vector with shape: vocab_size.
# Content tokens are set to True.
def get_content_word_mask(tokenizer, content_words_path):
    def word_to_ids(word):
        stripped_word = word.strip().lower()
        # Note: this is designed for the RoBERTa tokenizer.
        word_ids = []
        # Word with space before.
        id = tokenizer.encode(" " + stripped_word, add_special_tokens=False)
        if len(id) == 1:
            word_ids.append(id[0])
        # Capitalized word.
        id = tokenizer.encode(stripped_word.capitalize(), add_special_tokens=False)
        if len(id) == 1:
            word_ids.append(id[0])
        # Capitalized word with space before.
        id = tokenizer.encode(" " + stripped_word.capitalize(), add_special_tokens=False)
        if len(id) == 1:
            word_ids.append(id[0])
        return word_ids
    # Check for content words path.
    if not os.path.isfile(content_words_path):
        print("Content words path not found.")
        return None
    # Read content words file and get content token ids.
    content_token_ids = set()
    infile = codecs.open(content_words_path, 'rb', encoding='utf-8')
    content_pos_tags = set(CONTENT_POS_TAGS)
    for line in infile:
        word, pos_set = tuple(line.strip().split("\t"))
        if not bool(re.search("[a-zA-Z1-9]", word.strip())):
            # No letters or numbers. Exclude.
            continue
        # If the POS tags intersect with the content word POS tags.
        pos_set = set(pos_set.split(" "))
        if bool(pos_set & content_pos_tags):
            content_token_ids.update(word_to_ids(word))
    # Exclude stop words.
    for stop_word in STOP_WORDS:
        for id in word_to_ids(stop_word):
            content_token_ids.discard(id)
    # Return the mask.
    content_word_mask = np.zeros(tokenizer.vocab_size, dtype=bool)
    for id in content_token_ids:
        content_word_mask[id] = True
    return content_word_mask


# Computes cosine metrics for a single train domain.
# Uses the fine-tuned model for that domain (from finetune.py) unless
# use_pretrained is set to some other model, e.g. the pre-trained roberta-base.
# Outputs the dataset-level mean cosine similarities to:
# experiment_output_dir/task/train_domain/[finetuned/pretrained]_cosine_metrics.tsv
#
# Outputs the example-level cosine similarities (the mean cosine similarity between
# each example embedding and all training example embeddings) to:
# experiment_output_dir/task/train_domain/eval_[finetuned/pretrained]_cosine_annotations.npy
# With shape: (n_eval_examples).
#
# Optionally outputs the lexical semantic similarities too (sometimes shortened
# to semantic similarity here), at both the dataset level and example level.
# Example-level lexical semantic similarities are outputted to:
# experiment_output_dir/task/train_domain/eval_[finetuned/pretrained]_semantic_annotations.npy
# With shape: (n_eval_examples, 2).
# Columns correspond to lexical semantic similarities when unfiltered or filtered to content tokens.
# If computing lexical semantic similarities, content_words_path defines the
# content words (see get_content_word_mask()).
def compute_cosine_metrics(model_args, experiment_args, training_args, train_domain, eval_datasets, use_pretrained="",
                           layers=[-1,-2], include_lexical_semantic_similarity=False, content_words_path=""):
    print("Running model trained on domain: {}".format(train_domain))
    # Output directory.
    model_path = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix, train_domain)
    os.makedirs(model_path, exist_ok=True) # Make output directory for train domain.
    output_path = os.path.join(model_path, "finetuned_cosine_metrics.tsv")
    output_annotations_path = os.path.join(model_path, "eval_finetuned_cosine_annotations.npy")
    train_mean_normed_path = os.path.join(model_path, "train_finetuned_mean_normed_vector.npy")
    if include_lexical_semantic_similarity:
        train_mean_normed_tokens_path = os.path.join(model_path, "train_finetuned_mean_normed_token_vectors.npy")
        output_semantic_annotations_path = os.path.join(model_path, "eval_finetuned_semantic_annotations.npy")

    # Use a pre-trained model to extract representations.
    if use_pretrained != "":
        output_path = os.path.join(model_path, "pretrained_cosine_metrics.tsv")
        output_annotations_path = os.path.join(model_path, "eval_pretrained_cosine_annotations.npy")
        train_mean_normed_path = os.path.join(model_path, "train_pretrained_mean_normed_vector.npy")
        if include_lexical_semantic_similarity:
            train_mean_normed_tokens_path = os.path.join(model_path, "train_pretrained_mean_normed_token_vectors.npy")
            output_semantic_annotations_path = os.path.join(model_path, "eval_pretrained_semantic_annotations.npy")
        # Set the actual model path. E.g. "roberta-base".
        model_path = use_pretrained

    if os.path.isfile(output_annotations_path):
        print("Skipping: output file already exists: {}".format(output_annotations_path))
        return False

    # Load model.
    config = AutoConfig.from_pretrained(model_path, cache_dir=model_args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # Same as in finetune.py, but with padding.
    if experiment_args.task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        text_fields = ["review_body"]
    elif experiment_args.task in ["mnli"]:
        text_fields = ["premise", "hypothesis"]
    elif experiment_args.task == CUSTOM_TASK_NAME:
        text_fields = CUSTOM_TEXT_FIELDS
    else:
        print("Unrecognized task: {}".format(experiment_args.task))
    def preprocess_function(examples):
        inputs = tuple([examples[text_field] for text_field in text_fields])
        return tokenizer(
            *inputs,
            padding=True,
            max_length=model_args.max_seq_length,
            truncation=True,
        )

    content_word_mask = get_content_word_mask(tokenizer, content_words_path)
    print("{} content tokens found.".format(np.sum(content_word_mask)))

    # Get mean normed representation from the training dataset.
    # The mean pairwise cosine similarity between sets of vectors A and B is equal
    # to the dot product between the mean of the normed vectors in A and B.
    train_mean_normed_representation = None
    train_mean_normed_token_reps = None
    if os.path.isfile(train_mean_normed_path):
        train_mean_normed_representation = np.load(train_mean_normed_path, allow_pickle=False)
    if include_lexical_semantic_similarity and os.path.isfile(train_mean_normed_tokens_path):
        train_mean_normed_token_reps = np.load(train_mean_normed_tokens_path, allow_pickle=False)
    if (train_mean_normed_representation is None) or (include_lexical_semantic_similarity and train_mean_normed_token_reps is None):
        # Could not load the mean normed representations. Compute them here.
        train_dataset, _, _ = get_dataset(experiment_args.dataset_dir, experiment_args.task, train_domain)
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            processed_train_dataset = train_dataset.map(
                preprocess_function,
                batched=True, batch_size=training_args.per_device_eval_batch_size,
                desc="Running tokenizer on train dataset",
            )
        train_mean_normed_representation, _, train_mean_normed_token_reps, _, _ = get_representation_data(
                model, processed_train_dataset, eval_batch_size=training_args.per_device_eval_batch_size,
                layers=layers)
        np.save(train_mean_normed_path, train_mean_normed_representation, allow_pickle=False)
        if include_lexical_semantic_similarity:
            np.save(train_mean_normed_tokens_path, train_mean_normed_token_reps, allow_pickle=False)

    # Get cosine similarities from each eval dataset.
    outfile = codecs.open(output_path, 'w', encoding='utf-8')
    outfile.write("TrainDomain\tEvalDomain\tEvalExamples")
    if use_pretrained == "":
        outfile.write("\tFinetunedCosineSimilarity")
        if include_lexical_semantic_similarity:
            outfile.write("\tFinetunedSemanticSimilarity\tFinetunedSemanticSimilarityContent")
        outfile.write("\n")
    else:
        outfile.write("\tPretrainedCosineSimilarity")
        if include_lexical_semantic_similarity:
            outfile.write("\tPretrainedSemanticSimilarity\tPretrainedSemanticSimilarityContent")
        outfile.write("\n")
    all_eval_similarities = [] # Example similarities for each eval domain.
    all_eval_semantic_similarities = [] # Example lexical semantic similarities for each eval domain.
    for eval_domain, eval_dataset in eval_datasets.items():
        print("Evaluating train/eval domain pair: {0}, {1}".format(train_domain, eval_domain))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            processed_eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True, batch_size=training_args.per_device_eval_batch_size,
                desc="Running tokenizer on validation dataset",
            )
        eval_mean_normed_representation, eval_similarities, eval_mean_normed_token_reps, eval_semantic_similarities, eval_token_counts = get_representation_data(
                model, processed_eval_dataset, reference_representation=train_mean_normed_representation,
                reference_token_reps = train_mean_normed_token_reps if include_lexical_semantic_similarity else None,
                content_word_mask=content_word_mask,
                eval_batch_size=training_args.per_device_eval_batch_size, layers=layers)
        all_eval_similarities.append(eval_similarities)
        all_eval_semantic_similarities.append(eval_semantic_similarities)
        # Write outputs.
        dataset_cosine_similarity = np.dot(train_mean_normed_representation.reshape(-1), eval_mean_normed_representation.reshape(-1))
        outfile.write("{0}\t{1}\t{2}\t{3}".format(train_domain, eval_domain, len(eval_dataset), dataset_cosine_similarity))
        if include_lexical_semantic_similarity:
            # Shape: vocab_size.
            dataset_token_similarities = np.sum(np.multiply(train_mean_normed_token_reps, eval_mean_normed_token_reps), axis=-1)
            # Weight by eval frequency.
            eval_token_freqs = eval_token_counts / np.sum(eval_token_counts)
            dataset_token_similarity = np.nansum(np.multiply(dataset_token_similarities, eval_token_freqs))
            # Filtered to content tokens.
            dataset_content_token_similarities = dataset_token_similarities[content_word_mask]
            eval_content_token_freqs = eval_token_counts[content_word_mask] / np.sum(eval_token_counts[content_word_mask])
            dataset_content_token_similarity = np.nansum(np.multiply(dataset_content_token_similarities, eval_content_token_freqs))
            # Write lexical semantic similarity outputs.
            outfile.write("\t{0}\t{1}".format(dataset_token_similarity, dataset_content_token_similarity))
        outfile.write("\n")
    outfile.close()
    if include_lexical_semantic_similarity:
        all_eval_semantic_similarities = np.concatenate(all_eval_semantic_similarities, axis=0)
        np.save(output_semantic_annotations_path, all_eval_semantic_similarities, allow_pickle=False)
    all_eval_similarities = np.concatenate(all_eval_similarities, axis=0)
    np.save(output_annotations_path, all_eval_similarities, allow_pickle=False)
    print("Completed cosine computations for training domain: {}".format(train_domain))
    return True
