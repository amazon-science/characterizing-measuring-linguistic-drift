"""
Utilities for spaCy distance metrics, using the spaCy tokenizer and POS tagger.
E.g. POS n-gram divergences, content word frequency divergences, etc.
"""

import os
import codecs
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle
import numpy as np
from datasets import concatenate_datasets
import nltk
import spacy
from scipy.spatial import distance

from .custom_settings import CUSTOM_TASK_NAME, CUSTOM_TEXT_FIELDS
from .constants import SPACY_POS_TAGS, CONTENT_POS_TAGS
from .dataset_utils import get_eval_datasets, get_dataset


# You may first need to run: python3 -m spacy download [spacy_model_name]
# STOP_WORDS use en_core_web_sm by default.
spacy_nlp = spacy.load("en_core_web_sm")
STOP_WORDS = spacy_nlp.Defaults.stop_words
del spacy_nlp


# Input: a Hugging Face dataset.
# Output: a dictionary mapping (token, POS) tuples to frequencies. Note that this
# uses the spaCy tokenizer rather than a Hugging Face tokenizer, so it has much
# fewer subword pieces (https://spacy.io/usage/linguistic-features#tokenization).
# Also outputs tensors of POS n-gram counts (unigram, bigram, trigram, ... 5-gram).
# Note: min_pos_seq_length n means a minimum sentence length of n-1 with a SEP token
# when counting POS n-grams.
def get_spacy_frequencies(dataset, task, spacy_model_name="en_core_web_sm", max_seq_length=512, min_pos_seq_length=3):
    if task in ["sentiment_amazon_categories", "sentiment_amazon_categories_large", "sentiment_amazon_years"]:
        text_field = "review_body"
        processed_dataset = dataset
    elif task == "mnli" or task == CUSTOM_TASK_NAME:
        print("Concatenating input text fields.")
        text_fields = ["premise", "hypothesis"] if task == "mnli" else CUSTOM_TEXT_FIELDS
        def concatenate_inputs(example):
            example["input_text"] = " ".join([example[text_field].strip() for text_field in text_fields])
            return example
        processed_dataset = dataset.map(concatenate_inputs)
        processed_dataset = processed_dataset.remove_columns([text_field for text_field in text_fields if text_field != "input_text"])
        text_field = "input_text"
    else:
        print("Unrecognized task: {}".format(task))
        return None, None
    # Load nltk sentence tokenizer to separate examples into sentences.
    nltk.download('punkt')
    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # Load spaCy. We only need the tagger and tok2vec for POS-tagging.
    spacy_nlp = spacy.load(spacy_model_name, disable=['parser', 'ner',
            'entity_linker', 'entity_ruler', 'lemmatizer', 'textcat', 'textcat_multilabel',
            'senter', 'sentencizer', 'transformer'])
    # Prepare POS n-gram matrices.
    n_pos = len(SPACY_POS_TAGS)
    pos_to_idx = dict(zip(SPACY_POS_TAGS, range(n_pos)))
    ngrams = [np.zeros(n_pos, dtype=np.int32),
              np.zeros((n_pos, n_pos), dtype=np.int32),
              np.zeros((n_pos, n_pos, n_pos), dtype=np.int32),
              np.zeros((n_pos, n_pos, n_pos, n_pos), dtype=np.int32),
              np.zeros((n_pos, n_pos, n_pos, n_pos, n_pos), dtype=np.int32)]

    # Get token frequencies and POS n-grams.
    token_counts = Counter()
    for example in tqdm(processed_dataset):
        text = example[text_field]
        if text is None:
            continue
        example_sentences = sentence_detector.tokenize(text.strip())
        spacy_sentences = [spacy_nlp(sentence) for sentence in example_sentences]
        # Note: not padding the start with a SEP token because the start will automatically
        # be padded with SEP tokens anyways for the n-grams.
        token_pos_list = [] # List of (token, POS) tuples in the example.
        for spacy_sentence in spacy_sentences:
            # Note: POS uses the Universal POS tags.
            token_pos_list.extend([(spacy_token.text, spacy_token.pos_) for spacy_token in spacy_sentence])
            token_pos_list.append(("", "SEP"))
        if len(token_pos_list) > max_seq_length:
            # Note: this is different from the Hugging Face tokenizer sequence length,
            # because it uses the spaCy tokenizer (https://spacy.io/usage/linguistic-features#tokenization),
            # which separates on whitespace, handles special cases, then separates punctuation affixes.
            # print("Truncating example from length {0} to {1}.".format(len(token_pos_list), max_seq_length))
            token_pos_list = token_pos_list[:max_seq_length]
        # For POS n-gram counts. Pad start with SEP tokens.
        sep_idx = pos_to_idx["SEP"]
        prev_pos = [sep_idx, sep_idx, sep_idx, sep_idx] # n previous, ..., two previous, one previous.
        # Iterate through example tokens.
        for curr_token, curr_pos_name in token_pos_list:
            # Update token counts.
            if curr_token != "":
                token_counts[(curr_token, curr_pos_name)] += 1
            # Update POS n-gram counts, if at least min_pos_seq_length.
            curr_pos = pos_to_idx[curr_pos_name]
            if len(token_pos_list) >= min_pos_seq_length:
                ngrams[0][curr_pos] += 1
                ngrams[1][prev_pos[-1], curr_pos] += 1
                ngrams[2][prev_pos[-2], prev_pos[-1], curr_pos] += 1
                ngrams[3][prev_pos[-3], prev_pos[-2], prev_pos[-1], curr_pos] += 1
                ngrams[4][prev_pos[-4], prev_pos[-3], prev_pos[-2], prev_pos[-1], curr_pos] += 1
            # Move forward one token position.
            prev_pos[0] = prev_pos[1]
            prev_pos[1] = prev_pos[2]
            prev_pos[2] = prev_pos[3]
            prev_pos[3] = curr_pos
        # Note: not padding the end with SEP tokens.
    total_token_count = sum(token_counts.values())
    token_frequencies = dict()
    for tuple_key, count in token_counts.items():
        token_frequencies[tuple_key] = float(count) / total_token_count
    del token_counts
    # Return frequencies and n-gram counts.
    return token_frequencies, tuple(ngrams)


# Get spaCy distributions for the domain.
# Loads from the pickled spaCy token frequencies and POS n-gram counts if found
# in the spacy_cache; otherwise, computes them from the input dataset and
# pickles them. The inputs dataset, task, spacy_model_name, min_pos_seq_length,
# and max_seq_length are only used if the pickled spaCy data is not found.
#
# The dataset_id is often [domain]_[train/eval].
# Outputs the content word frequency distribution, the overall spaCy token
# frequency distribution, and the token frequency distribution for adjectives,
# adverbs, nouns, proper nouns, and verbs.
# Also outputs the POS unigram, bigram, ..., 5-gram distributions.
# Output token frequencies are dictionaries from tokens to frequencies.
def get_spacy_distributions(dataset, task, spacy_cache, dataset_id,
                            spacy_model_name="en_core_web_sm", max_seq_length=512, min_pos_seq_length=3):
    # Get spaCy frequencies.
    os.makedirs(spacy_cache, exist_ok=True)
    token_frequencies_path = os.path.join(spacy_cache, "{}_frequencies.pickle".format(dataset_id))
    pos_ngrams_path = os.path.join(spacy_cache, "{}_ngrams.pickle".format(dataset_id))
    if os.path.isfile(token_frequencies_path):
        print("Loading spaCy frequencies for dataset: {}".format(dataset_id))
        token_frequencies = pickle.load(codecs.open(token_frequencies_path, "rb"))
        # Assume POS n-grams were saved at the same time.
        pos_ngrams = pickle.load(codecs.open(pos_ngrams_path, "rb"))
    else:
        print("Computing spaCy frequencies for dataset: {}".format(dataset_id))
        token_frequencies, pos_ngrams = get_spacy_frequencies(dataset, task,
            spacy_model_name="en_core_web_sm", max_seq_length=max_seq_length, min_pos_seq_length=min_pos_seq_length)
        with open(token_frequencies_path, 'wb') as handle:
            pickle.dump(token_frequencies, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(pos_ngrams_path, 'wb') as handle:
            pickle.dump(pos_ngrams, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pos_ngrams = list(pos_ngrams)

    # Get content word distributions (open classes in Universal Dependencies).
    # Content words also exclude stop words.
    content_frequencies = defaultdict(lambda: 0.0)
    adjective_frequencies = defaultdict(lambda: 0.0)
    adverb_frequencies = defaultdict(lambda: 0.0)
    noun_frequencies = defaultdict(lambda: 0.0)
    propnoun_frequencies = defaultdict(lambda: 0.0)
    verb_frequencies = defaultdict(lambda: 0.0)
    all_token_frequencies = defaultdict(lambda: 0.0)
    for tuple_key, frequency in token_frequencies.items():
        token, pos = tuple_key
        all_token_frequencies[token] += frequency
        if pos in CONTENT_POS_TAGS and token not in STOP_WORDS:
            content_frequencies[token] += frequency
        # Token distributions for individual POS tags.
        if pos == "ADJ":
            adjective_frequencies[token] += frequency
        elif pos == "ADV":
            adverb_frequencies[token] += frequency
        elif pos == "NOUN":
            noun_frequencies[token] += frequency
        elif pos == "PROPN":
            propnoun_frequencies[token] += frequency
        elif pos == "VERB":
            verb_frequencies[token] += frequency
    # Normalize frequency distributions over content and function words.
    # all_token_frequencies does not require normalization because those frequencies
    # should already sum to 1.0.
    def normalize_frequencies(frequency_dict):
        total = sum(frequency_dict.values())
        for token, orig_frequency in frequency_dict.items():
            frequency_dict[token] = orig_frequency / total
        return frequency_dict
    # Normalize.
    content_frequencies = normalize_frequencies(content_frequencies)
    adjective_frequencies = normalize_frequencies(adjective_frequencies)
    adverb_frequencies = normalize_frequencies(adverb_frequencies)
    noun_frequencies = normalize_frequencies(noun_frequencies)
    propnoun_frequencies = normalize_frequencies(propnoun_frequencies)
    verb_frequencies = normalize_frequencies(verb_frequencies)
    collected_frequencies = (content_frequencies, all_token_frequencies, adjective_frequencies,
                             adverb_frequencies, noun_frequencies, propnoun_frequencies,
                             verb_frequencies)
    # Process POS n-grams.
    for i in range(len(pos_ngrams)):
        pos_ngrams[i] = pos_ngrams[i] / np.sum(pos_ngrams[i])
    return collected_frequencies, tuple(pos_ngrams)


# Converts frequency distributions.
# Input: two frequency distributions as dictionaries from tokens to frequencies.
# Output: two frequency tensors with shape (union_vocab_size).
def frequency_dicts_to_arrays(freq_dict_a, freq_dict_b):
    union_tokens = set(list(freq_dict_a.keys()) + list(freq_dict_b.keys()))
    token_to_idx = dict(zip(union_tokens, range(len(union_tokens))))
    distr_a = np.zeros(len(union_tokens))
    for token, freq in freq_dict_a.items():
        distr_a[token_to_idx[token]] += freq
    distr_b = np.zeros(len(union_tokens))
    for token, freq in freq_dict_b.items():
        distr_b[token_to_idx[token]] += freq
    return distr_a, distr_b


# Computes drift metrics based on spaCy tokenization and tagging.
# Computes between all train/eval domains.
def compute_all_spacy_metrics(model_args, experiment_args, metric_args, train_domains, eval_domains, outpath):
    spacy_cache = os.path.join(experiment_args.experiment_output_dir, experiment_args.task + experiment_args.dir_suffix, "spacy_cache")
    print("Getting eval datasets.")
    eval_datasets = get_eval_datasets(
        experiment_args.dataset_dir, experiment_args.task, eval_domains)

    print("Computing distances.")
    outfile = codecs.open(outpath, 'w', encoding='utf-8')
    outfile.write("TrainDomain\tEvalDomain\t"
                  "pos_unigram_distance\tpos_bigram_distance\tpos_trigram_distance\t"
                  "pos_4gram_distance\tpos_5gram_distance\t"
                  "content_word_distance\tspacy_token_distance\t"
                  "adjective_distance\tadverb_distance\tnoun_distance\t"
                  "propnoun_distance\tverb_distance\n")
    for train_domain in train_domains:
        print("Running train domain: {}".format(train_domain))
        train_dataset, _, _ = get_dataset(experiment_args.dataset_dir, experiment_args.task, train_domain)
        # Load distributions for the first domain.
        collected_frequencies_a, ngrams_a = get_spacy_distributions(train_dataset,
                                                experiment_args.task, spacy_cache, train_domain+"_train",
                                                max_seq_length=model_args.max_seq_length,
                                                min_pos_seq_length=metric_args.min_pos_seq_length)
        for eval_domain in eval_domains:
            # Load distributions for the second domain.
            collected_frequencies_b, ngrams_b = get_spacy_distributions(eval_datasets[eval_domain],
                                                    experiment_args.task, spacy_cache, eval_domain+"_eval",
                                                    max_seq_length=model_args.max_seq_length,
                                                    min_pos_seq_length=metric_args.min_pos_seq_length)
            # POS n-gram distances.
            # Note: computes base e.
            ngram_dists = []
            for i in range(len(ngrams_a)):
                ngram_dist = distance.jensenshannon(ngrams_a[i].flatten(), ngrams_b[i].flatten())
                ngram_dists.append(str(ngram_dist))
            # Content word distance, all spaCy token distance (distance between
            # overall spaCy token distributions), adjective distance, noun distance,
            # verb distance, etc.
            frequency_dists = []
            for i in range(len(collected_frequencies_a)):
                distr_a, distr_b = frequency_dicts_to_arrays(collected_frequencies_a[i], collected_frequencies_b[i])
                frequency_dist = distance.jensenshannon(distr_a, distr_b)
                frequency_dists.append(str(frequency_dist))
            # Write output.
            outfile.write("{0}\t{1}\t{2}\t{3}\n".format(
                    train_domain, eval_domain,
                    "\t".join(ngram_dists),
                    "\t".join(frequency_dists)))
    outfile.close()
    print("Computed spaCy dataset-level metrics.")
    return True


# Class that annotates input text with the POS sequence cross-entropy and
# content word cross-entropy relative to some reference.
class StructVocExampleAnnotator:
    # Initialize with the POS n-gram conditional probabilities and the
    # dictionary of content word frequencies (mapping tokens to frequencies).
    # The n-gram probabilities should be an array of conditional probabilities
    # P(w_i | w_{i-1}, ..., w_{i-n+1}) where the last array index is the token
    # id for w_i, and the first n-1 array indices are the token ids for w_{i-n+1}
    # through w_{i-1} respectively.
    def __init__(self, ngram_probs, content_frequencies_dict, spacy_model_name="en_core_web_sm"):
        # Prepare for POS sequence probabilities.
        self.ngram_logprobs = np.log(ngram_probs)
        self.ngram_n = len(ngram_probs.shape)
        n_pos = len(SPACY_POS_TAGS)
        self.pos_to_idx = dict(zip(SPACY_POS_TAGS, range(n_pos)))
        # Prepare for content word probabilities.
        self.min_token_logprob = np.log(min(content_frequencies_dict.values()))
        self.content_logprobs_dict = dict()
        for token, freq in content_frequencies_dict.items():
            self.content_logprobs_dict[token] = np.log(freq)
        self.content_pos_indices = [self.pos_to_idx[pos] for pos in CONTENT_POS_TAGS]
        # Load the sentence tokenizer.
        nltk.download('punkt')
        self.sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        # Load spaCy. We only need the tagger and tok2vec for POS-tagging.
        self.spacy_nlp = spacy.load(spacy_model_name, disable=['parser', 'ner',
                'entity_linker', 'entity_ruler', 'lemmatizer', 'textcat', 'textcat_multilabel',
                'senter', 'sentencizer', 'transformer'])

    # Returns POS sequence XEnt and content word unigram XEnt.
    # min_pos_seq_length 3 means minimum sentence length 2 plus SEP token.
    def get_sequence_perplexity(self, example_text, max_seq_length=512, min_pos_seq_length=3):
        example_sentences = self.sentence_detector.tokenize(example_text.strip())
        spacy_sentences = [self.spacy_nlp(sentence) for sentence in example_sentences]
        # List of (token, POS_idx) tuples in the example.
        # Note: POS uses the Universal POS tags.
        sep_idx = self.pos_to_idx["SEP"]
        token_pos_list = [("", sep_idx) for _ in range(self.ngram_n - 1)] # Pad start with SEP.
        for spacy_sentence in spacy_sentences:
            token_pos_list.extend([(spacy_token.text, self.pos_to_idx[spacy_token.pos_]) for spacy_token in spacy_sentence])
            token_pos_list.append(("", sep_idx))
        if len(token_pos_list) > max_seq_length+self.ngram_n-1:
            # Suppress warning.
            # print("Truncating example from length {0} to {1}.".format(len(token_pos_list), max_seq_length))
            token_pos_list = token_pos_list[:max_seq_length+self.ngram_n-1]
        # Compute log probabilities.
        pos_logprobs = []
        content_logprobs = []
        for token_i in range(self.ngram_n-1, len(token_pos_list)):
            pos_sequence = tuple([token_pos_list[i][1] for i in range(token_i-self.ngram_n+1, token_i+1)])
            pos_logprobs.append(self.ngram_logprobs[pos_sequence])
            # Check if content word.
            token, pos_idx = token_pos_list[token_i]
            if pos_idx in self.content_pos_indices and token not in STOP_WORDS:
                if token in self.content_logprobs_dict:
                    token_logprob = self.content_logprobs_dict[token]
                else:
                    token_logprob = self.min_token_logprob
                content_logprobs.append(token_logprob)
        # Get final POS sequence xent and content word xent.
        # Adjust min_pos_seq_length due to padding at the start with SEP tokens.
        if len(token_pos_list) < min_pos_seq_length+self.ngram_n-1:
            pos_sequence_xent = np.nan
        else:
            pos_sequence_xent = -1.0 * np.mean(pos_logprobs)
        if len(content_logprobs) == 0: # No content words.
            content_word_xent = np.nan
        else:
            content_word_xent = -1.0 * np.mean(content_logprobs)
        return pos_sequence_xent, content_word_xent


# Compute structural and vocabulary cross-entropies for all evaluation examples,
# relative to each training domain. Outputs annotations in eval_structural_vocab_xent_annotations.npy
# in each train domain directory, with shape (n_eval_examples, 2). Columns correspond
# to structural and vocabulary cross-entropy.
def annotate_structural_vocab_xent(experiment_output_dir, dataset_dir, task, train_domains, eval_domains,
                                   dir_suffix="", max_seq_length=512, min_pos_seq_length=3, ngram_n=5):
    # Get entire eval pool.
    print("Loading all eval examples.")
    eval_pool = get_eval_datasets(dataset_dir, task, eval_domains).values()
    eval_pool = concatenate_datasets(eval_pool)
    print("{} eval examples.".format(len(eval_pool)))

    if task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
        text_field = "review_body"
    elif task == "mnli" or task == CUSTOM_TASK_NAME:
        print("Concatenating input text fields.")
        text_fields = ["premise", "hypothesis"] if task == "mnli" else CUSTOM_TEXT_FIELDS
        def concatenate_inputs(example):
            example["input_text"] = " ".join([example[text_field].strip() for text_field in text_fields])
            return example
        eval_pool = eval_pool.map(concatenate_inputs)
        eval_pool = eval_pool.remove_columns([text_field for text_field in text_fields if text_field != "input_text"])
        text_field = "input_text"
    else:
        print("Unrecognized task: {}".format(task))
        return False

    # Get content word and n-gram distributions from the training sets for each domain.
    spacy_cache = os.path.join(experiment_output_dir, task + dir_suffix, "spacy_cache")
    for train_domain in train_domains:
        print("Running for train domain: {}".format(train_domain))
        os.makedirs(os.path.join(experiment_output_dir, task+dir_suffix, train_domain), exist_ok=True)
        output_path = os.path.join(experiment_output_dir, task + dir_suffix, train_domain, "eval_structural_vocab_xent_annotations.npy")
        if os.path.isfile(output_path):
            print("Skipping: already found annotated predictions file.")
            continue
        # If they exist, load predictions now to check that there are the same number of examples as eval_pool.
        # Assume the same order of examples.
        # Skips if evaluate.py has not been run yet.
        predictions_path = os.path.join(experiment_output_dir, task + dir_suffix, train_domain, "domain_eval_predictions.npy")
        if os.path.isfile(predictions_path):
            print("Loading predictions to check array size.")
            predictions = np.load(predictions_path) # Shape: n_examples, n_classes.
            if predictions.shape[0] != len(eval_pool):
                print("ERROR for training domain {0}: {1} eval examples and {2} eval predictions found.".format(
                    train_domain, len(eval_pool), predictions.shape[0]))
                return
            del predictions

        print("Getting spaCy frequencies and n-grams from training dataset.")
        train_dataset, _, _ = get_dataset(dataset_dir, task, train_domain)
        # Note: ngram_probs are n-gram distributions instead of conditional n-gram distributions.
        collected_frequencies, ngram_probs = get_spacy_distributions(train_dataset, task,
                                            spacy_cache, train_domain+"_train", max_seq_length=max_seq_length,
                                            min_pos_seq_length=min_pos_seq_length)
        content_frequencies = collected_frequencies[0]
        del train_dataset
        del collected_frequencies
        # Get n-gram counts.
        ngrams_path = os.path.join(spacy_cache, "{}_ngrams.pickle".format(train_domain+"_train"))
        ngram_counts = pickle.load(codecs.open(ngrams_path, "rb"))
        ngram_counts = ngram_counts[ngram_n - 1] + 1 # Add one for smoothing.
        ngram_probs = ngram_counts / np.sum(ngram_counts, axis=-1, keepdims=True)
        del ngram_counts
        example_annotator = StructVocExampleAnnotator(ngram_probs, content_frequencies)

        print("Annotating eval pool with structural and vocabulary perplexities.")
        def add_sequence_perplexity(example):
            pos_xent, content_xent = example_annotator.get_sequence_perplexity(
                example[text_field], max_seq_length=max_seq_length,
                min_pos_seq_length=min_pos_seq_length)
            example["structural_xent"] = pos_xent
            example["vocab_xent"] = content_xent
            return example
        eval_pool = eval_pool.map(add_sequence_perplexity)

        structural_xent = np.array(eval_pool[:]["structural_xent"]).reshape(-1, 1) # Shape: n_examples, 1.
        vocab_xent = np.array(eval_pool[:]["vocab_xent"]).reshape(-1, 1) # Shape: n_examples, 1.
        output_array = np.concatenate([structural_xent, vocab_xent], axis=-1)
        np.save(output_path, output_array, allow_pickle=False)
        print("Saved annotated eval predictions.")

    print("Annotated example-level structural and vocabulary drift metrics.")
    return True
