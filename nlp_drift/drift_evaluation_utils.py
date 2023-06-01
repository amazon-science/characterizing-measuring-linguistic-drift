"""
Utilities for drift metric evaluations, including output data compilation,
regressions, RMSEs, and ROC AUCs.
"""

import os
import codecs
import pandas as pd
import numpy as np
import datasets
from tqdm import tqdm
from datasets import concatenate_datasets, Dataset
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc

from .custom_settings import CUSTOM_TASK_NAME, CUSTOM_N_LABELS
from .dataset_utils import get_all_domains, get_eval_datasets


# Combines the output data for each training domain into one file.
# The file [filename].tsv from each fine-tuned model directory is
# collected into all_[filename].tsv.
def combine_output_data(experiment_output_dir, task, filename, dir_suffix=""):
    train_domains = get_all_domains(task)
    dataframes = []
    for train_domain in train_domains:
        datafile = os.path.join(experiment_output_dir, task + dir_suffix, train_domain, filename)
        if os.path.isfile(datafile):
            df = pd.read_csv(datafile, sep="\t")
            dataframes.append(df)
    full_df = pd.concat(dataframes)
    output_file = os.path.join(experiment_output_dir, task + dir_suffix, "all_{}".format(filename))
    full_df.to_csv(output_file, sep="\t", index=False)
    return True


# Get the eval examples annotated with drift metrics for every train/eval domain pair.
# Each row will be an example relative to one train domain.
# as_dict can be "train" (map train domains to datasets), "train,eval" (map
# train,eval pairs to datasets), or "no" (concatenate all).
# Default columns: train_domain (as int), eval_domain (as int), is_correct (if available).
# Other possible columns:
# structural_xent, vocab_xent, frequency_js_div, frequency_xent, sequence_length,
# finetuned_cosine, pretrained_cosine, semantic_all, semantic_content (our lexical semantic drift metric),
# or any columns from the original dataset.
def get_all_example_data(experiment_output_dir, dataset_dir, task, columns_to_keep=[], dir_suffix="", as_dict="no"):
    columns_to_keep = columns_to_keep + ["train_domain", "eval_domain", "is_correct"]
    train_domains = get_all_domains(task, eval=False)
    eval_domains = get_all_domains(task, eval=True)
    # Get categorical annotation for eval domains.
    datasets.utils.logging.set_verbosity_error()
    eval_pool = get_eval_datasets(dataset_dir, task, eval_domains)
    eval_domain_annotation = []
    for eval_domain_i, eval_tuple in enumerate(eval_pool.items()):
        eval_domain_name, eval_dataset = eval_tuple
        eval_domain_data = np.ones(len(eval_dataset), dtype=int) * eval_domain_i
        eval_domain_annotation.append(eval_domain_data)
        # Also keep any columns that should be kept from the original dataset.
        eval_pool[eval_domain_name] = eval_dataset.remove_columns(
            [col for col in eval_dataset.column_names if col not in columns_to_keep])
    eval_domain_annotation = np.concatenate(eval_domain_annotation, axis=0)
    eval_pool = concatenate_datasets(eval_pool.values())
    if eval_pool.num_columns == 0:
        eval_pool = None
    print("{} total eval examples.".format(eval_domain_annotation.shape[0]))

    # Get annotations from each train domain.
    # Also add categorical annotation for train domain.
    full_dataset = dict()
    for train_domain_i, train_domain in enumerate(train_domains):
        # Dataset of eval examples relative to this train domain.
        dataset = dict()
        # Categorical annotation for train domain.
        train_domain_annotation = np.ones(eval_domain_annotation.shape[0], dtype=int) * train_domain_i
        dataset["train_domain"] = train_domain_annotation
        dataset["eval_domain"] = eval_domain_annotation

        # Prepare paths.
        predictions_path = os.path.join(experiment_output_dir, task + dir_suffix, train_domain, "domain_eval_predictions.npy")
        finetuned_cosine_annotations_path = os.path.join(experiment_output_dir, task + dir_suffix, train_domain, "eval_finetuned_cosine_annotations.npy")
        # For model-agnostic metrics, only search in the first fine-tuning output directory.
        first_suffix = "" if dir_suffix == "" else "0"
        pretrained_cosine_annotations_path = os.path.join(experiment_output_dir, task + first_suffix, train_domain, "eval_pretrained_cosine_annotations.npy")
        pretrained_semantic_annotations_path = os.path.join(experiment_output_dir, task + first_suffix, train_domain, "eval_pretrained_semantic_annotations.npy")
        xent_annotations_path = os.path.join(experiment_output_dir, task + first_suffix, train_domain, "eval_structural_vocab_xent_annotations.npy")
        frequency_annotations_path = os.path.join(experiment_output_dir, task + first_suffix, train_domain, "eval_frequency_annotations.npy")

        # Get predictions is_correct.
        if task in ["sentiment_amazon_categories", "sentiment_amazon_categories_small", "sentiment_amazon_years"]:
            n_classes = 2
        elif task == "mnli":
            n_classes = 3
        elif task == CUSTOM_TASK_NAME:
            n_classes = CUSTOM_N_LABELS
        else:
            print("Unrecognized task: {}".format(task))
        if os.path.isfile(predictions_path):
            predictions_and_labels = np.load(predictions_path, allow_pickle=False)
            assert predictions_and_labels.shape[0] == eval_domain_annotation.shape[0]
            predictions = np.argmax(predictions_and_labels[:, :n_classes], axis=-1)
            is_correct = np.equal(predictions, np.rint(predictions_and_labels[:, -1]))
            # Note: is_correct is casted to an int for statsmodels logistic regression fitting.
            dataset["is_correct"] = is_correct.astype(int)
            del predictions_and_labels # Note: labels is already in the original dataset.
            del predictions

        # Get structural and vocab XEnt annotations.
        if os.path.isfile(xent_annotations_path):
            xent_annotations = np.load(xent_annotations_path, allow_pickle=False)
            assert xent_annotations.shape[0] == eval_domain_annotation.shape[0]
            structural_xent = xent_annotations[:, 0]
            vocab_xent = xent_annotations[:, 1]
            dataset["structural_xent"] = structural_xent
            dataset["vocab_xent"] = vocab_xent
            del xent_annotations

        # Get frequency annotations.
        if os.path.isfile(frequency_annotations_path):
            frequency_annotations = np.load(frequency_annotations_path, allow_pickle=False)
            assert frequency_annotations.shape[0] == eval_domain_annotation.shape[0]
            frequency_js_div = np.square(frequency_annotations[:, 0])
            frequency_xent = frequency_annotations[:, 1]
            sequence_lengths = frequency_annotations[:, 2]
            dataset["frequency_js_div"] = frequency_js_div
            dataset["frequency_xent"] = frequency_xent
            dataset["sequence_length"] = sequence_lengths
            del frequency_annotations

        # Get cosine similarity annotations, converted to distances.
        if os.path.isfile(finetuned_cosine_annotations_path):
            finetuned_cosine_annotations = 1.0 - np.load(finetuned_cosine_annotations_path, allow_pickle=False)
            assert finetuned_cosine_annotations.shape[0] == eval_domain_annotation.shape[0]
            dataset["finetuned_cosine"] = finetuned_cosine_annotations
        if os.path.isfile(pretrained_cosine_annotations_path):
            pretrained_cosine_annotations = 1.0 - np.load(pretrained_cosine_annotations_path, allow_pickle=False)
            assert pretrained_cosine_annotations.shape[0] == eval_domain_annotation.shape[0]
            dataset["pretrained_cosine"] = pretrained_cosine_annotations
        if os.path.isfile(pretrained_semantic_annotations_path):
            pretrained_semantic_annotations = 1.0 - np.load(pretrained_semantic_annotations_path, allow_pickle=False)
            assert pretrained_semantic_annotations.shape[0] == eval_domain_annotation.shape[0]
            dataset["semantic_all"] = pretrained_semantic_annotations[:, 0]
            dataset["semantic_content"] = pretrained_semantic_annotations[:, 1]

        # Create dataset.
        dataset = Dataset.from_dict(dataset)
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in columns_to_keep])
        # Add back in specified columns from eval pool.
        if eval_pool is not None:
            for col in eval_pool.column_names:
                dataset = dataset.add_column(name=col, column=eval_pool[col])
        full_dataset[train_domain] = dataset
    if as_dict == "train":
        return full_dataset
    elif as_dict == "train,eval":
        # Assume same eval domain vector for all training domains.
        eval_domain_vector = np.array(full_dataset[train_domains[0]]["eval_domain"], dtype=int)
        for train_domain in train_domains:
            dataset = full_dataset.pop(train_domain, None)
            for eval_domain_i, eval_domain in enumerate(eval_domains):
                eval_filtered_dataset = dataset.select(np.nonzero(eval_domain_vector == eval_domain_i)[0])
                full_dataset[train_domain + "," + eval_domain] = eval_filtered_dataset
        return full_dataset
    else:
        full_dataset = concatenate_datasets(full_dataset.values())
        return full_dataset


# Runs the regression function for all training domains, evaluating on all
# evaluation domains.
#
# Runs a regression for each metric(s) string in the independent_vars_strings list.
# A metric(s) string should be a comma-separated list of drift metrics to use.
#
# The regression_fn should take a dataset to fit, an eval dataset, and a formula as
# input, outputting the eval prediction probabilities (e.g. see run_logistic_regression).
# The full_dataset_dict should map [train_domain,eval_domain] to the dataset of
# evaluation examples and drift metrics corresponding to that train/eval domain.
# The dataset should include the necessary columns to run the regression (i.e.
# independent and dependent variables).
# The prediction_type is used for output logging.
#
# Outputs predicted accuracies to:
# experiment_output_dir/task/eval_performance_predictions.tsv
# Outputs individual example probability predictions to a subdirectory in each
# train domain directory (when save_predictions is True):
# experiment_output_dir/task/train_domain/individual_performance_predictions
# This contains one .npy file for each regression, containing the predicted
# probabilities of getting each evaluation example correct (using the model trained
# on train_domain) based on the drift metric(s) for that regression.
def run_regressions(full_dataset_dict, regression_fn, experiment_output_dir, task,
                    dependent_var="is_correct", dir_suffix="", k_fold=5, outfile_name="temp.tsv",
                    independent_vars_strings=["frequency_xent"], prediction_type="logistic", save_predictions=False):
    output_dir = os.path.join(experiment_output_dir, task + dir_suffix)
    outpath = os.path.join(output_dir, outfile_name)
    indomain_outpath = os.path.join(output_dir, "indomain_" + outfile_name)
    if type(save_predictions) == bool:
        save_predictions = [save_predictions for _ in independent_vars_strings]
    train_domains = get_all_domains(task, eval=False)
    eval_domains = get_all_domains(task, eval=True)

    # Prepare output.
    if os.path.isfile(outpath):
        print("Appending results to existing output file.")
        outfile = codecs.open(outpath, 'a', encoding='utf-8')
    else:
        outfile = codecs.open(outpath, 'w', encoding='utf-8')
        outfile.write("TrainDomain\tEvalDomain\tPredicted\tPredictionType\tFitSet\tPredictors\n")
    if os.path.isfile(indomain_outpath):
        indomain_outfile = codecs.open(indomain_outpath, 'a', encoding='utf-8')
    else:
        indomain_outfile = codecs.open(indomain_outpath, 'w', encoding='utf-8')
        indomain_outfile.write("TrainDomain\tEvalDomain\tPredicted\tPredictionType\tFitSet\tPredictors\tFold\n")
    # Run regressions.
    # full_dataset_dict should be a map from [train_domain],[eval_domain] to the corresponding dataset.
    for train_domain in train_domains:
        # Prepare the directory for individual example predictions.
        predictions_dir = os.path.join(output_dir, train_domain, "individual_performance_predictions")
        if not os.path.isdir(predictions_dir):
            os.mkdir(predictions_dir)
        print("Running train domain: {}".format(train_domain))
        train_indomain_dataset = full_dataset_dict[train_domain + "," + train_domain]
        train_domain_j = eval_domains.index(train_domain)
        # Run regression on all in-domain examples, evaluate on other domains.
        # Run for all sets of independent variables.
        for vars_i, independent_vars_string in enumerate(independent_vars_strings):
            independent_vars = independent_vars_string.split(",")
            formula = "{0} ~ {1}".format(dependent_var, " + ".join(independent_vars))
            _, indomain_regression = regression_fn(train_indomain_dataset, None, formula=formula)
            all_predictions = []
            for eval_domain in eval_domains:
                if train_domain == eval_domain:
                    # In-domain predictions will be filled in later.
                    all_predictions.append(None)
                    continue
                eval_set = full_dataset_dict[train_domain + "," + eval_domain]
                predictions, _ = regression_fn(train_indomain_dataset, eval_set, formula=formula,
                                               existing_reg=indomain_regression, output_accuracy=False)
                all_predictions.append(predictions)
                predicted_performance = np.nanmean(predictions)
                outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                    train_domain, eval_domain, predicted_performance, prediction_type, "same_source_indomain",
                    ",".join(independent_vars)))
            # In-domain predictions using k-fold validation.
            indomain_predictions = -1.0 * np.ones(len(train_indomain_dataset))
            if "fold_i" not in train_indomain_dataset.column_names:
                # Create folds.
                fold_indices =  np.tile(np.arange(k_fold), (len(train_indomain_dataset) // k_fold) + 1)
                fold_indices = fold_indices[:len(train_indomain_dataset)]
                train_indomain_dataset = train_indomain_dataset.add_column(name="fold_i", column=fold_indices)
            for fold_i in range(k_fold):
                train_fold = train_indomain_dataset.filter(
                    lambda x: x["fold_i"] != fold_i, batch_size=10000)
                eval_fold = train_indomain_dataset.filter(
                    lambda x: x["fold_i"] == fold_i, batch_size=10000)
                predictions, _ = regression_fn(train_fold, eval_fold, formula=formula, existing_reg=None, output_accuracy=False)
                indomain_predictions[fold_indices == fold_i] = predictions
                predicted_indomain_performance = np.nanmean(predictions)
                indomain_outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n".format(
                    train_domain, train_domain, predicted_indomain_performance, prediction_type, "same_source_indomain",
                    ",".join(independent_vars), fold_i))
            all_predictions[train_domain_j] = indomain_predictions # Fill in in-domain predictions.
            all_predictions = np.concatenate(all_predictions, axis=0)
            # Write performance predictions for individual examples.
            if save_predictions[vars_i]:
                predictions_filename = "-".join(independent_vars) + ".npy"
                predictions_outpath = os.path.join(predictions_dir, predictions_filename)
                np.save(predictions_outpath, all_predictions, allow_pickle=False)
    outfile.close()
    indomain_outfile.close()
    return True


# Return ground truth accuracy (is_correct for each example) for eval.
# Only eval_set, formula, and output_accuracy is used.
def run_ground_truth_regression(train_set, eval_set, formula, existing_reg=None, output_accuracy=False):
    if eval_set is None:
        return None, None
    dependent_var = formula.split("~")[0].strip()
    is_correct = np.array(eval_set[dependent_var])
    if output_accuracy:
        return np.nanmean(is_correct), None
    else:
        return is_correct, None


# Regression with only intercept (mean).
def run_constant_regression(train_set, eval_set, formula, existing_reg=None, output_accuracy=False):
    train_mean = existing_reg
    if train_mean is None:
        dependent_var = formula.split("~")[0].strip()
        train_mean = np.nanmean(train_set[dependent_var])
    if eval_set is None:
        return None, train_mean
    # Constant predictions.
    predictions = np.ones(len(eval_set)) * train_mean
    if output_accuracy:
        return np.nanmean(predictions), train_mean
    else:
        return predictions, train_mean


# Run a single logistic regression.
# Does not run any outlier filtering of the train set (empirically, no filtering
# seems to better predict out-of-domain model performance).
# The train_set is used to fit the regression if existing_reg is not provided.
def run_logistic_regression(train_set, eval_set, formula, existing_reg=None, output_accuracy=False):
    logistic_reg = existing_reg
    if logistic_reg is None:
        print("Running logistic regression with {} examples.".format(len(train_set)))
        # No regularization.
        logistic_reg = smf.logit(formula=formula, data=train_set).fit(disp=0)
        # print(logistic_reg.params)
    if eval_set is None:
        return None, logistic_reg

    # Predict eval set.
    # Outputs probabilities.
    eval_is_nan = np.zeros(len(eval_set), dtype=bool)
    independent_vars = formula.split("~")[1].strip().split(" + ")
    for var in independent_vars:
        if var in eval_set.column_names:
            eval_is_nan[np.isnan(np.array(eval_set[var]))] = True
    predictions = np.zeros(len(eval_set))
    # print("NaN eval examples: {}".format(np.sum(eval_is_nan)))
    predictions[eval_is_nan] = np.nan
    predictions[np.logical_not(eval_is_nan)] = logistic_reg.predict(eval_set) # Shape: n_examples with no nans.

    if output_accuracy:
        # Mean predicted accuracy over all examples in the eval set.
        return np.nanmean(predictions), logistic_reg
    else:
        return predictions, logistic_reg


# Run all logistic regressions.
# See run_regressions() for more details.
def run_logistic_regressions(independent_vars_strings, experiment_output_dir, dataset_dir, task,
                             dir_suffix="", k_fold=5, outfile_name="temp.tsv", save_predictions=False):
    # Preparation.
    columns_to_keep = set()
    for independent_vars_string in independent_vars_strings:
        independent_vars = independent_vars_string.split(",")
        columns_to_keep.update(independent_vars)
    example_data = get_all_example_data(experiment_output_dir, dataset_dir, task,
                        columns_to_keep=list(columns_to_keep), dir_suffix=dir_suffix, as_dict="train,eval")
    # Run logistic regressions.
    run_regressions(example_data, run_logistic_regression, experiment_output_dir, task,
                    dependent_var="is_correct", dir_suffix=dir_suffix, k_fold=k_fold, outfile_name=outfile_name,
                    independent_vars_strings=independent_vars_strings, prediction_type="logistic", save_predictions=save_predictions)
    return True


# Predictions using constant regressions (intercept only, no independent variables).
# This is equivalent to predicting no out-of-domain performance drop.
# See run_regressions() for more details.
def run_constant_regressions(experiment_output_dir, dataset_dir, task,
                             dir_suffix="", k_fold=5, outfile_name="temp.tsv"):
    example_data = get_all_example_data(experiment_output_dir, dataset_dir, task,
                            columns_to_keep=[], dir_suffix=dir_suffix, as_dict="train,eval")
    run_regressions(example_data, run_constant_regression, experiment_output_dir, task,
                    dependent_var="is_correct", dir_suffix=dir_suffix, k_fold=k_fold, outfile_name=outfile_name,
                    independent_vars_strings=[""], prediction_type="constant", save_predictions=False)
    return True


# Ground truth accuracies.
# See run_regressions() for more details, except this will output ground truth
# for each eval set (i.e. not an actual regression).
def run_ground_truth_regressions(experiment_output_dir, dataset_dir, task,
                                 dir_suffix="", k_fold=5, outfile_name="temp.tsv"):
    example_data = get_all_example_data(experiment_output_dir, dataset_dir, task,
                            columns_to_keep=[], dir_suffix=dir_suffix, as_dict="train,eval")
    run_regressions(example_data, run_ground_truth_regression, experiment_output_dir, task,
                    dependent_var="is_correct", dir_suffix=dir_suffix, k_fold=k_fold, outfile_name=outfile_name,
                    independent_vars_strings=[""], prediction_type="ground_truth", save_predictions=False)
    return True


# Collects a (n_train_domain, n_eval_domain) matrix of the predicted performance between domains,
# assuming the regressions have already been run.
# The dir_suffix is appended to the task name when searching for the path.
def get_single_performance_matrix(experiment_output_dir, task, train_domains, eval_domains, drift_metric, dir_suffix="", indomain=False):
    # Choose the metric to read and the metric file.
    train_column_name = "TrainDomain"
    eval_column_name = "EvalDomain"
    prediction_column_name = "Predicted"
    if indomain:
        metric_file = "indomain_eval_performance_predictions.tsv"
    else:
        metric_file = "eval_performance_predictions.tsv"

    # Read metric values.
    datafile = os.path.join(experiment_output_dir, task + dir_suffix, metric_file)
    metric_df = pd.read_csv(datafile, sep="\t")
    # All regressions fitted to a single training and eval domain.
    metric_df = metric_df[metric_df["FitSet"] == "same_source_indomain"]

    # Filter to target drift metric(s).
    if drift_metric == "constant":
        metric_df = metric_df[metric_df["PredictionType"] == "constant"]
    elif drift_metric == "ground_truth":
        metric_df = metric_df[metric_df["PredictionType"] == "ground_truth"]
    else:
        metric_df = metric_df[metric_df["PredictionType"] == "logistic"]
        metric_df = metric_df[metric_df["Predictors"] == drift_metric]

    # Special case: in-domain evaluation.
    # This will have a different shape from usual: n_train_domains, n_folds.
    if indomain:
        indomain_results = []
        for domain_i, domain_name in enumerate(train_domains):
            # Will have one result per fold.
            filtered_df = metric_df[metric_df[train_column_name] == domain_name][metric_df[eval_column_name] == domain_name]
            indomain_results.append(np.array(filtered_df[metric_to_read]))
        indomain_results = np.stack(indomain_results, axis=0)
        return indomain_results

    # Fill in predicted performance matrix.
    performance_matrix = np.zeros((len(train_domains), len(eval_domains)))
    performance_matrix[:,:] = np.nan
    for domain_i, domain_name_i in enumerate(train_domains):
        for domain_j, domain_name_j in enumerate(eval_domains):
            filtered_df = metric_df[metric_df[train_column_name] == domain_name_i][metric_df[eval_column_name] == domain_name_j]
            if len(filtered_df) == 0:
                continue
            if len(filtered_df) > 1:
                print("WARNING: multiple metric values for train/eval pair: {0}, {1}".format(domain_name_i, domain_name_j))
            performance_values = filtered_df[prediction_column_name]
            performance_value = np.mean(performance_values)
            performance_matrix[domain_i, domain_j] = performance_value
    return performance_matrix


# If multiple fine-tuning runs, collects a matrix of shape (n_train_domain, n_eval_domain, n_runs).
# Otherwise, outputs the same as get_single_performance_matrix().
def get_performance_matrix(experiment_output_dir, task, train_domains, eval_domains,
                           drift_metric, indomain=False, n_runs=-1):
    # Set dir_suffix.
    use_dir_suffix = True
    if n_runs == -1:
        use_dir_suffix = False
        n_runs = 1
    # Get data for all runs.
    performance_matrices = []
    for run_i in range(n_runs):
        dir_suffix = str(run_i) if use_dir_suffix else ""
        performance_matrix = get_single_performance_matrix(experiment_output_dir,
                    task, train_domains, eval_domains, drift_metric, dir_suffix=dir_suffix, indomain=indomain)
        performance_matrices.append(performance_matrix)
    performance_matrices = np.stack(performance_matrices, axis=-1)
    return performance_matrices


# Computes the RMSE for each drift metric and task.
# Each metric_string should be a comma-separated list of drift metrics used in the regression.
# Returns a pandas data frame with columns: MetricString, RMSE.
def get_rmses(experiment_output_dir, task, train_domains, eval_domains, metric_strings, n_runs=-1):
    rmses = pd.DataFrame(columns=["MetricString", "RMSE"])
    for metric_string in metric_strings:
        # Get x and y metrics.
        matrix_x = get_performance_matrix(experiment_output_dir, task, train_domains, eval_domains, metric_string, n_runs=n_runs)
        matrix_y = get_performance_matrix(experiment_output_dir, task, train_domains, eval_domains, "ground_truth", n_runs=n_runs)
        # Predicted performance only for out-of-domain.
        for train_domain_i, train_domain in enumerate(train_domains):
            train_domain_j = eval_domains.index(train_domain)
            matrix_x[train_domain_i, train_domain_j, :] = np.nan
            matrix_y[train_domain_i, train_domain_j, :] = np.nan
            # If train domain is based on year, then exclude past years.
            if train_domain[:4] == "year":
                train_year = int(train_domain[4:])
                for eval_j, eval_domain in enumerate(eval_domains):
                    if eval_domain[:4] != "year":
                        continue
                    eval_year = int(eval_domain[4:])
                    if train_year > eval_year:
                        matrix_x[train_domain_i, eval_j, :] = np.nan
                        matrix_y[train_domain_i, eval_j, :] = np.nan
        # Raw error.
        # print(matrix_x[:,:,0] - matrix_y[:,:,0])

        # As flattened vectors.
        vector_x = matrix_x.flatten()
        vector_y = matrix_y.flatten()
        nan_filter = np.logical_not(np.logical_or(np.isnan(vector_x), np.isnan(vector_y)))
        vector_x = vector_x[nan_filter]
        vector_y = vector_y[nan_filter]
        rmse = np.sqrt(np.mean(np.square(vector_x - vector_y)))

        # Add to data frame.
        row = pd.DataFrame({"MetricString": [metric_string], "RMSE": [rmse]})
        rmses = pd.concat([rmses, row], ignore_index=True)
        print("RMSE {0}:\t{1}".format(metric_string, rmse))

        if metric_string == "constant":
            # Print raw performance change mean and stdev.
            print("Performance change mean: {}".format(np.mean(vector_y - vector_x)))
            print("Performance change stdev: {}".format(np.std(vector_y - vector_x)))
    return rmses


# Data should include is_correct (ground truth whether the model predicted correctly)
# and prediction (predicted probability).
# Returns the ROC AUC.
def get_roc_auc(data):
    is_correct = np.array(data["is_correct"], dtype=bool)
    predictions = np.array(data["prediction"], dtype=float)
    predictions[np.isnan(predictions)] = np.nanmean(predictions)
    # ROC AUC.
    fpr, tpr, thresholds = roc_curve(is_correct, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc


# Computes the mean ROC AUC for each drift metric and task.
# Each metric_string should be a comma-separated list of drift metrics used in the regression.
# Returns a pandas data frame with columns: MetricString, InOutDomain, ROC_AUC.
def get_roc_aucs(experiment_output_dir, dataset_dir, task, train_domains, eval_domains, metric_strings, n_runs=-1):
    # Setup.
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    # Set dir_suffix.
    use_dir_suffix = True
    if n_runs == -1:
        use_dir_suffix = False
        n_runs = 1

    roc_aucs = pd.DataFrame(columns=["MetricString", "InOutDomain", "ROC_AUC"])
    for metric_string in metric_strings:
        independent_vars = metric_string.split(",")
        # Get data for all fine-tuning runs.
        all_runs_data = []
        for run_i in range(n_runs):
            dir_suffix = str(run_i) if use_dir_suffix else ""
            # Just need train_domain, eval_domain, is_correct.
            # Assume will be in standard order (sorted by train_domain, then eval_domain).
            example_data = get_all_example_data(experiment_output_dir, dataset_dir,
                                task, columns_to_keep=[], dir_suffix=dir_suffix, as_dict="train")
            # Add the regression predictions.
            for train_domain in train_domains:
                predictions_path = os.path.join(experiment_output_dir, task + dir_suffix,
                        train_domain, "individual_performance_predictions", "-".join(independent_vars) + ".npy")
                predictions = np.load(predictions_path, allow_pickle=False)
                example_data[train_domain] = example_data[train_domain].add_column(name="prediction", column=predictions)
                run_i_column = np.ones(len(predictions), dtype=int) * run_i
                example_data[train_domain] = example_data[train_domain].add_column(name="run", column=run_i_column)
            all_runs_data.append(example_data)

        # Compute ROC AUCs for individual models, in-domain and out-of-domain.
        indomain_roc_aucs = []
        outdomain_roc_aucs = []
        # Same eval domain vector for all fine-tuning runs, all training domains, so just take the first.
        eval_domain_vector = np.array(all_runs_data[0][train_domains[0]]["eval_domain"], dtype=int)
        for run_data in all_runs_data: # For each fine-tuning run.
            for train_domain in tqdm(train_domains): # For each training domain.
                # In-domain.
                train_j = eval_domains.index(train_domain)
                in_domain_data = run_data[train_domain].select(np.nonzero(eval_domain_vector == train_j)[0])
                roc_auc = get_roc_auc(in_domain_data)
                indomain_roc_aucs.append(roc_auc)
                # Out-of-domain.
                outdomain_data = run_data[train_domain].select(np.nonzero(eval_domain_vector != train_j)[0])
                # If train domain is based on year, then exclude past years.
                if train_domain[:4] == "year":
                    year_filter = np.zeros_like(eval_domain_vector, dtype=bool)
                    train_year = int(train_domain[4:])
                    for eval_j, eval_domain in enumerate(eval_domains):
                        if eval_domain[:4] != "year":
                            continue
                        eval_year = int(eval_domain[4:])
                        if train_year < eval_year:
                            eval_mask = eval_domain_vector == eval_j
                            year_filter[eval_mask] = True
                    outdomain_data = run_data[train_domain].select(np.nonzero(year_filter)[0])
                    if len(outdomain_data) == 0:
                        print("No out-of-domain examples for domain: {}".format(train_domain))
                        continue
                # Out-of-domain ROC AUC.
                roc_auc = get_roc_auc(outdomain_data)
                outdomain_roc_aucs.append(roc_auc)

        # Add to data frame.
        mean_indomain = np.mean(indomain_roc_aucs)
        mean_outdomain = np.mean(outdomain_roc_aucs)
        rows = pd.DataFrame({"MetricString": [metric_string]*2, "InOutDomain": ["indomain", "outdomain"], "ROC_AUC": [mean_indomain, mean_outdomain]})
        roc_aucs = pd.concat([roc_aucs, rows], ignore_index=True)
        print("In-domain {0} mean ROC AUC: {1}".format(metric_string, mean_indomain))
        print("Out-domain {0} mean ROC AUC: {1}".format(metric_string, mean_outdomain))
    return roc_aucs
