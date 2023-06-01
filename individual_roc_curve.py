"""
Outputs the ROC curve and accuracy error for a given task, metric, train domain,
and eval domain.
Sample usage:

python3 individual_roc_curve.py \
--experiment_output_dir="experiment_output" \
--dataset_dir="datasets" \
--task="mnli" --dir_suffix=0 \
--outpath="figure.pdf" \
--train_domain="mnli_telephone" --eval_domain="mnli_fiction" \
--metric_string="structural_xent,vocab_xent,semantic_content"

"""

import argparse
import numpy as np
import os
import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from nlp_drift.dataset_utils import get_all_domains
from nlp_drift.drift_evaluation_utils import get_all_example_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_output_dir', required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--outpath', required=True)
    # Comma-separated in order if regression used multiple metrics, e.g.:
    # structural_xent,vocab_xent,semantic_content
    # Or:
    # finetuned_cosine
    parser.add_argument('--metric_string', required=True)
    # Can be an integer index or the domain name.
    parser.add_argument('--train_domain', required=True)
    parser.add_argument('--eval_domain', required=True)
    parser.add_argument('--dir_suffix', default="")
    return parser


def main(args):
    train_domains = get_all_domains(args.task)
    eval_domains = get_all_domains(args.task, eval=True)
    try:
        # If domains specified as integer indices.
        train_domain_i = int(args.train_domain)
        eval_domain_j = int(args.eval_domain)
    except:
        # Domains specified with domain names.
        train_domain_i = train_domains.index(args.train_domain)
        eval_domain_j = eval_domains.index(args.eval_domain)
    train_domain = train_domains[train_domain_i]
    eval_domain = eval_domains[eval_domain_j]
    datasets.utils.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    print("For train -> eval: {0} -> {1}".format(train_domain, eval_domain))

    # Just need train_domain, eval_domain, is_correct.
    example_data = get_all_example_data(args.experiment_output_dir, args.dataset_dir,
                        args.task, columns_to_keep=[], dir_suffix=args.dir_suffix, as_dict="train")
    example_data = example_data[train_domain]
    # Add predictions from the regression model.
    predictions_path = os.path.join(args.experiment_output_dir, args.task + args.dir_suffix,
            train_domain, "individual_performance_predictions", "{}.npy".format(args.metric_string.replace(",", "-")))
    predictions = np.load(predictions_path, allow_pickle=False)
    example_data = example_data.add_column(name="prediction", column=predictions)

    eval_domain_vector = np.array(example_data["eval_domain"], dtype=int)
    target_data = example_data.select(np.nonzero(eval_domain_vector == eval_domain_j)[0])
    in_domain_data = example_data.select(np.nonzero(eval_domain_vector == eval_domains.index(train_domain))[0])

    # Accuracy and predicted accuracy.
    print("In-domain accuracy: {}".format(np.mean(in_domain_data["is_correct"])))
    print("Target domain accuracy: {}".format(np.mean(target_data["is_correct"])))
    print("Predicted target accuracy: {}".format(np.nanmean(target_data["prediction"])))

    # ROC AUC.
    is_correct = np.array(target_data["is_correct"], dtype=bool)
    predictions = np.array(target_data["prediction"], dtype=float)
    predictions[np.isnan(predictions)] = np.nanmean(predictions)
    fpr, tpr, thresholds = roc_curve(is_correct, predictions)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: {}".format(roc_auc))
    # Plot ROC curve.
    plt.figure(figsize=(3,3))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(args.outpath, facecolor='white', bbox_inches='tight')
    plt.clf()
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
