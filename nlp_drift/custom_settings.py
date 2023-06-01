"""
Custom settings for user-specified datasets.
Dataset tsv files should be placed in a datasets directory, under the subdirectory
[CUSTOM_TASK_NAME]. Each domain should have two files:
[domain_name]_train.tsv
[domain_name]_eval.tsv

The tsv columns should include the text fields.
The label name is required only if running fine-tuning or other metrics that
require a supervised dataset.
Scripts will need to specify the datasets directory path.

Default values use the tiny subset of MNLI in the sample_datasets directory.

"""

# Eval domains must be a superset of the train domains.
CUSTOM_TASK_NAME = "mnli_tiny"
CUSTOM_TRAIN_DOMAINS = ["mnli_fiction", "mnli_government", "mnli_telephone"]
CUSTOM_EVAL_DOMAINS = ["mnli_fiction", "mnli_government", "mnli_telephone"]
# The text fields that are concatenated as inputs.
# E.g. ["premise", "hypothesis"] for MNLI.
CUSTOM_TEXT_FIELDS = ["premise", "hypothesis"]
# Label name is optional if only computing model-agnostic drift metrics,
# not fine-tuning models or running regressions.
CUSTOM_LABEL_NAME = "label"
CUSTOM_N_LABELS = 3
