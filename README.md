# Characterizing and Measuring Linguistic Dataset Drift

Experiments assessing different drift metrics in their ability to predict NLP model performance on novel data.
Drift metrics can be computed for any text datasets, and the full experiments (e.g. model fine-tuning) can be run for any sequence classification task.
Steps 2a, 2b, and 2c below can be run largely independently from one another, unless otherwise specified (e.g. if only the model-agnostic metrics are desired, then 2b can be omitted).
The specific drift outputs from step 2 do not need to be understood in detail, because they will be compiled automatically in step 3a.
More detailed usage for each script can be found in the Python script file headers.

---

## 0. Setup.

Tested on Python 3.7.12, Pytorch 1.9.1, and Transformers 4.20.1.
To create the conda environment:
```bash
conda create --name drift_experiments python=3.7
conda activate drift_experiments
conda config --remove channels conda-forge
conda install pytorch=1.9.1 -c pytorch
conda config --add channels conda-forge
conda install --file requirements.txt
```
Note that removing conda-forge is initially necessary to ensure that pytorch installs from the pytorch channel, for GPU support.
Then, download the spaCy model:
```bash
python3 -m spacy download en_core_web_sm
```

---

## 1. Collect or upload datasets.

We include scripts to pull the Amazon Reviews datasets by product category, the Amazon Reviews datasets by review year, and the MNLI dataset by source domain.
These scripts can be slow due to the size of the Amazon Reviews dataset.

These datasets are placed in the "datasets" directory, and our code will automatically use the correct settings for these datasets if the task flag is set to "sentiment_amazon_categories", "sentiment_amazon_years", or "mnli" respectively, and the dataset_dir flag points to the "datasets" directory.

To use your own datasets, you can set the variables in `nlp_drift/custom_settings.py` to set your training and evaluation domains, the custom task name, and your dataset settings.
By default, these custom settings use the "mnli_tiny" task in the `sample_datasets` directory.

To pull Amazon Reviews data by product category:
```bash
python3 get_amazon_reviews_datasets.py --output_dir="datasets" \
--max_per_category=100000 --max_per_category_year=5000
```
To compile Amazon Reviews data by year as well:
```bash
python3 compile_amazon_reviews_years.py --dataset_dir="datasets" \
--n_eval=5000 --n_test=5000
```
To pull the MNLI data by source domain:
```bash
python3 get_mnli_datasets.py --output_dir="datasets" \
--max_per_domain=-1
```

---

## 2a. Run frequency divergence metrics and spaCy metrics.

Compute token frequency divergence metrics at both the example and dataset level.
Dataset-level drift values are outputted to `experiment_output_dir/task/frequency_metrics.tsv`.
Example-level drift values are outputted into each train domain output directory `experiment_output_dir/task/train_domain/eval_frequency_annotations.npy`, with shape (n_eval_examples, 3), concatenating all evaluation domains.
Columns correspond to JS-distance, cross-entropy, and raw example sequence length.
These metrics use the specified Hugging Face model tokenizer.
```bash
python3 compute_drift_metrics.py \
--tokenizer_name="roberta-base" --metrics="frequency" \
--experiment_output_dir="experiment_output" --task="mnli_tiny" \
--dataset_dir="sample_datasets" \
--max_seq_length=512
```

Then, compute spaCy drift metrics (e.g. POS n-gram divergences, content word frequency divergences, etc) at the dataset level.
Dataset-level drift values are outputted to `experiment_output_dir/task/spacy_metrics.tsv`.
```bash
python3 compute_drift_metrics.py \
--metrics="spacy" \
--experiment_output_dir="experiment_output" --task="mnli_tiny" \
--dataset_dir="sample_datasets" --max_seq_length=512
```
For debugging, the content word and POS n-gram frequencies can be viewed per evaluation domain, outputted to `experiment_output_dir/task/eval_domain_data` by running the script below.
```bash
python3 compile_frequencies_ngrams.py \
--task="mnli_tiny" \
--experiment_output_dir="experiment_output"
```

We can also compute spaCy metrics at the example level (i.e. example-level vocabulary and structural drift).
Example-level drift values are outputted into each train domain output directory `experiment_output_dir/task/train_domain/eval_structural_vocab_xent_annotations.npy`, with shape (n_eval_examples, 2).
Columns correspond to structural and vocabulary cross-entropy.
```bash
python3 compute_structural_vocab_drift.py \
--experiment_output_dir="experiment_output" --task="mnli_tiny" \
--dataset_dir="sample_datasets" --max_seq_length=512
```

---

## 2b. Fine-tune models.

Fine-tune a language model on each training domain for a sequence classification task.
Outputs each model to `experiment_output_dir/task/train_domain`.
```bash
python3 finetune.py \
--model_name_or_path="roberta-base" --max_seq_length=512 \
--output_dir="placeholder" --save_strategy="no" \
--do_train --do_eval --evaluation_strategy="steps" --eval_steps=1000 \
--per_device_train_batch_size=8 --gradient_accumulation_steps=4 \
--lr_scheduler_type="linear" --learning_rate=0.00002 --warmup_ratio=0.10 \
--num_train_epochs=4 --per_device_eval_batch_size=8 \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--experiment_output_dir="experiment_output" --seed=42
```

Evaluate the fine-tuned models on all evaluation domains.
Outputs the tsv of cross-domain accuracies to `experiment_output_dir/task/all_domain_eval_results.tsv`.
Outputs the raw predictions for each model to `experiment_output_dir/task/train_domain/domain_eval_predictions.npy`, with shape: (n_eval_examples, n_classes+1).
The first columns correspond to class logits, and the last column is the true label.
```bash
python3 evaluate.py \
--max_seq_length=512 \
--per_device_eval_batch_size=16 --output_dir="placeholder" \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--experiment_output_dir="experiment_output"
```

---

## 2c. Compute embedding cosine metrics.

Compute pre-trained embedding cosine similarities.
Outputs the tsv of mean cosine similarities between training/eval domains to `experiment_output_dir/task/all_pretrained_cosine_metrics.tsv`.
Includes dataset-level lexical semantic similarities as well (eval token frequency-weighted lexical semantic similarity).
Outputs the example-level cosine similarities into each train domain output directory `experiment_output_dir/task/train_domain/eval_pretrained_cosine_annotations.npy`, with shape: (n_eval_examples).
The example-level lexical semantic similarities are outputted into `experiment_output_dir/task/train_domain/eval_pretrained_semantic_annotations.npy`, with shape: (n_eval_examples, 2).
Columns correspond to lexical semantic similarities when unfiltered or filtered to content tokens.
```bash
python3 compute_cosine_metrics.py \
--max_seq_length 512 \
--per_device_eval_batch_size 16 --output_dir="placeholder" \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--experiment_output_dir="experiment_output" \
--pos_dict_path="English_pos_dict.tsv" \
--model_name_or_path="roberta-base"
```

Then, if step 2b has been run, we can compute fine-tuned embedding cosine similarities.
Similar to above, outputs the tsv of mean cosine similarities between training/eval domains to `experiment_output_dir/task/all_finetuned_cosine_metrics.tsv`.
Outputs the example-level cosine similarities into each train domain output directory `experiment_output_dir/task/train_domain/eval_finetuned_cosine_annotations.npy`, with shape: (n_eval_examples).
```bash
python3 compute_cosine_metrics.py \
--max_seq_length 512 \
--per_device_eval_batch_size 16 --output_dir="placeholder" \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--experiment_output_dir="experiment_output"
```

---

## 3a. Compile drift metrics (optional).

If some subset of the commands in 2a, 2b, and 2c has been run, then the example-level drift metrics can be compiled into one file `experiment_output_dir/task/eval_example_drift_metrics.tsv`.
This step is optional, but it can be useful to see the drift metrics all in one place.
Each row corresponds to a single evaluation example relative to one training domain.
Depending on the steps run above, columns can include the training and evaluation domains, the various drift metrics, whether the trained model correctly predicted the example, the task-specific text fields, and the ground truth label.
Note that the dataset-level drift metrics are already organized in the output tsv files from the steps above.
Then, all drift metrics can be found in `experiment_output_dir/task`.
```bash
python3 compile_drift_metrics.py \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--experiment_output_dir="experiment_output"
```

---

## 3b. Run logistic regressions.

Run a logistic regression for each model, and each drift metric(s), predicting whether the model will get each example correct.
Outputs predicted accuracies to `experiment_output_dir/task/eval_performance_predictions.tsv`.
Outputs individual example probability predictions to a subdirectory in each train domain directory `experiment_output_dir/task/train_domain/individual_performance_predictions`.
This contains one npy file for each regression, containing the predicted probabilities of getting each evaluation example correct (using the model trained on train_domain) based on the drift metric(s) for that regression.
```bash
python3 run_regressions.py \
--experiment_output_dir="experiment_output" \
--task="mnli_tiny" --dataset_dir="sample_datasets" \
--outfile_name="eval_performance_predictions.tsv"
```

---

## 3c. Evaluate performance predictions.

Compute the RMSE for each drift metric(s).
Outputs to: `experiment_output_dir/task/rmses.tsv`.
```bash
python3 get_rmse_scores.py \
--experiment_output_dir="experiment_output" \
--task="mnli_tiny"
```

Compute the mean ROC AUC for each drift metric(s).
Outputs to: `experiment_output_dir/task/roc_aucs.tsv`.
```bash
python3 get_roc_scores.py \
--experiment_output_dir="experiment_output" \
--dataset_dir="sample_datasets" \
--task="mnli_tiny"
```

---

## Citation.

If you use these sources, please consider citing the following work:
```
@inproceedings{chang-etal-2023-characterizing,
    title = "Characterizing and Measuring Linguistic Dataset Drift",
    author = "Chang, Tyler and
    Halder, Kishaloy and
    Anna John, Neha and
    Vyas, Yogarshi and
    Benajiba, Yassine and
    Ballesteros, Miguel and
    Roth, Dan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2023",
    publisher = "Association for Computational Linguistics"
}
```
