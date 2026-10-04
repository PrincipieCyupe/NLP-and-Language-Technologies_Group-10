# Gender-Based Violence Tweet Classification

## Overview

This project investigates the classification of gender-based violence (GBV) tweets using machine learning and deep learning approaches.

The task is to classify tweets into five categories:

- Sexual violence
- Physical violence
- Emotional violence
- Economic violence
- Harmful traditional practices

The study compares traditional text classification with sequential and pretrained language models to investigate whether understanding word order and context provides a meaningful improvement over keyword-based classification.

## Problem Investigated

The dataset is highly imbalanced, with sexual violence representing most of the observations while economic violence and harmful traditional practices contain very few examples.

Exploratory analysis also showed that several classes contain very strong keywords. This creates an important question for the study:

> How effectively can sequential models classify GBV tweets into five categories, and does modelling word order add meaningful value over a keyword-based baseline?

The project therefore compares different modelling approaches using the same preprocessing and validation strategy.

## Dataset

The training dataset contains **39,650 labelled tweets**.

The main columns are:

```text
Tweet_ID
tweet
type
```

The target variable contains five GBV categories.

The dataset is strongly imbalanced, with sexual violence representing more than 80% of the training samples.

Because of this imbalance, **Macro-F1** is used as the main evaluation metric.

## Data Preprocessing

Preprocessing is intentionally kept light so that useful social media information and word order are preserved.

The notebook performs:

- HTML entity decoding
- Unicode normalization
- URL replacement
- Username replacement
- Whitespace normalization
- Missing value checks
- Duplicate checks
- Duplicate-aware train and validation splitting

Punctuation, hashtags, emojis and word order are retained.

Stop-word removal, stemming and lemmatization are not applied globally.

A duplicate-aware stratified split is used to reduce the risk of identical normalized tweets appearing in both the training and validation sets.

## Models

The following approaches are investigated:

### 1. TF-IDF + Logistic Regression

Used as the traditional machine learning baseline.

Experiments include:

- Unigram TF-IDF
- Balanced class weights
- Unigrams and bigrams

### 2. Bidirectional LSTM

A sequence model that processes tweets in both directions to capture context.

Experiments compare:

- Baseline BiLSTM
- BiLSTM with class weights
- Unidirectional LSTM

### 3. 1D CNN

A convolutional model designed to detect informative short phrases.

The notebook compares different convolution window sizes.

### 4. CNN-BiLSTM

A hybrid model combining:

- Word embeddings
- Convolution
- Bidirectional LSTM
- Attention or max pooling
- Softmax classification

### 5. DistilBERT

A pretrained Transformer model fine-tuned for the five-class classification problem.

The notebook compares full fine-tuning with an experiment where the pretrained encoder is frozen.

## Technologies Used

The project uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- PyTorch
- Hugging Face Transformers
- SciPy
- Jupyter Notebook
- Google Colab

## Evaluation

The models are evaluated using:

- Macro-F1
- Accuracy
- Per-class precision, recall and F1
- Confusion matrices
- ROC-AUC
- PR-AUC
- Bootstrap confidence intervals
- McNemar tests
- Error analysis

Macro-F1 is the main metric because each class contributes equally regardless of its size.

## Main Results

The best models achieved approximately the following validation Macro-F1 scores:

| Model | Macro-F1 |
|---|---:|
| TF-IDF + Logistic Regression | 0.977 |
| BiLSTM | 0.989 |
| 1D CNN | 0.992 |
| CNN-BiLSTM | 0.993 |
| DistilBERT | 0.996 |

The neural models perform slightly better than the TF-IDF baseline.

However, the differences between the strongest models are very small.

The analysis also shows that the dataset contains strong class-specific keywords, meaning that high validation scores do not necessarily indicate that the models fully understand the meaning of the tweets.

## Key Findings

- TF-IDF + Logistic Regression performs strongly despite using a relatively simple representation.
- Class balancing improves the traditional baseline on minority classes.
- BiLSTM and CNN models reduce validation errors compared with TF-IDF.
- CNN and BiLSTM perform very similarly, suggesting that short phrases contain much of the useful information.
- Adding more architectural complexity does not always produce a meaningful improvement.
- DistilBERT achieves the strongest overall validation performance.
- The models perform much worse on manually written challenge tweets that do not follow the strong keyword patterns found in the training data.
- Validation results should therefore be interpreted carefully.

## Notebook

The complete workflow is contained in:

```text
notebook/GBV_rerun_Preprocessing_Modeling_with_Part3.ipynb
```

The notebook contains:

1. Data loading
2. Exploratory data analysis
3. Data quality checks
4. Text preprocessing
5. Train and validation splitting
6. TF-IDF + Logistic Regression experiments
7. BiLSTM experiments
8. 1D CNN experiments
9. CNN-BiLSTM experiments
10. DistilBERT experiments
11. Model comparison
12. Statistical evaluation
13. Error analysis
14. Keyword reliance analysis
15. Challenge tweet evaluation
16. Test predictions
17. Submission generation

## Installation

Python 3.10 or later is recommended.

Create a virtual environment:

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Notebook

### Google Colab

Google Colab is recommended because the notebook includes TensorFlow, PyTorch and DistilBERT experiments.

1. Open `notebook/GBV_rerun_Preprocessing_Modeling_with_Part3.ipynb` in Google Colab.
2. Go to **Runtime > Change runtime type**.
3. Select a GPU runtime if available.
4. Run the notebook from the first cell.
5. Upload `Train.csv` and `Test.csv` when prompted.
6. Continue running the cells from top to bottom.
7. Do not skip preprocessing or model setup cells.

A GPU is strongly recommended for DistilBERT and the neural network experiments.

### Local Jupyter Notebook

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebook/GBV_rerun_Preprocessing_Modeling_with_Part3.ipynb
```

If running locally, the Google Colab upload section may need to be replaced with direct file loading:

```python
import pandas as pd

train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
```

Then run the remaining notebook cells in order.

## Reproducing the Results

For the closest reproduction of the reported results:

1. Use the original `Train.csv` and `Test.csv` files.
2. Install the dependencies from `requirements.txt`.
3. Start with a fresh runtime or kernel.
4. Run the notebook from the first cell to the last.
5. Keep the original preprocessing and split configuration.
6. Keep the random seeds unchanged.
7. Use a GPU for the neural and DistilBERT experiments when possible.

The neural experiments use the following seeds:

```python
SEEDS = [42, 7, 123]
```

Small differences may still occur because neural network training can vary depending on the hardware, TensorFlow, PyTorch and CUDA environment.

## Output

During execution, the notebook generates processed datasets, experiment results, evaluation tables, plots and final predictions.

The final workflow includes generation of a submission file containing:

```text
Tweet_ID
type
```

for the test dataset.

## Limitations

The dataset has several important limitations:

- Severe class imbalance
- Very small minority classes
- Strong class-specific keywords
- Repeated tweets
- Some ambiguous or conflicting examples
- Possible differences between training and test distributions

The very high validation scores should therefore not be interpreted as evidence that the models will perform equally well on all real-world GBV discussions.

## Responsible Use

GBV data can contain sensitive personal experiences and allegations.

The models in this project are intended for research and analytical purposes.

They should not be used to determine whether an individual experienced violence, whether an allegation is true, or to make decisions about individuals without appropriate human review.

Any real-world application should consider privacy, data governance, bias, annotation quality and the consequences of incorrect predictions.

## Authors

**Principie Cyubahiro**

**Ineza Bella Melissa**

**Ulrich Rukazambuga**

## Acknowledgements

The project uses the Gender-Based Violence Tweet Classification dataset and open-source tools from the Python machine learning and natural language processing ecosystem.
