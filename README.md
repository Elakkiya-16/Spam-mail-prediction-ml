# Spam Mail Prediction using Machine Learning

## Overview
This project classifies text messages as **spam** or **ham** (legitimate) using TF-IDF feature extraction and Logistic Regression.

## Problem Statement
Given the text of a message, predict whether it belongs to the spam or ham category.

The notebook encodes the labels as:
- `0` — spam
- `1` — ham

## Dataset
- File: `mail_data.csv`
- Records: 5,572
- Columns: 2
- `Category`: message label (`spam` or `ham`)
- `Message`: message text

## Technologies and Libraries
- Python
- NumPy
- Pandas
- scikit-learn
- `TfidfVectorizer`
- Logistic Regression
- `accuracy_score`

## Workflow
1. Import the required libraries.
2. Load `mail_data.csv` into a Pandas DataFrame.
3. Replace missing values with empty strings using `where()` and `pd.notnull()`.
4. Encode `spam` as `0` and `ham` as `1`.
5. Separate message text (`X`) and labels (`Y`).
6. Split the records using `train_test_split(test_size=0.2, random_state=3)`.
7. Create a `TfidfVectorizer` with `min_df=1`, English stop-word removal and lowercase conversion.
8. Fit the vectorizer on training text and transform the training text into TF-IDF features.
9. Transform test text with the already-fitted vectorizer.
10. Convert the target labels to integers.
11. Train a Logistic Regression classifier.
12. Evaluate training and testing accuracy.
13. Transform a new message with the same fitted vectorizer and predict its class.

## TF-IDF
TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It converts text into numerical feature vectors. The vectorizer learns its vocabulary and inverse-document-frequency values from the training messages. The test and new messages are transformed using that already-fitted vectorizer.

## Model
The notebook uses **Logistic Regression** to classify the TF-IDF vectors as spam or ham.

## Results
| Evaluation | Accuracy |
|---|---:|
| Training accuracy | 96.70% |
| Testing accuracy | 96.59% |

Accuracy is the proportion of predictions that match the actual labels for the evaluated split. Precision, recall, F1-score and a confusion matrix can provide additional insight, especially into spam messages incorrectly classified as legitimate.

## Example Prediction
The notebook tests this example message:
> I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times

The notebook's displayed prediction is `1`, which maps to `ham`, so it prints `Ham mail`.

## How to Run
1. Place `Spam_Mail_Prediction_using_Machine_Learning.ipynb` and `mail_data.csv` in an accessible working location.
2. Install the dependencies:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
3. Update the CSV path in the notebook if necessary.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the cells from top to bottom.

## Notes
- The model expects text input.
- New messages must be transformed with the same fitted `TfidfVectorizer` used during training.
- The notebook uses a random train-test split without specifying `stratify`; results correspond to that split and may vary if the split changes.
