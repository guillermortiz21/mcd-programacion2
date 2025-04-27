# Programación 2
# Maestría en Ciencia de Datos
# Challenge 2
# Guillermo Ortiz Macías

# Python libraries
print("Importing libraries")
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import torch
import mlflow
import warnings
warnings.filterwarnings('ignore')

def get_dataset():
    # The dataset has information about reviews of jobs in glassdoor,
    # and a column that indicates if the person recomends or not the job.
    data_path = "glassdoor_reviews.csv"
    df_job_reviews = pd.read_csv(data_path)
    return df_job_reviews

def preProcess_dataset(df_job_reviews):
    # Preprocess dataset so that it only has the text review and the column where
    # the user says if the job is recommended or not.
    
    # Remove rows without an opinion
    df_job_reviews = df_job_reviews[df_job_reviews['recommend'] != 'o']

    # Join the headline, pros and cons columns to have only one text review.
    df_job_reviews['text'] = df_job_reviews['headline'] + " " + df_job_reviews['pros'] + " " + df_job_reviews['cons']

    # Drop all columns except text and recommend
    df_job_reviews = df_job_reviews[['text', 'recommend']].copy()

    # Remove null values
    df_job_reviews = df_job_reviews.dropna()

    # Reset indexes
    df_job_reviews = df_job_reviews.reset_index().drop('index',axis=1)

    # Convert recommend column values to 0 (not recommended) or 1 (Recommended)
    df_job_reviews['recommend'] = np.where(df_job_reviews['recommend'] == 'x', 0,1)

    return df_job_reviews

def get_model_and_tokenizer():
    # This link comes from the site huggingFace.co:
    # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
    # This is a model of sentiment analysis that works with 6 different languages, including
    # spanish and english.
    # The model returns a sentiment analysis with a probability of 5 different sentiments,
    # from very bad to very good.
    model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def get_sentiment_review(review, tokenizer, model):
    tokens = tokenizer.encode(review, return_tensors="pt")
    tokenizer.decode(tokens[0])
    result = model(tokens)
    # Get biggest probability from 0 to 4
    sentiment = torch.argmax(result.logits)
    # If sentiment is 0,1 or 2 return 0: Negative sentiment
    if sentiment <= 2:
        return 0
    # else, return 1: possitive sentiment
    return 1

def get_dataset_sample(df_job_reviews, sample_size):
    df_sample = df_job_reviews.sample(sample_size)
    return df_sample

def get_sentiment_predictions(df_sample, tokenizer, model):
    # Now make a prediction of the sentiment for each of the
    sentiment_predictions = []
    len_df = len(df_sample.index)
    i = 1
    print(f"Predicting {i} of {len_df}")
    for index, row in df_sample.iterrows():
        if(i % 50 == 0):
            print(f"Predicting {i} of {len_df}")
        # The model has a limit of 512 tokens that can analyze at the same time.
        # Because of that I get only the first 512 tokens for each review.
        review = row['text'][:512]
        pred_rec = get_sentiment_review(review, tokenizer, model)
        sentiment_predictions.append(pred_rec)
        i = i + 1
    return sentiment_predictions

def evaluate_model(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # Log metrics in mlflow
    mlflow.log_metric("Confusion matrix_TP", conf_matrix[0][0])
    mlflow.log_metric("Confusion matrix_TN", conf_matrix[1][1])
    mlflow.log_metric("Confusion matrix_FP", conf_matrix[0][1])
    mlflow.log_metric("Confusion matrix_FN", conf_matrix[1][0])
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Accuracy", accuracy)
    print(f"Confussion matrix:\n{conf_matrix}")
    print(f"Precision:\n{precision:.4f}")
    print(f"Accuracy:\n{accuracy:.4f}")

def main():
    # Start MLFlow experiment
    mlflow.set_experiment("Glassdoor reviews sentiment analysis")
    with mlflow.start_run():
        # Get dataset
        print("Getting dataset")
        df_job_reviews = get_dataset()
        # Pre process it
        print("Pre processing dataset")
        df_job_reviews = preProcess_dataset(df_job_reviews)
        # Get text classification model and tokenizer
        print("Getting tokenizer and LLM model")
        tokenizer, model = get_model_and_tokenizer()
        # The dataset has a length of 603109 rows, it will take a lot of time to
        # test the sentiment of all of them. Instead I will make a sample of 10,000
        # reviews and test the model with it.
        sample_size = 200
        df_sample = get_dataset_sample(df_job_reviews, sample_size)
        # With the model and the tokenizer I can now make predictions of the
        # job reviews in the dataset.
        print("Predicting sentiment for dataset reviews")
        sentiment_predictions = get_sentiment_predictions(df_sample, tokenizer, model)
        y_test = df_sample['recommend']
        print("Evaluating models")
        evaluate_model(y_test, y_pred=sentiment_predictions)

main()
