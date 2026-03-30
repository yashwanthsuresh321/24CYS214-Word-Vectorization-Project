import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.preprocess import preprocess

# 1. DATA LOADING 
def load_data():
    """
    Loads the IMDb dataset using a relative path for reproducibility.
    Expected structure: Project_Root/data/IMDB_Dataset.csv
    """
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "..", "data", "IMDB_Dataset.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure the 'data' folder contains 'IMDB_Dataset.csv'.")
    
    df = pd.read_csv(file_path)
    return df

# 2. CONVENTIONAL METHODS 

def run_bow_experiment(X_train, X_test, y_train, y_test):
    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    
    model = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
    model.fit(X_train_bow, y_train)
    y_pred = model.predict(X_test_bow)
    return y_pred

def run_tfidf_experiment(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=15000, sublinear_tf=True, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    
    model = LinearSVC(C=1.0, max_iter=2000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    return y_pred

# 3. DEEP LEARNING METHODS 

def run_word2vec_experiment(tokenized_train, tokenized_test, y_train, y_test):
    w2v_model = Word2Vec(sentences=tokenized_train, sg=1, vector_size=200, window=5, min_count=2, workers=4, epochs=10)
    
    def get_mean_vector(tokens):
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(200)

    X_train_w2v = np.array([get_mean_vector(t) for t in tokenized_train])
    X_test_w2v = np.array([get_mean_vector(t) for t in tokenized_test])
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_w2v, y_train)
    return model.predict(X_test_w2v)

def run_bert_config():
    config = {
        "model": "bert-base-uncased",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5
    }
    print(f"BERT Configuration: {config}")
    return config

# 4. EVALUATION UTILITY 
def evaluate_results(y_true, y_pred, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4%}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4%}")
    print(f"Precision: {precision_score(y_true, y_pred):.4%}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4%}")

if __name__ == "__main__":
    try:
        data = load_data()
        data['cleaned_review'] = data['review'].apply(preprocess)
        
        X_train, X_test, y_train, y_test = train_test_split(
            data['cleaned_review'], data['sentiment'], test_size=0.5, random_state=42
        )
        
        y_pred_bow = run_bow_experiment(X_train, X_test, y_train, y_test)
        evaluate_results(y_test, y_pred_bow, "BoW + Logistic Regression")
        
    except Exception as e:
        print(f"Error: {e}")