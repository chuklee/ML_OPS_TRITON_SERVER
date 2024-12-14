import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from pprint import pprint
from tqdm.notebook import tqdm

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer

import faiss

import matplotlib.pyplot as plt
import seaborn as sns

from utils import download_and_extract_dataset, read_dataset, preprocess_dataset, generate_negative_samples, preprocess_title, NEGATIVE_SAMPLES_PER_USER

if __name__ == "__main__":
    df_movies, df_ratings, df_users = read_dataset()
    df_combined = preprocess_dataset(df_movies, df_ratings, df_users)
    df_train, df_test = train_test_split(df_combined, train_size=0.8, random_state=42, shuffle=True)
    user_positive_items = df_combined.groupby('user_id')['item_id'].apply(set).to_dict()
    all_item_ids = df_combined['item_id'].unique()
    missing_item_ids = set(all_item_ids) - set(df_movies['item_id'])
    df_neg_train = generate_negative_samples(
        df_source=df_combined,
        user_ids=df_train['user_id'].unique(),
        all_item_ids=all_item_ids,
        user_positive_items=user_positive_items,
        df_movies=df_movies,
        df_users=df_users,
        sample_size=NEGATIVE_SAMPLES_PER_USER
    )
    df_neg_test = generate_negative_samples(
        df_source=df_combined,
        user_ids=df_test['user_id'].unique(),
        all_item_ids=all_item_ids,
        user_positive_items=user_positive_items,
        df_movies=df_movies,
        df_users=df_users,
        sample_size=NEGATIVE_SAMPLES_PER_USER
    )
    df_neg_train['like'] = 0
    df_neg_test['like'] = 0
    df_train = pd.concat([df_train, df_neg_train], axis=0).reset_index(drop=True)
    df_test = pd.concat([df_test, df_neg_test], axis=0).reset_index(drop=True)
    df_train["title"] = df_train["title"].apply(preprocess_title)
    df_test["title"] = df_test["title"].apply(preprocess_title)
    unique_item_ids = df_combined['item_id'].unique()
    item_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_item_ids)}
    for df_name, df in zip(['df_combined', 'df_train', 'df_test'], [df_combined, df_train, df_test]):
        print(f"\nMapping 'item_id' for {df_name}...")
        df['item_id'] = df['item_id'].map(item_id_mapping)
    initial_train_size = df_train.shape[0]
    df_train = df_train.dropna(subset=['item_id']).reset_index(drop=True)
    final_train_size = df_train.shape[0]
    initial_test_size = df_test.shape[0]
    df_test = df_test.dropna(subset=['item_id']).reset_index(drop=True)
    final_test_size = df_test.shape[0]
    for df_name, df in zip(['df_combined', 'df_train', 'df_test'], [df_combined, df_train, df_test]):
        df['item_id'] = df['item_id'].astype(int)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
