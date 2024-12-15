from typing import Dict, List, Optional
import redis
import torch
import pandas as pd
import numpy as np
from models import cls_model
from utils import encode_title, preprocess_dataset, read_dataset
import faiss
import json
from datetime import datetime

class DataPipeline:
    def __init__(self):
        # Load and preprocess initial data
        self.df_movies, self.df_ratings, self.df_users = read_dataset()
        self.df_combined = preprocess_dataset(self.df_movies, self.df_ratings, self.df_users)
        
        # Initialize title embeddings
        self.dict_titleEmb = encode_title(self.df_combined, self.df_combined)
        self.all_unique_titles = self.df_combined["title"].unique()
        self.title_to_idx = {title: idx for idx, title in enumerate(self.all_unique_titles)}
        
        # Create title embedding tensor
        list_titleEmb = [self.dict_titleEmb[title] for title in self.title_to_idx]
        self.title_emb_tensor = torch.stack(list_titleEmb)

    def get_item_features(self, item_id: int) -> Dict:
        """Get all features for a specific item"""
        item_data = self.df_movies[self.df_movies['item_id'] == item_id].iloc[0]
        return item_data.to_dict()

    def get_user_features(self, user_id: int) -> Dict:
        """Get all features for a specific user"""
        user_data = self.df_users[self.df_users['user_id'] == user_id].iloc[0]
        return user_data.to_dict()
    
    def get_similar_demographic_users(self, features: Dict) -> List[int]:
        """Find users with similar demographic features"""
        similar_users = self.df_users[
            (self.df_users['age'].between(features['age'] - 5, features['age'] + 5)) &
            (self.df_users['occupation'] == features['occupation'])
        ]['user_id'].tolist()
        return similar_users

    def get_popular_items_for_demographic(self, similar_users: List[int], n: int = 20) -> List[Dict]:
        """Get popular items among similar users"""
        popular_items = (
            self.df_ratings[self.df_ratings['user_id'].isin(similar_users)]
            .groupby('item_id')
            .agg({'rating': ['count', 'mean']})
            .sort_values(by=[('rating', 'count'), ('rating', 'mean')], ascending=False)
            .head(n)
        ).index.tolist()
        
        return [self.get_item_features(item_id) for item_id in popular_items]

    def store_new_user(self, user_id: int, features: Dict):
        """Store new user data"""
        new_user = pd.DataFrame([{
            'user_id': user_id,
            'age': features['age'],
            'occupation': features['occupation'],
            'gender': 'F' if features['gen_F'] == 1 else 'M'
        }])
        self.df_users = pd.concat([self.df_users, new_user], ignore_index=True)