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
