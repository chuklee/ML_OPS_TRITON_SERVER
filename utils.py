import os
import subprocess
import wget
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch.nn import functional as F


NEGATIVE_SAMPLES_PER_USER = 80

import torch
from torch.utils.data import Dataset

class cls_dataset(Dataset):
    def __init__(self, data, user_count, item_count, title_emb_tensor, title_to_idx):
        super().__init__()
        self.m_data = data.reset_index(drop=True)
        self.user_count = user_count
        self.item_count = item_count
        self.title_emb_tensor = title_emb_tensor  # Keep on CPU
        self.title_to_idx = title_to_idx

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.m_data)

    def __getitem__(self, index):
        """Retrieve a single sample from the dataset."""
        user_id = self.m_data.at[index, "user_id"]
        item_id = self.m_data.at[index, "item_id"]

        # User features
        user_features = torch.tensor(
            self.m_data.iloc[index][["user_id", "age", "occupation"]].values.astype(np.float32),
            dtype=torch.float32
        )
        user_features = torch.cat((
            user_features,
            torch.tensor(
                [self.m_data.at[index, "gen_F"], self.m_data.at[index, "gen_M"]],
                dtype=torch.float32
            )
        ))

        # Item features
        item_features = torch.tensor(
            self.m_data.iloc[index][[
                "item_id", "timestamp", "unknown", "Action", "Adventure", "Animation", "Children",
                "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
            ]].values.astype(np.float32),
            dtype=torch.float32
        )

        # Title embedding
        title = self.m_data.at[index, "title"]
        title_idx = self.title_to_idx.get(title, -1)
        if title_idx != -1:
            ts_title = self.title_emb_tensor[title_idx]
        else:
            ts_title = torch.zeros(384)  # Keep on CPU

        # Concatenate item features with title embeddings
        concatenated = torch.cat((item_features, ts_title), dim=0)

        return (
            user_features,
            concatenated,
            torch.tensor(int(self.m_data.at[index, "like"]), dtype=torch.long),
            torch.tensor(float(self.m_data.at[index, "rating"]), dtype=torch.float32)
        )

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
def download_and_extract_dataset():
    """
    Download and extract the MovieLens dataset 1m
    """	
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    
    print("Downloading MovieLens dataset...")
    if not os.path.exists(zip_path):
        wget.download(url, zip_path)
    
    # Unzip the file
    print("\nExtracting dataset...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Dataset downloaded and extracted successfully!")

def read_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the dataset
    """
    # Read the data files with the correct separator and column names
    df_movies = pd.read_csv("data/ml-1m/movies.dat", 
                        sep="::", 
                        header=None, 
                        names=['item_id', 'title', 'genres'],
                        encoding='latin-1',
                        engine='python')



    df_ratings = pd.read_csv("data/ml-1m/ratings.dat", 
                            sep="::", 
                            header=None, 
                            names=['user_id', 'item_id', 'rating', 'timestamp'],
                            encoding='latin-1',
                            engine='python')

    df_users = pd.read_csv("data/ml-1m/users.dat", 
                        sep="::", 
                        header=None, 
                        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                        encoding='latin-1',
                        engine='python')
    return df_movies, df_ratings, df_users

def preprocess_dataset(df_movies: pd.DataFrame, df_ratings: pd.DataFrame, df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset
    """
    all_genres = set()
    for genres in df_movies['genres'].str.split('|'):
        all_genres.update(genres)

    # Replace "Children's" with "Children" to match format
    all_genres = {genre.replace("Children's", "Children") for genre in all_genres}
    # Replace "Film-Noir" with "Film_Noir" to match format
    all_genres = {genre.replace("Film-Noir", "Film_Noir") for genre in all_genres}
    # Create genre columns, including 'unknown'
    genres = sorted(list(all_genres))
    genres.append('unknown')  # Add 'unknown' to the genre list
    genre_columns = {genre: [] for genre in genres}

    # Fill genre columns, handling missing genres
    for _, row in df_movies.iterrows():
        movie_genres = row['genres'].split('|')
        # Replace genre names to match format
        movie_genres = [g.replace("Children's", "Children").replace("Film-Noir", "Film_Noir") for g in movie_genres]
        
        if not movie_genres:  # Check if movie_genres is empty
            for genre in genres:
                genre_columns[genre].append(1 if genre == 'unknown' else 0)
        else:
            for genre in genres:
                genre_columns[genre].append(1 if genre in movie_genres else 0)

    # Add genre columns to movies dataframe
    for genre in genres:
        df_movies[genre] = genre_columns[genre]

    # Drop the original genres column
    df_movies = df_movies.drop('genres', axis=1)

    # Create gender dummy variables
    df_users = pd.get_dummies(df_users, prefix=['gen'], columns=['gender'])

    # Merge all dataframes
    df_combined = pd.merge(df_ratings, df_users, on='user_id', how='inner')
    df_combined = pd.merge(df_combined, df_movies, on='item_id', how='inner')


    # Reorder columns to match desired format
    column_order = ['user_id', 'item_id', 'rating', 'timestamp', 'age', 'occupation', 
                    'zip_code', 'title'] + genres + ['gen_F', 'gen_M']

    df_combined = df_combined[column_order]
    model_norm = StandardScaler()
    df_combined[["age","timestamp"]] = model_norm.fit_transform(df_combined[["age", "timestamp"]])
    df_combined["title"] = df_combined["title"].str.lower().str.replace(r"[\:\&\,]","")
    df_combined["user_id"] = df_combined["user_id"] - 1
    df_combined["item_id"] = df_combined["item_id"] - 1
    df_combined['like'] = (df_combined['rating'] >= 4).astype(int)
    return df_combined

def preprocess_title(title):
    """
    Preprocess the movie title by converting to lowercase and removing specific punctuation.
    """
    if pd.isna(title):
        raise ValueError(f"Title is NaN: {title}")
    return title.lower().replace(':', '').replace('&', '').replace(',', '')

def generate_negative_samples(df_source, user_ids, all_item_ids, user_positive_items, df_movies, df_users, sample_size=NEGATIVE_SAMPLES_PER_USER):
    """
    Generate negative samples with complete item and user information.

    Args:
        df_source (pd.DataFrame): The source DataFrame containing user-item interactions.
        user_ids (list): List of unique user IDs.
        all_item_ids (list): List of all unique item IDs.
        user_positive_items (dict): Dictionary mapping user IDs to sets of positive item IDs.
        df_movies (pd.DataFrame): DataFrame containing movie information.
        df_users (pd.DataFrame): DataFrame containing user information.
        sample_size (int): Number of negative samples to generate per user.

    Returns:
        pd.DataFrame: DataFrame containing negative samples with item and user information.
    """
    neg_samples = {
        'user_id': [],
        'item_id': []
    }

    for user in user_ids:
        positive_items = user_positive_items.get(user, set())
        negative_candidates = np.setdiff1d(all_item_ids, list(positive_items))

        if len(negative_candidates) == 0:
            print(f"User {user} has interacted with all items. Skipping negative sampling.")
            continue

        if len(negative_candidates) >= sample_size:
            sampled_items = np.random.choice(negative_candidates, size=sample_size, replace=False)
        else:
            sampled_items = np.random.choice(negative_candidates, size=sample_size, replace=True)

        neg_samples['user_id'].extend([user] * len(sampled_items))
        neg_samples['item_id'].extend(sampled_items)

    # Create DataFrame with negative samples
    df_neg = pd.DataFrame(neg_samples)

    # Merge with users data to get user features using an inner join to ensure all user_ids are valid
    df_neg = df_neg.merge(df_users, on='user_id', how='inner')

    # Merge with movies data to get item features
    df_neg = df_neg.merge(df_movies, on='item_id', how='left')  # Assuming all item_ids are present

    # Assign default values for negative samples
    df_neg['rating'] = 0  # Assuming 0 indicates no rating
    df_neg['timestamp'] = df_source['timestamp'].max() + 1  # Assign a timestamp after the last interaction
    df_neg['like'] = 0  # Negative samples have 'like' as 0

    # Identify all columns that need to be present
    required_columns = df_source.columns.tolist()

    # Fill missing columns with default values
    for col in required_columns:
        if col not in df_neg.columns:
            if col in ['age', 'occupation', 'zip_code']:
                df_neg[col] = 0  # Assign a default integer value
            elif col in ['gen_F', 'gen_M']:
                df_neg[col] = False  # Assign a default boolean value
            else:
                df_neg[col] = 0  # Assign a generic default value
        else:
            # For user-specific columns, ensure no NaNs
            if col in ['age', 'occupation', 'zip_code']:
                df_neg[col] = df_neg[col].fillna(0)
            elif col in ['gen_F', 'gen_M']:
                df_neg[col] = df_neg[col].fillna(False)

    return df_neg
def debug_shapes(model, x1, x2):
    """Debug function to print model shapes"""
    print("Input shapes:")
    print(f"x1: {x1.shape}")
    print(f"x2: {x2.shape}")
    
    # Forward pass through user model
    user_emb = model.m_modelUser(x1)
    print(f"User embedding: {user_emb.shape}")
    
    # Forward pass through item model
    item_emb = model.m_modelItem(x2)
    print(f"Item embedding: {item_emb.shape}")
    
    # Concatenate embeddings
    combined = torch.cat([user_emb, item_emb], dim=1)
    print(f"Combined embedding: {combined.shape}")
    
    # Final classification
    output = model.m_modelClassify(combined)
    print(f"Final output: {output.shape}")

def prepare_data(df_combined, df_movies, df_users):
    """
    Prepare train and test datasets with negative sampling
    """
    df_train, df_test = train_test_split(df_combined, train_size=0.8, random_state=42, shuffle=True)


    # Step 2: Create user-item interaction dictionary
    user_positive_items = df_combined.groupby('user_id')['item_id'].apply(set).to_dict()

    # Step 3: Get all unique item IDs
    all_item_ids = df_combined['item_id'].unique()
    # Verify that all item_ids exist in df_movies
    missing_item_ids = set(all_item_ids) - set(df_movies['item_id'])
    if missing_item_ids:
        # Optionally, remove these item_ids from all_item_ids to prevent sampling them
        all_item_ids = np.array(list(set(all_item_ids) - missing_item_ids))

    # Step 4: Generate negative samples for training and testing
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

    # Apply title preprocessing to df_train and df_test after combining
    df_train["title"] = df_train["title"].apply(preprocess_title)
    df_test["title"] = df_test["title"].apply(preprocess_title)


    unique_item_ids = df_combined['item_id'].unique()
    item_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_item_ids)}

    # Apply mapping to all DataFrames
    for df_name, df in zip(['df_combined', 'df_train', 'df_test'], [df_combined, df_train, df_test]):
        df['item_id'] = df['item_id'].map(item_id_mapping)


    # Step 8: Drop any rows with missing 'item_id's
    df_train = df_train.dropna(subset=['item_id']).reset_index(drop=True)

    df_test = df_test.dropna(subset=['item_id']).reset_index(drop=True)

    # Step 9: Convert 'item_id' to integer type
    for df_name, df in zip(['df_combined', 'df_train', 'df_test'], [df_combined, df_train, df_test]):
        df['item_id'] = df['item_id'].astype(int)

    # Step 10: Shuffle the final datasets
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True )

    return df_train,df_test
def encode_title(df_train,df_test):
    # Initialize the Sentence Transformer model
    model_strans = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize the dictionary to store embeddings
    dict_titlEmb = {}

    # Combine unique titles from both training and testing sets
    all_unique_titles = pd.concat([df_train, df_test])["title"].unique()

    # Encode each unique title and store it in the dictionary
    for title in all_unique_titles:
        dict_titlEmb[title] = torch.tensor(model_strans.encode(title))

    # Verify that all titles are embedded
    combined_unique_titles = set(df_train['title'].unique()).union(set(df_test['title'].unique()))
    missing_titles = combined_unique_titles - set(dict_titlEmb.keys())

    if missing_titles:
        # Optionally, handle missing titles
        for title in missing_titles:
            dict_titlEmb[title] = torch.tensor(model_strans.encode(title))
    return dict_titlEmb
