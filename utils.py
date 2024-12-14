import os
import subprocess
import wget
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

NEGATIVE_SAMPLES_PER_USER = 80


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