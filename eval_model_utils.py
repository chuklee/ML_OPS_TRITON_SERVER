import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from models import cls_model
from utils import encode_title

def get_user_top_rated_movies(user_id, k, df_combined):
    """
    Retrieve top k rated movies for a specific user.
    
    Args:
    - user_id: ID of the user
    - k: Number of top rated movies to retrieve
    
    Returns:
    - DataFrame of top rated movies
    """
    # Filter ratings for the specific user
    user_ratings = df_combined[df_combined['user_id'] == user_id]
    
    # Check if the DataFrame is empty
    if user_ratings.empty:
        print(f"No ratings found for user ID {user_id}.")
        return None  # or handle accordingly

    # Sort by rating in descending order and get top k
    top_rated = user_ratings.sort_values('rating', ascending=False).head(k)
    
    # List of genre columns
    genre_columns = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    # Return the relevant columns including genre columns
    return top_rated[['item_id', 'title', 'rating'] + genre_columns]

def get_top_k_recommendations(model, user_id, k, device, df_combined, df_movies):
    """
    Get top k item recommendations for a given user with more detailed information.
    
    Args:
    - model: Trained recommendation model
    - user_id: ID of the user to get recommendations for
    - k: Number of top recommendations to retrieve
    - device: Device to run the model on (cuda/cpu)
    
    Returns:
    - List of top k item recommendations with their predicted scores
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Convert user_id to tensor and move to device
    user_tensor = torch.tensor([user_id]).to(device)
    
    # Get similarity scores for the user with all items
    with torch.no_grad():
        similarity_scores = model.predict(user_tensor).squeeze().cpu()
    
    # Sort items by similarity scores in descending order
    _, top_item_indices = torch.topk(similarity_scores, k=k)
    
    # Prepare recommendations with item details
    recommendations = []
    unique_item_ids = df_combined['item_id'].unique()
    item_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_item_ids)}

    for item_idx in top_item_indices:
        # Find the original item ID
        original_item_id = list(item_id_mapping.keys())[list(item_id_mapping.values()).index(item_idx.item())]
        
        # Get movie details
        movie_info = df_movies[df_movies['item_id'] == original_item_id]
        
        # Check if movie_info is empty
        if movie_info.empty:
            print(f"No movie information found for item ID {original_item_id}.")
            continue  # Skip this iteration if no movie info is found
        
        # Extract genre information safely
        genre_columns = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Ensure all genre columns exist, if not, create them
        movie_genres = []
        for genre in genre_columns:
            if genre in movie_info.columns:
                movie_genres.append(movie_info[genre].values[0])
            else:
                movie_genres.append(0)
        
        recommendations.append({
            'item_id': original_item_id,
            'title': movie_info['title'].values[0],
            'score': similarity_scores[item_idx].item(),
            'genres': movie_genres  # Store the one-hot encoded genres
        })
    
    return recommendations

def detailed_user_recommendations(user_id, modelRec, df_combined, device, df_train,df_test, title_to_idx, title_emb_tensor):
    """
    Provide a comprehensive recommendation analysis for a user.
    """
    print(f"Recommendation Analysis for User ID {user_id}:\n")
    
    modelRec.to(device)
    modelRec.eval()
    
    # Initialize counters for prediction statistics
    liked_correct = 0
    liked_total = 0
    disliked_correct = 0
    disliked_total = 0
    
    # Get user data
    user_rated_movies = df_combined[df_combined['user_id'] == user_id]
    
    # Separate training and test data for this user
    user_train_data = df_train[df_train['user_id'] == user_id]
    user_test_data = df_test[df_test['user_id'] == user_id]
    
    print(f"\nData Distribution for User {user_id}:")
    print(f"Total ratings: {len(user_rated_movies)}")
    print(f"Training set ratings: {len(user_train_data)}")
    print(f"Test set ratings: {len(user_test_data)}")
    
    # Analyze only test data
    liked_movies = user_test_data[user_test_data['rating'] >= 4]
    disliked_movies = user_test_data[user_test_data['rating'] < 4]
    
    print(f"\nAnalyzing only test set data:")
    print(f"Number of liked movies in test set: {len(liked_movies)}")
    print(f"Number of disliked movies in test set: {len(disliked_movies)}\n")
    
    # Prepare user features
    user_data = df_combined[df_combined['user_id'] == user_id].iloc[0]
    user_features = torch.tensor([
        user_data['user_id'],
        user_data['age'],
        user_data['occupation'],
        user_data['gen_F'],
        user_data['gen_M']
    ], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict likes for liked movies (test set only)
    for _, movie in liked_movies.iterrows():
        liked_total += 1
        
        # Prepare item features
        item_features = torch.tensor(
            movie[[
                'item_id', 'timestamp', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western'
            ]].values.astype(np.float32)
        ).unsqueeze(0).to(device)
        
        # Get title embedding
        title = movie['title']
        title_idx = title_to_idx.get(title, -1)
        if title_idx != -1:
            title_emb = title_emb_tensor[title_idx].to(device)
        else:
            title_emb = torch.zeros(384, device=device)
        
        # Concatenate item features with title embedding
        item_features = torch.cat((item_features, title_emb.unsqueeze(0)), dim=1)
        
        with torch.no_grad():
            logits = modelRec(user_features, item_features)
            predicted_like = torch.argmax(logits, dim=1).item()
            
            if predicted_like == 1:  # Model predicted "like"
                liked_correct += 1
                

    # Predict likes for disliked movies (test set only)
    for _, movie in disliked_movies.iterrows():
        disliked_total += 1
        
        # Prepare item features
        item_features = torch.tensor(
            movie[[
                'item_id', 'timestamp', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                'Thriller', 'War', 'Western'
            ]].values.astype(np.float32)
        ).unsqueeze(0).to(device)
        
        # Get title embedding
        title = movie['title']
        title_idx = title_to_idx.get(title, -1)
        if title_idx != -1:
            title_emb = title_emb_tensor[title_idx].to(device)
        else:
            title_emb = torch.zeros(384, device=device)
        
        # Concatenate item features with title embedding
        item_features = torch.cat((item_features, title_emb.unsqueeze(0)), dim=1)
        
        with torch.no_grad():
            logits = modelRec(user_features, item_features)
            predicted_like = torch.argmax(logits, dim=1).item()
            
            if predicted_like == 0:  # Model predicted "dislike"
                disliked_correct += 1
                

    # Calculate and display statistics (test set only)
    print("\n=== Prediction Statistics (Test Set Only) ===")
    
    # Liked movies statistics
    liked_accuracy = (liked_correct / liked_total * 100) if liked_total > 0 else 0
    print(f"\nLiked Movies (Rating >= 4):")
    print(f"Total in test set: {liked_total}")
    print(f"Correctly Predicted: {liked_correct}")
    print(f"Accuracy: {liked_accuracy:.2f}%")
    
    # Disliked movies statistics
    disliked_accuracy = (disliked_correct / disliked_total * 100) if disliked_total > 0 else 0
    print(f"\nDisliked Movies (Rating < 4):")
    print(f"Total in test set: {disliked_total}")
    print(f"Correctly Predicted: {disliked_correct}")
    print(f"Accuracy: {disliked_accuracy:.2f}%")
    
    # Overall statistics
    total_movies = liked_total + disliked_total
    total_correct = liked_correct + disliked_correct
    overall_accuracy = (total_correct / total_movies * 100) if total_movies > 0 else 0
    print(f"\nOverall Test Set Statistics:")
    print(f"Total Movies in test set: {total_movies}")
    print(f"Total Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
def analyze_multiple_users(num_users, seed, df_test, modelRec, device, df_combined, title_to_idx,title_emb_tensor):
    """
    Analyze predictions for multiple random users.
    
    Args:
    - num_users: Number of users to analyze
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Get unique users from test set
    unique_users = df_test['user_id'].unique()
    
    # Randomly sample users
    selected_users = np.random.choice(unique_users, min(num_users, len(unique_users)), replace=False)
    
    # Initialize global statistics
    total_liked_correct = 0
    total_liked = 0
    total_disliked_correct = 0
    total_disliked = 0
    
    # Store individual user statistics
    user_stats = []
    
    modelRec.to(device)
    modelRec.eval()
    
    print(f"Analyzing {len(selected_users)} users...\n")
    
    for user_id in tqdm(selected_users):
        # Get user test data
        user_test_data = df_test[df_test['user_id'] == user_id]
        
        if user_test_data.empty:
            continue
            
        # Split into liked and disliked
        liked_movies = user_test_data[user_test_data['rating'] >= 4]
        disliked_movies = user_test_data[user_test_data['rating'] < 4]
        
        # Initialize user statistics
        user_liked_correct = 0
        user_disliked_correct = 0
        
        # Prepare user features
        user_data = df_combined[df_combined['user_id'] == user_id].iloc[0]
        user_features = torch.tensor([
            user_data['user_id'],
            user_data['age'],
            user_data['occupation'],
            user_data['gen_F'],
            user_data['gen_M']
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Process liked movies
        for _, movie in liked_movies.iterrows():
            item_features = torch.tensor(
                movie[[
                    'item_id', 'timestamp', 'unknown', 'Action', 'Adventure', 'Animation',
                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western'
                ]].values.astype(np.float32)
            ).unsqueeze(0).to(device)
            
            title = movie['title']
            title_idx = title_to_idx.get(title, -1)
            if title_idx != -1:
                title_emb = title_emb_tensor[title_idx].to(device)
            else:
                title_emb = torch.zeros(384, device=device)
            
            item_features = torch.cat((item_features, title_emb.unsqueeze(0)), dim=1)
            
            with torch.no_grad():
                logits = modelRec(user_features, item_features)
                predicted_like = torch.argmax(logits, dim=1).item()
                
                if predicted_like == 1:
                    user_liked_correct += 1
        
        # Process disliked movies
        for _, movie in disliked_movies.iterrows():
            item_features = torch.tensor(
                movie[[
                    'item_id', 'timestamp', 'unknown', 'Action', 'Adventure', 'Animation',
                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western'
                ]].values.astype(np.float32)
            ).unsqueeze(0).to(device)
            
            title = movie['title']
            title_idx = title_to_idx.get(title, -1)
            if title_idx != -1:
                title_emb = title_emb_tensor[title_idx].to(device)
            else:
                title_emb = torch.zeros(384, device=device)
            
            item_features = torch.cat((item_features, title_emb.unsqueeze(0)), dim=1)
            
            with torch.no_grad():
                logits = modelRec(user_features, item_features)
                predicted_like = torch.argmax(logits, dim=1).item()
                
                if predicted_like == 0:
                    user_disliked_correct += 1
        
        # Update global statistics
        total_liked_correct += user_liked_correct
        total_liked += len(liked_movies)
        total_disliked_correct += user_disliked_correct
        total_disliked += len(disliked_movies)
        
        # Calculate user accuracies
        liked_acc = (user_liked_correct / len(liked_movies) * 100) if len(liked_movies) > 0 else 0
        disliked_acc = (user_disliked_correct / len(disliked_movies) * 100) if len(disliked_movies) > 0 else 0
        overall_acc = ((user_liked_correct + user_disliked_correct) / 
                      (len(liked_movies) + len(disliked_movies)) * 100)
        
        # Store user statistics
        user_stats.append({
            'user_id': user_id,
            'liked_accuracy': liked_acc,
            'disliked_accuracy': disliked_acc,
            'overall_accuracy': overall_acc,
            'total_movies': len(liked_movies) + len(disliked_movies)
        })
    
    # Calculate global statistics
    print("\n=== Global Statistics ===")
    
    # Liked movies
    global_liked_acc = (total_liked_correct / total_liked * 100) if total_liked > 0 else 0
    print(f"\nLiked Movies (Rating >= 4):")
    print(f"Total: {total_liked}")
    print(f"Correctly Predicted: {total_liked_correct}")
    print(f"Accuracy: {global_liked_acc:.2f}%")
    
    # Disliked movies
    global_disliked_acc = (total_disliked_correct / total_disliked * 100) if total_disliked > 0 else 0
    print(f"\nDisliked Movies (Rating < 4):")
    print(f"Total: {total_disliked}")
    print(f"Correctly Predicted: {total_disliked_correct}")
    print(f"Accuracy: {global_disliked_acc:.2f}%")
    
    # Overall
    total_movies = total_liked + total_disliked
    total_correct = total_liked_correct + total_disliked_correct
    global_acc = (total_correct / total_movies * 100) if total_movies > 0 else 0
    print(f"\nOverall Statistics:")
    print(f"Total Movies: {total_movies}")
    print(f"Total Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {global_acc:.2f}%")
    
    # Calculate distribution of accuracies
    accuracies = np.array([stat['overall_accuracy'] for stat in user_stats])
    print(f"\nAccuracy Distribution:")
    print(f"Mean: {np.mean(accuracies):.2f}%")
    print(f"Median: {np.median(accuracies):.2f}%")
    print(f"Std Dev: {np.std(accuracies):.2f}%")
    print(f"Min: {np.min(accuracies):.2f}%")
    print(f"Max: {np.max(accuracies):.2f}%")
    
    return user_stats

def evaluation(userCount, itemCount, device, df_train,df_test, df_combined):
    
    # Load model
    modelRec = cls_model(userCount, itemCount, user_embSize=32, item_embSize=32)
    modelRec.load_state_dict(torch.load('modelRec.pth'))
    modelRec.to(device)  # Move the model to the appropriate device
    modelRec.eval()  # Set the model to evaluation mode
    
    #Evaluation
    dict_titlEmb = encode_title(df_train,df_test)
    all_unique_titles = pd.concat([df_train, df_test])["title"].unique()
    title_to_idx = {title: idx for idx, title in enumerate(all_unique_titles)}
    list_titlEmb = [dict_titlEmb[title] for title in title_to_idx]
    title_emb_tensor = torch.stack(list_titlEmb)
    test_user_id = 10  # You can change this to any user ID in your dataset
    detailed_user_recommendations(test_user_id, modelRec, df_combined, device, df_train,df_test, title_to_idx, title_emb_tensor)
    user_stats = analyze_multiple_users(150, 42, df_test, modelRec, device, df_combined, title_to_idx,title_emb_tensor)
    print(user_stats)
    return user_stats