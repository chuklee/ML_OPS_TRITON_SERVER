import torch
import faiss
import numpy as np
from typing import List, Dict, Optional
import redis
from models import cls_model

class ModelServer:
    def __init__(self, model_path: str, device: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.device = torch.device(device)
        
        # Initialize model
        self.model = cls_model(userCount=6040, itemCount=3706)  # Update with your actual counts
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        
        # Initialize FAISS index for item embeddings
        self.build_item_index()

    def build_item_index(self):
        """Build FAISS index for fast similarity search"""
        with torch.no_grad():
            item_embeddings = self.model.m_itemEmb.weight.cpu().numpy()
            
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(item_embeddings.shape[1])  # Inner product similarity
        self.index.add(item_embeddings)

    def get_user_embedding(self, user_id: int, user_features: Dict) -> np.ndarray:
        """Get user embedding, using cache if available"""
        try:
            cache_key = f"user_emb_{user_id}".encode()  # Encode the key as bytes
            cached_embedding = self.redis_client.get(cache_key)
                
            if cached_embedding is not None:  # Check for None instead of truthiness
                return np.frombuffer(cached_embedding, dtype=np.float32).reshape(1, -1)
            
            # Get user embedding from the embedding layer
            user_id_tensor = torch.tensor([user_id], device=self.device)
            user_emb = self.model.m_userEmb(user_id_tensor)
            
            # Create user features tensor
            user_features_tensor = torch.tensor([
                user_features['age'],
                user_features['occupation'],
                user_features.get('gen_F', 0),
                user_features.get('gen_M', 0)
            ], dtype=torch.float32, device=self.device)
            
            # Concatenate user embedding with features
            user_input = torch.cat([user_features_tensor, user_emb.squeeze(0)], dim=0).unsqueeze(0)
            
            # Generate embedding through the user model
            with torch.no_grad():
                user_embedding = self.model.m_modelUser(user_input).cpu().numpy()
            
            # Cache the embedding
            self.redis_client.setex(
                cache_key,
                3600,  # Cache for 1 hour
                user_embedding.tobytes()
            )
                
            return user_embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate user embedding: {str(e)}")

    def get_recommendations(self, user_embedding: np.ndarray, k: int = 20) -> List[int]:
        """Get top-k recommendations using FAISS"""
        distances, indices = self.index.search(user_embedding, k)
        return indices[0].tolist()  # Return just the indices