from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
from data_pipeline import DataPipeline
from model_serving import ModelServer

app = FastAPI()

# Initialize services
data_pipeline = DataPipeline()
model_server = ModelServer(
    model_path='models/modelRec.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

class UserFeatures(BaseModel):
    age: int
    occupation: int
    gender: str

class RecommendationResponse(BaseModel):
    items: List[Dict]
    next_cursor: str

@app.post("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    cursor: Optional[str] = None,
    batch_size: int = 20
) -> RecommendationResponse:
    try:
        # Get user features
        user_features = data_pipeline.get_user_features(user_id)
        
        # Add gender encoding to user features
        user_features['gen_F'] = 1 if user_features['gender'] == 'F' else 0
        user_features['gen_M'] = 1 if user_features['gender'] == 'M' else 0
        
        # Get user embedding
        user_embedding = model_server.get_user_embedding(user_id, user_features)
        
        # Get recommendations
        recommended_items = model_server.get_recommendations(user_embedding, k=batch_size)
        
        # Enrich recommendations with item details
        enriched_recommendations = []
        for item_id in recommended_items:
            item_features = data_pipeline.get_item_features(item_id)
            enriched_recommendations.append({
                'item_id': item_id,
                'title': item_features['title'],
                'genres': [genre for genre, value in item_features.items() 
                          if value == 1 and genre not in ['item_id', 'title']],
            })
        
        # Generate next cursor (you might want to implement your own cursor logic)
        next_cursor = str(int(cursor or '0') + batch_size)
        
        return RecommendationResponse(
            items=enriched_recommendations,
            next_cursor=next_cursor
        )
    
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/new")
async def create_new_user(user_features: UserFeatures):
    """Endpoint to handle new users"""
    try:
        # Convert gender to one-hot encoding
        gen_F = 1 if user_features.gender == 'F' else 0
        gen_M = 1 if user_features.gender == 'M' else 0
        
        # Create user features dict
        features = {
            'age': user_features.age,
            'occupation': user_features.occupation,
            'gen_F': gen_F,
            'gen_M': gen_M
        }
        
        # Generate initial embedding for the user
        # You might want to implement a better initialization strategy
        user_id = model_server.get_next_user_id()  # You'll need to implement this
        user_embedding = model_server.get_user_embedding(user_id, features)
        
        return {"user_id": user_id, "status": "success"}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)