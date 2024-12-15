import requests

# Get recommendations for a user
response = requests.post(
    "http://localhost:8000/recommendations/1",
    params={"batch_size": 20}
)
recommendations = response.json()
print(recommendations)