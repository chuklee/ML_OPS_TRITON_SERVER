import requests

# Define the API endpoint
BASE_URL = "http://localhost:8000"

# 1. Create a new user
new_user_data = {
    "age": 25,
    "occupation": 4,  # Example: Student
    "gender": "F"
}

# Create new user request
create_user_response = requests.post(
    f"{BASE_URL}/users/new",
    json=new_user_data
)

# Print the response
print("New User Creation Response:")
print(create_user_response.json())

# 2. Get recommendations for the newly created user
if create_user_response.status_code == 200:
    new_user_id = create_user_response.json()["user_id"]
    
    # Get recommendations for the new user
    recommendations_response = requests.post(
        f"{BASE_URL}/recommendations/{new_user_id}",
        params={"batch_size": 20}
    )
    
    print("\nInitial Recommendations for New User:")
    print(recommendations_response.json())
else:
    print("Failed to create new user:", create_user_response.text)