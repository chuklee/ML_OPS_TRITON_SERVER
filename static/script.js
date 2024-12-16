document.getElementById('newUserForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const userInput = {
        age: parseInt(document.getElementById('age').value),
        occupation: parseInt(document.getElementById('occupation').value),
        gender: document.getElementById('gender').value
    };
    
    try {
        // Create new user
        const createUserResponse = await fetch('/users/new', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userInput)
        });
        
        const responseData = await createUserResponse.json();
        
        if (responseData.user_id) {
            // Get recommendations
            const recommendationsResponse = await fetch(
                `/recommendations/${responseData.user_id}?batch_size=20`,
                {
                    method: 'POST'
                }
            );
            
            const recommendations = await recommendationsResponse.json();
            
            // Display recommendations
            displayRecommendations(recommendations.items);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
});

function displayRecommendations(movies) {
    const recommendationsSection = document.getElementById('recommendationsSection');
    const recommendationsList = document.getElementById('recommendationsList');
    
    recommendationsSection.style.display = 'block';
    recommendationsList.innerHTML = '';
    
    movies.forEach(movie => {
        const movieCard = document.createElement('div');
        movieCard.className = 'movie-card';
        
        movieCard.innerHTML = `
            <div class="movie-title">${movie.title}</div>
            <div class="movie-genres">${movie.genres.join(', ')}</div>
        `;
        
        recommendationsList.appendChild(movieCard);
    });
}