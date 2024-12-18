document.getElementById('trainButton').addEventListener('click', async () => {
    const trainButton = document.getElementById('trainButton');
    const trainingStatus = document.getElementById('trainingStatus');
    const progressFill = trainingStatus.querySelector('.progress-fill');
    
    try {
        // Disable button and show loading state
        trainButton.disabled = true;
        trainingStatus.style.display = 'block';
        progressFill.style.width = '0%';
        
        // Start the training process
        const response = await fetch('/model/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                epochs: 2,
                batch_size: 512,
                learning_rate: 0.0001,
                user_emb_size: 32,
                item_emb_size: 32
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'training_started') {
            // Start polling for training status
            const statusCheckInterval = setInterval(async () => {
                try {
                    const statusResponse = await fetch('/model/training-status');
                    
                    if (!statusResponse.ok) {
                        throw new Error(`HTTP error! status: ${statusResponse.status}`);
                    }
                    
                    const statusData = await statusResponse.json();
                    
                    // Update the status display
                    updateTrainingStatus(statusData);
                    
                    // If training is complete or failed, stop polling
                    if (!statusData.is_training) {
                        clearInterval(statusCheckInterval);
                        trainButton.disabled = false;
                        
                        if (statusData.status === 'completed') {
                            alert(`Training completed successfully!\nValidation Accuracy: ${statusData.metrics.final_val_accuracy.toFixed(4)}`);
                        } else if (statusData.status === 'failed') {
                            alert(`Training failed: ${statusData.error}`);
                            progressFill.style.backgroundColor = '#dc3545';
                        }
                    }
                } catch (error) {
                    console.error('Status check error:', error);
                    // Don't stop polling on temporary errors
                    if (error.message.includes('Failed to fetch')) {
                        console.log('Connection error, will retry...');
                    } else {
                        clearInterval(statusCheckInterval);
                        trainButton.disabled = false;
                        alert('Error checking training status. Please check the console for details.');
                    }
                }
            }, 1000); // Poll every second instead of every 2 seconds
        }
    } catch (error) {
        console.error('Training error:', error);
        alert('Error starting training. Please check the console for details.');
        trainButton.disabled = false;
        trainingStatus.style.display = 'none';
    }
});

function updateTrainingStatus(status) {
    const trainingStatus = document.getElementById('trainingStatus');
    const statusText = trainingStatus.querySelector('p');
    const progressFill = trainingStatus.querySelector('.progress-fill');
    
    try {
        // Update status message
        statusText.textContent = `Training status: ${status.status} (${status.progress}%)`;
        
        // Update progress bar with animation
        requestAnimationFrame(() => {
            progressFill.style.width = `${status.progress}%`;
            
            // Add different colors based on status
            if (status.status === 'failed') {
                progressFill.style.backgroundColor = '#dc3545'; // red for error
            } else if (status.status === 'completed') {
                progressFill.style.backgroundColor = '#28a745'; // green for completion
            } else {
                progressFill.style.backgroundColor = '#007bff'; // blue for in progress
            }
        });
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

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