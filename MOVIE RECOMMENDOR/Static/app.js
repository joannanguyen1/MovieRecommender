/* script.js */

document.addEventListener('DOMContentLoaded', function () {
    const recommendButtons = document.querySelectorAll('.recommend-btn');
    
    recommendButtons.forEach(button => {
        button.addEventListener('click', function () {
            const userId = this.getAttribute('data-user-id');
            getRecommendations(userId);
        });
    });
});

function getRecommendations(userId) {
    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `user_id=${userId}`
    })
    .then(response => response.json())
    .then(data => {
        alert(`Recommendations for User ${userId}: ${data.recommendations.join(', ')}`);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
