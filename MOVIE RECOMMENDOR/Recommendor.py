from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

app = Flask(__name__)

class Recommender:
    def __init__(self, movie_path='data/movies.csv', rating_path='data/ratings.csv'):
        print("Hi world")  # Print "Hi world" at the start
        self.movies = pd.read_csv(movie_path)  # Load the movies CSV into a pandas DataFrame
        self.ratings = pd.read_csv(rating_path)  # Load the ratings CSV into a pandas DataFrame
        self.model = self.build_model()  # Initialize the deep learning model
        self.similarity_matrix = None  # This will store the cosine similarity matrix

    def build_model(self):
        # Simple neural network for collaborative filtering
        num_users = self.ratings['userId'].nunique()
        num_movies = self.ratings['movieId'].nunique()
        latent_dim = 50

        model = nn.Sequential(
            nn.Embedding(num_users, latent_dim),
            nn.Embedding(num_movies, latent_dim),
            nn.Linear(latent_dim, 1)
        )

        return model

    def train_model(self, epochs=10, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Prepare the data for training
        user_ids = torch.tensor(self.ratings['userId'].values - 1)  # Subtract 1 to make userId zero-indexed
        movie_ids = torch.tensor(self.ratings['movieId'].values - 1)  # Subtract 1 to make movieId zero-indexed
        ratings = torch.tensor(self.ratings['rating'].values, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            user_embedding = self.model[0](user_ids)
            movie_embedding = self.model[1](movie_ids)
            preds = (user_embedding * movie_embedding).sum(dim=1)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def collaborative_filtering(self):
        user_movie_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating')
        user_movie_matrix.fillna(0, inplace=True)

        svd = TruncatedSVD(n_components=50, random_state=42)
        latent_matrix = svd.fit_transform(user_movie_matrix)
        self.similarity_matrix = cosine_similarity(latent_matrix)

    def recommend_movies(self, user_id):
        if self.similarity_matrix is None:
            self.collaborative_filtering()
        user_index = user_id - 1  # Assuming userId starts from 1
        user_similarity = self.similarity_matrix[user_index]
        recommended_movie_ids = user_similarity.argsort()[-5:][::-1]  # Top 5 recommendations

        recommended_movies = self.movies[self.movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
        return recommended_movies

    def main(self):
        self.collaborative_filtering()  # Perform collaborative filtering
        self.train_model()  # Train the deep learning model

recommender = Recommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommender.recommend_movies(user_id)
    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    recommender.main()  # Initialize the recommender system and train the model when the server starts
    app.run(debug=True)
