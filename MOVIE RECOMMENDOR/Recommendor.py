import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.optim as optim

class Recommender:
    def __init__(self, movie_path='data/movies.csv', rating_path='data/ratings.csv'):
        self.movies = pd.read_csv(movie_path) # puts the movies csv into pandas data frame
        self.ratings = pd.read_csv(rating_path)# puts ratings csv into pandas data fram
        self.model = None #intiate value to none for later/ will store deep learning 
        self.similarity_matrix = None # This is where the cosine similarity matrix will gp

    def collaborative_filtering(self):
        user_movie_matrix = self.ratings.pivot_table(index='userId', columns='movieId', values='rating') # creates pivot table where the rows is UserUd columns MovieIDs and values as rating 
        user_movie_matrix.fillna(0, inplace=True) # fills all empty with zero 

        svd = TruncatedSVD(n_components=50, random_state=42) # instalizles the SVD model
        latent_matrix = svd.fit_transform(user_movie_matrix) 
        self.similarity_matrix = cosine_similarity(latent_matrix)

    def get_recommendations(self, user_id, top_n=10):
        if self.similarity_matrix is None:
            self.collaborative_filtering()

        similar_users = self.similarity_matrix[user_id - 1]
        similar_users_indices = similar_users.argsort()[::-1][:top_n]

        recommended_movies = []
        for idx in similar_users_indices:
            similar_user_movies = self.ratings[self.ratings['userId'] == idx + 1]
            recommended_movies.extend(similar_user_movies['movieId'].tolist())

        return list(set(recommended_movies))
    