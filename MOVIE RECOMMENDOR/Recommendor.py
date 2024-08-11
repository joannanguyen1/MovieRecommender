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
    
    def build_deep_learning_model(self, n_users, n_items):
        class RecommenderModel(nn.Module):
            def __init__(self, n_users, n_items, embedding_dim=50):
                super(RecommenderModel, self).__init__()
                self.user_embedding = nn.Embedding(n_users, embedding_dim)
                self.item_embedding = nn.Embedding(n_items, embedding_dim)
                self.fc1 = nn.Linear(embedding_dim * 2, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)

            def forward(self, user_input, item_input):
                user_embedded = self.user_embedding(user_input)
                item_embedded = self.item_embedding(item_input)
                x = torch.cat([user_embedded, item_embedded], dim=-1)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        self.model = RecommenderModel(n_users, n_items)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def train_deep_learning_model(self, train_users, train_items, train_ratings, epochs=10, batch_size=64):
        if self.model is None:
            n_users = len(self.ratings['userId'].unique())
            n_items = len(self.ratings['movieId'].unique())
            self.build_deep_learning_model(n_users, n_items)

        train_users = torch.LongTensor(train_users)
        train_items = torch.LongTensor(train_items)
        train_ratings = torch.FloatTensor(train_ratings)

        dataset = torch.utils.data.TensorDataset(train_users, train_items, train_ratings)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items).squeeze()
                loss = self.criterion(predictions, batch_ratings)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}')

    def predict_rating(self, user_id, movie_id):
        user_tensor = torch.LongTensor([user_id])
        item_tensor = torch.LongTensor([movie_id])
        with torch.no_grad():
            self.model.eval()
            prediction = self.model(user_tensor, item_tensor).item()
        return prediction
