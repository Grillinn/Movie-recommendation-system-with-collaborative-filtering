import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def create_user_item_matrix(data):
    """
    Create a user-item matrix from the data.
    """
    user_item_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return csr_matrix(user_item_matrix.values), user_item_matrix

def train_model(user_item_matrix):
    """
    Train a KNN model on the user-item matrix.
    """
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    return model