import numpy as np

def get_movie_recommendations(model, user_item_matrix, user_id, n_recommendations=10):
    """
    Get movie recommendations for a given user.
    """
    user_index = user_id - 1  # Assuming user IDs start from 1
    distances, indices = model.kneighbors(user_item_matrix[user_index], n_neighbors=n_recommendations+1)
    recommendations = []
    for i in range(1, len(distances.flatten())):
        movie_index = indices.flatten()[i]
        recommendations.append(user_item_matrix.columns[movie_index])
    return recommendations