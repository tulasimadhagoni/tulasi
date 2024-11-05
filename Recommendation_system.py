import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
    'item_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'D'],
    'rating': [5, 3, 4, 4, 5, 2, 3, 5, 4]
}

df = pd.DataFrame(data)

user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_recommendations(user_id, num_recommendations=2):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    user_ratings = user_item_matrix.loc[similar_users].sum().sort_values(ascending=False)
    
    known_items = user_item_matrix.loc[user_id]
    recommendations = user_ratings[known_items == 0]
    
    return recommendations.head(num_recommendations)

recommendations = get_recommendations(user_id=1, num_recommendations=2)
print("Recommendations for user 1:")
print(recommendations)