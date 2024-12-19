import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


data = pd.read_csv('clothes_dataset.csv')

data.head()
data.info()
data.describe().T
data.isna().sum()

# Define the features to be used for recommendation
features = ['Brand Name', 'Category', 'Occasion', 'Color Theme', 'Style', 'Color', 'Pattern', 'Season', 'Material', 'Price Range']

# One-hot encode the features
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[features]).toarray()

# Convert encoded features to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(encoded_df)

def get_recommendations(item_index, num_recommendations=5):
    # Get the similarity scores for the given item
    similarity_scores = list(enumerate(similarity_matrix[item_index]))
    
    # Sort the items based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar items
    similar_items_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
    
    # Return the recommended items
    return df.iloc[similar_items_indices]

# Example usage: Get 5 recommendations for the first item
recommended_items = get_recommendations(0, 5)
print(recommended_items)
