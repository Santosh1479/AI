import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from extract_features import extract_features

def find_similar_images(input_image_path, feature_database, top_n=5):
    input_features = extract_features(input_image_path)
    similarity_scores = []
    for filename, features in feature_database:
        similarity = cosine_similarity([input_features], [features])[0][0]
        similarity_scores.append((filename, similarity))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Normalize similarity scores to be between 1 and 100
    max_score = similarity_scores[0][1] if similarity_scores else 1
    similarity_scores = [(filename, (score / max_score) * 100) for filename, score in similarity_scores]
    
    return similarity_scores[:top_n]

# Load the feature database using pickle
with open('e:\\AI\\data\\feature_database.pkl', 'rb') as f:
    feature_database = pickle.load(f)

# Path to the input image
input_image_path = 'e:\\AI\\input_image'

# Iterate through all images in the input images folder
for input_filename in os.listdir(input_image_path):
    if input_filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_image_full_path = os.path.join(input_image_path, input_filename)
        
        # Find similar images
        similar_images = find_similar_images(input_image_full_path, feature_database)
        
        # Display the matching images and their corresponding similarity scores
        print(f"\n📌 Matching Images for {input_filename} (Based on Feature Similarity):")
        for filename, score in similar_images:
            print(f"- {filename}: Similarity Score = {score:.2f}%")