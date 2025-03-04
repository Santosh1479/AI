import os
import pandas as pd
import pickle
from recognize_image import find_similar_images

# Load dataset
dataset = pd.read_csv('e:\\AI\\data\\dataset.csv')

# Fill NaN values with empty strings
dataset = dataset.fillna('')

# Normalize dataset (strip spaces, convert to lowercase)
dataset = dataset.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# Remove duplicate disease entries by aggregating symptoms
disease_symptom_map = {}
for index, row in dataset.iterrows():
    disease_name = row['Disease']
    symptoms = set(filter(None, row[1:].values))  # Remove empty strings
    
    if disease_name in disease_symptom_map:
        disease_symptom_map[disease_name].update(symptoms)
    else:
        disease_symptom_map[disease_name] = symptoms

# Function to find diseases based on symptoms
def find_disease(symptoms):
    symptoms = set(symptoms)  # Convert to set for faster lookup
    matching_diseases = []
    
    for disease, disease_symptoms in disease_symptom_map.items():
        match_count = len(symptoms & disease_symptoms)  # Count common symptoms
        total_symptoms = len(disease_symptoms)
        
        if total_symptoms > 0:
            match_percentage = (match_count / total_symptoms) * 100
        else:
            match_percentage = 0
        
        if match_percentage > 0:
            matching_diseases.append((disease, match_percentage))
    
    # Sort by highest match percentage
    matching_diseases.sort(key=lambda x: x[1], reverse=True)
    
    return matching_diseases

# Path to the root folder containing the images
root_folder = 'e:\\AI\\Diseases_names'

# Path to the input images folder
input_images_folder = 'e:\\AI\\input_image'

# Check if the input images folder exists
if not os.path.exists(input_images_folder):
    print(f"Error: The input images folder '{input_images_folder}' does not exist.")
else:
    # Load the feature database using pickle
    with open('e:\\AI\\data\\feature_database.pkl', 'rb') as f:
        feature_database = pickle.load(f)

    predicted_diseases = []
    highest_similarity_score = 0
    folder_info = {}  # Store max scores and proper folder names
    image_with_100_percent = None

    # Iterate through all images in the input images folder
    for input_filename in os.listdir(input_images_folder):
        if input_filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_image_path = os.path.join(input_images_folder, input_filename)
            
            print(f"\nComparing input image: {input_filename}")
            
            similar_images = find_similar_images(input_image_path, feature_database)
            
            print("\n📌 Matching Images (Based on Feature Similarity):")
            for sim_filename, score in similar_images:
                print(f"- {sim_filename}: Similarity Score = {score:.2f}%")
                print(f"Comparing {input_filename} with {sim_filename}")
                
                # Extract full folder path
                full_path = os.path.dirname(sim_filename)
                folder_name = os.path.basename(full_path)
                
                # Update folder info
                if full_path not in folder_info or score > folder_info[full_path]['max_score']:
                    folder_info[full_path] = {
                        'display_name': folder_name.split('(')[0].strip() if score < 100 else folder_name,
                        'max_score': score
                    }

                # Update highest score tracking
                if score > highest_similarity_score:
                    highest_similarity_score = score
                    highest_similarity_folder = folder_name

                # Check for 100% similarity score
                if score == 100:
                    image_with_100_percent = sim_filename

    # Display the image name with 100% probability
    if image_with_100_percent:
        image_name = os.path.basename(image_with_100_percent)
        image_name = os.path.splitext(image_name)[0]  # Remove file extension
        image_name = ''.join([i for i in image_name if not i.isdigit()])  # Remove numbers
        print(f"\n🔹 Image with 100% Probability: {image_name}")
    else:
        print("\n🔹 No image with 100% probability found")