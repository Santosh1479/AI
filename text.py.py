# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import os
# import pickle
# from werkzeug.utils import secure_filename
# from recognize_image import find_similar_images

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# # Load dataset
# dataset = pd.read_csv('C:/Users/Santosh/Desktop/New folder (2)/AI/data/dataset.csv')

# # Fill NaN values with empty strings
# dataset = dataset.fillna('')

# # Normalize dataset (strip spaces, convert to lowercase)
# dataset = dataset.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# # Remove duplicate disease entries by aggregating symptoms
# disease_symptom_map = {}
# for index, row in dataset.iterrows():
#     disease_name = row['Disease']
#     symptoms = set(filter(None, row[1:].values))  # Remove empty strings
    
#     if disease_name in disease_symptom_map:
#         disease_symptom_map[disease_name].update(symptoms)
#     else:
#         disease_symptom_map[disease_name] = symptoms

# # Function to find diseases based on symptoms
# def find_disease(symptoms):
#     symptoms = set(symptoms)  # Convert to set for faster lookup
#     matching_diseases = []
    
#     for disease, disease_symptoms in disease_symptom_map.items():
#         match_count = len(symptoms & disease_symptoms)  # Count common symptoms
#         total_symptoms = len(disease_symptoms)
        
#         if total_symptoms > 0:
#             match_percentage = (match_count / total_symptoms) * 100
#         else:
#             match_percentage = 0
        
#         if match_percentage > 0:
#             matching_diseases.append((disease, match_percentage))
    
#     # Sort by highest match percentage
#     matching_diseases.sort(key=lambda x: x[1], reverse=True)
    
#     return matching_diseases

# @app.route('/diagnosis', methods=['POST'])
# def diagnose():
#     data = request.json
#     user_input = data.get('input', '')
#     user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]
    
#     matching_diseases = find_disease(user_symptoms)
    
#     if matching_diseases:
#         return jsonify({'disease': matching_diseases[0][0]})
#     else:
#         return jsonify({'disease': 'No disease found matching the provided symptoms'})

# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file provided'}), 400
    
#     image = request.files['image']
#     if image.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     filename = secure_filename(image.filename)
#     image_path = os.path.join('C:/Users/Santosh/Desktop/New folder (2)/AI/input_image', filename)
#     image.save(image_path)
    
#     # Call the image matching function
#     result = process_image(image_path)
    
#     return jsonify(result)

# def process_image(image_path):
#     # Load the feature database using pickle
#     with open('e:/AI/data/feature_database.pkl', 'rb') as f:
#         feature_database = pickle.load(f)

#     highest_similarity_score = 0
#     folder_info = {}  # Store max scores and proper folder names
#     image_with_100_percent = None

#     similar_images = find_similar_images(image_path, feature_database)
    
#     for sim_filename, score in similar_images:
#         full_path = os.path.dirname(sim_filename)
#         folder_name = os.path.basename(full_path)
        
#         if full_path not in folder_info or score > folder_info[full_path]['max_score']:
#             folder_info[full_path] = {
#                 'display_name': folder_name.split('(')[0].strip() if score < 100 else folder_name,
#                 'max_score': score
#             }

#         if score > highest_similarity_score:
#             highest_similarity_score = score
#             highest_similarity_folder = folder_name

#         if score == 100:
#             image_with_100_percent = sim_filename

#     if image_with_100_percent:
#         image_name = os.path.basename(image_with_100_percent)
#         image_name = os.path.splitext(image_name)[0]  # Remove file extension
#         image_name = ''.join([i for i in image_name if not i.isdigit()])  # Remove numbers
#         return {'disease': image_name}
#     else:
#         return {'disease': 'No disease found with 100% probability'}

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import pandas as pd

# app = Flask(__name__)

# # Load dataset
# dataset = pd.read_csv('C:/Users/Santosh/Desktop/New folder (2)/AI/dataset.csv')

# # Fill NaN values with empty strings
# dataset = dataset.fillna('')

# # Normalize dataset (strip spaces, convert to lowercase)
# dataset = dataset.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# # Remove duplicate disease entries by aggregating symptoms
# disease_symptom_map = {}
# for index, row in dataset.iterrows():
#     disease_name = row['Disease']
#     symptoms = set(filter(None, row[1:].values))  # Remove empty strings
    
#     if disease_name in disease_symptom_map:
#         disease_symptom_map[disease_name].update(symptoms)
#     else:
#         disease_symptom_map[disease_name] = symptoms

# # Function to find diseases based on symptoms
# def find_disease(symptoms):
#     symptoms = set(symptoms)  # Convert to set for faster lookup
#     matching_diseases = []
    
#     for disease, disease_symptoms in disease_symptom_map.items():
#         match_count = len(symptoms & disease_symptoms)  # Count common symptoms
#         total_symptoms = len(disease_symptoms)
        
#         if total_symptoms > 0:
#             match_percentage = (match_count / total_symptoms) * 100
#         else:
#             match_percentage = 0
        
#         if match_percentage > 0:
#             matching_diseases.append((disease, match_percentage))
    
#     # Sort by highest match percentage
#     matching_diseases.sort(key=lambda x: x[1], reverse=True)
    
#     return matching_diseases

# @app.route('/diagnosis', methods=['POST'])
# def diagnose():
#     data = request.json
#     user_input = data.get('input', '')
#     user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]
    
#     matching_diseases = find_disease(user_symptoms)
    
#     if matching_diseases:
#         return jsonify({'disease': matching_diseases[0][0]})
#     else:
#         return jsonify({'disease': 'No disease found matching the provided symptoms'})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import pickle
from werkzeug.utils import secure_filename
from recognize_image import find_similar_images

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load dataset
dataset = pd.read_csv('C:/Users/Santosh/Desktop/New folder (2)/AI/data/dataset.csv')

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

# Load the feature database once
with open('C:/Users/Santosh/Desktop/New folder (2)/AI/data/feature_database.pkl', 'rb') as f:
    feature_database = pickle.load(f)

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

@app.route('/diagnosis', methods=['POST'])
def diagnose():
    data = request.json
    user_input = data.get('input', '')
    user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]
    
    matching_diseases = find_disease(user_symptoms)
    
    if matching_diseases:
        return jsonify({'disease': matching_diseases[0][0]})
    else:
        return jsonify({'disease': 'No disease found matching the provided symptoms'})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(image.filename)
    image_path = os.path.join('C:/Users/Santosh/Desktop/New folder (2)/AI/input_image', filename)
    image.save(image_path)
    
    # Call the image matching function
    result = process_image(image_path)
    
    return jsonify(result)

def process_image(image_path):
    highest_similarity_score = 0
    folder_info = {}  # Store max scores and proper folder names
    image_with_100_percent = None

    similar_images = find_similar_images(image_path, feature_database)
    
    for sim_filename, score in similar_images:
        full_path = os.path.dirname(sim_filename)
        folder_name = os.path.basename(full_path)
        
        if full_path not in folder_info or score > folder_info[full_path]['max_score']:
            folder_info[full_path] = {
                'display_name': folder_name.split('(')[0].strip() if score < 100 else folder_name,
                'max_score': score
            }

        if score > highest_similarity_score:
            highest_similarity_score = score
            highest_similarity_folder = folder_name

        if score == 100:
            image_with_100_percent = sim_filename

    if image_with_100_percent:
        image_name = os.path.basename(image_with_100_percent)
        image_name = os.path.splitext(image_name)[0]  # Remove file extension
        image_name = ''.join([i for i in image_name if not i.isdigit()])  # Remove numbers
        return {'disease': image_name}
    else:
        return {'disease': 'No disease found with 100% probability'}

if __name__ == '__main__':
    app.run(debug=True)