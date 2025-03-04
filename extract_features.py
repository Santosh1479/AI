import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import pickle

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

def build_feature_database(root_folder):
    feature_database = []
    for subdir, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(subdir, filename)
                features = extract_features(image_path)
                feature_database.append((filename, features))
    return feature_database

# Path to the root folder containing the images
root_folder = 'e:\\AI\\Diseases_names'
feature_database_path = 'e:\\AI\\data\\feature_database.pkl'

# Display the root folder containing the images
print(f"Root folder containing the images: {root_folder}")

# Build the feature database
feature_database = build_feature_database(root_folder)

# Save the feature database to a file using pickle
with open(feature_database_path, 'wb') as f:
    pickle.dump(feature_database, f)
print("Feature database built and saved successfully!")