import pandas as pd
import pytesseract
from PIL import Image
import csv

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR'

# Function to extract text from image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to convert text to CSV
def convert_text_to_csv(text, csv_path):
    lines = text.split('\n')
    symptoms = [line.strip() for line in lines if line.strip()]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Symptom'])
        for symptom in symptoms:
            writer.writerow([symptom])

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

# Display sample symptoms
print("\nSample symptoms from dataset:")
sample_symptoms = list(set().union(*disease_symptom_map.values()))  # Get all unique symptoms
print(", ".join(sample_symptoms[:10]))  # Show first 10 symptoms

# Path to the input image
image_path = 'e:\\AI\\data\\symptoms_image\\Tinea_corporis-1296x728-slide3.jpg'

# Extract text from image
extracted_text = extract_text_from_image(image_path)
print("\nExtracted Text from Image:")
print(extracted_text)

# Convert extracted text to CSV
csv_path = 'e:\\AI\\data\\extracted_symptoms.csv'
convert_text_to_csv(extracted_text, csv_path)

# Load extracted symptoms from CSV
extracted_symptoms_df = pd.read_csv(csv_path)
user_symptoms = extracted_symptoms_df['Symptom'].tolist()

# Find matching diseases
matching_diseases = find_disease(user_symptoms)

if matching_diseases:
    print(f"\n🔹 Disease Found: **{matching_diseases[0][0]}**")  # Best match
    print("\n📌 Matching Diseases (Based on Symptom Match %):")
    for disease, score in matching_diseases[:5]:  # Show top 5 matches
        print(f"- {disease}: {score:.2f}% match")
else:
    print("\n❌ No disease found matching the provided symptoms.")