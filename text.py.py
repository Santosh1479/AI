import pandas as pd

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

# Prompt user for symptoms input
user_input = input("\nEnter the symptoms separated by commas: ").strip().lower()
user_symptoms = [symptom.strip() for symptom in user_input.split(',') if symptom.strip()]  # Remove empty inputs

# Find matching diseases
matching_diseases = find_disease(user_symptoms)

if matching_diseases:
    print(f"\n🔹 Disease Found: **{matching_diseases[0][0]}**")  # Best match
    print("\nMatching Diseases (Based on Symptom Match %):")
    for disease, score in matching_diseases[:5]:  # Show top 5 matches
        print(f"- {disease}: {score:.2f}% match")
else:
    print("\n No disease found matching the provided symptoms.")
