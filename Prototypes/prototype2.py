import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.ensemble import RandomForestClassifier
nltk.download('punkt')

# Symptom vocabulary
symptom_list = ["fever", "cough", "fatigue", "headache", "nausea"]

# Updated dataset
X = np.array([
    [1, 1, 1, 0, 0],  # Influenza
    [1, 1, 0, 0, 0],  # Common Cold
    [0, 0, 0, 1, 0],  # Tension Headache
    [0, 0, 1, 1, 1],  # Migraine
    [1, 0, 0, 0, 1],  # Gastroenteritis
    [1, 1, 1, 0, 0],  # Influenza
    [0, 1, 0, 0, 0],  # Common Cold
    [1, 0, 1, 0, 0],  # Influenza
    [0, 0, 0, 1, 1],  # Migraine
    [1, 0, 0, 1, 0],  # Gastroenteritis
])
y = ["Influenza", "Common Cold", "Tension Headache", "Migraine", "Gastroenteritis", 
     "Influenza", "Common Cold", "Influenza", "Migraine", "Gastroenteritis"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Symptom extraction
def extract_symptoms(user_input):
    tokens = word_tokenize(user_input.lower())
    extracted_symptoms = [token for token in tokens if token in symptom_list]
    return extracted_symptoms

def symptoms_to_vector(extracted_symptoms, symptom_list):
    return [1 if symptom in extracted_symptoms else 0 for symptom in symptom_list]

# Diagnosis function
def diagnose(user_input):
    symptoms = extract_symptoms(user_input)
    vector = symptoms_to_vector(symptoms, symptom_list)
    vector_array = np.array([vector])
    prediction = model.predict(vector_array)[0]
    probabilities = model.predict_proba(vector_array)[0]
    prob_dict = {str(disease): float(prob) for disease, prob in zip(model.classes_, probabilities)}
    return symptoms, prediction, prob_dict

# Test it
user_input = "I have a fever and cough"
symptoms, disease, probs = diagnose(user_input)
print(f"Extracted Symptoms: {symptoms}")
print(f"Predicted Disease: {disease}")
print(f"Probabilities: {probs}")