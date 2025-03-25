import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Symptom vocabulary
symptom_list = ["fever", "cough", "fatigue", "headache", "nausea"]

# Function to extract symptoms
def extract_symptoms(user_input):
    tokens = word_tokenize(user_input.lower())
    extracted_symptoms = [token for token in tokens if token in symptom_list]
    return extracted_symptoms

# Function to create symptom vector
def symptoms_to_vector(extracted_symptoms, symptom_list):
    return [1 if symptom in extracted_symptoms else 0 for symptom in symptom_list]

# Test the pipeline
user_input = "I have a fever and cough"
symptoms = extract_symptoms(user_input)
vector = symptoms_to_vector(symptoms, symptom_list)

# print(f"Extracted Symptoms: {symptoms}")
# print(f"Symptom Vector: {vector}")
# Output:
# Extracted Symptoms: ['fever', 'cough']
# Symptom Vector: [1, 1, 0, 0, 0]

import numpy as np
from sklearn.linear_model import LogisticRegression

# Toy dataset
X = np.array([
    [1, 1, 1, 0, 0],  # Influenza
    [1, 1, 0, 0, 0],  # Common Cold
    [0, 0, 0, 1, 0],  # Tension Headache
    [0, 0, 1, 1, 1],  # Migraine
    [1, 0, 0, 0, 1],  # Gastroenteritis
    [1, 1, 1, 0, 0],  # Influenza
])
y = ["Influenza", "Common Cold", "Tension Headache", "Migraine", "Gastroenteritis", "Influenza"]

# Train the model
model = LogisticRegression(multi_class="multinomial", max_iter=200)
model.fit(X, y)

# Test prediction
test_symptoms = np.array([vector])  # "fever and cough"
prediction = model.predict(test_symptoms)
probabilities = model.predict_proba(test_symptoms)

print(f"Predicted Disease: {prediction[0]}")
print(f"Probabilities: {dict(zip(model.classes_, probabilities[0]*100))}")
# Output might be:
# Predicted Disease: Common Cold
# Probabilities: {'Common Cold': 0.6, 'Gastroenteritis': 0.05, 'Influenza': 0.3, ...}