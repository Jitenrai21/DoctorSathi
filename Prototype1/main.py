import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow import keras
nltk.download('punkt')

# Load the saved model
model = keras.models.load_model("diagnosis_model.h5")

# Symptom vocabulary (must match training script)
symptom_list = ["fever", "cough", "fatigue", "headache", "nausea", "sore throat", "chills", "shortness of breath"]
disease_list = ["Influenza", "Common Cold", "Tension Headache", "Migraine", "Gastroenteritis", 
                "Pneumonia", "Food Poisoning", "Strep Throat", "Bronchitis", "Sinus Infection"]

# Symptom extraction
def extract_symptoms(user_input):
    tokens = word_tokenize(user_input.lower())
    extracted_symptoms = []
    
    # Check for multi-word symptoms first
    i = 0
    
    # Find the maximum symptom length in symptom_list
    max_symptom_length = max(len(symptom.split()) for symptom in symptom_list)  # e.g., 3 for "shortness of breath"

    while i < len(tokens):
        found = False
        # Check phrases from max length down to 1
        for length in range(max_symptom_length, 0, -1):
            if i + length <= len(tokens):
                phrase = " ".join(tokens[i:i + length])
                if phrase in symptom_list:
                    extracted_symptoms.append(phrase)
                    i += length  # Skip ahead by the length of the matched phrase
                    found = True
                    break
        if not found:
            i += 1  # Move to next token if no match
    
    return extracted_symptoms

def symptoms_to_vector(extracted_symptoms, symptom_list):
    return [1 if symptom in extracted_symptoms else 0 for symptom in symptom_list]

# Diagnosis function with clean output
def diagnose(user_input):
    symptoms = extract_symptoms(user_input)
    if not symptoms:
        return "Sorry! Provided Symtoms are not recognized. :(", None, None
    vector = symptoms_to_vector(symptoms, symptom_list)
    vector_array = np.array([vector])
    probabilities = model.predict(vector_array, verbose=0)[0]
    prediction_idx = np.argmax(probabilities)
    prediction = disease_list[prediction_idx]
    prob_dict = {disease: round(float(prob*100), 2) for disease, prob in zip(disease_list, probabilities)}
    return symptoms, prediction, prob_dict

# Simple command-line interface
def main():
    print("\nWelcome I'm Doctor Sathi!")
    print("Enter your symptoms (e.g., 'fever and cough') or type 'exit' to quit:")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            print("Goodbye! Stay Healthy. ;)")
            break
        symptoms, disease, probs = diagnose(user_input)
        if disease is None:
            print(symptoms)
        else:
            print(f"Extracted Symptoms: {symptoms}")
            print(f"Predicted Disease: {disease}")
            print(f"Probabilities: {probs}")
            print("\nEnter more symptoms or 'exit' to quit:")

if __name__ == "__main__":
    main()