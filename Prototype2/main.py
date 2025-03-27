import numpy as np
from tensorflow import keras
import spacy
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
nltk.download('punkt')

# Load spaCy model (en_core_web_lg with 300D embeddings)
try:
    nlp = spacy.load("en_core_web_lg")
    print(f"Using model with vector shape: {nlp.vocab.vectors.shape}")  # Should be (20000, 300)
except OSError:
    print("Error: 'en_core_web_lg' not found. Run 'python -m spacy download en_core_web_lg' and try again.")
    exit(1)

# Load the saved model
model = keras.models.load_model("diagnosis_model.h5")

# Disease list (must match training)
disease_list = ["Bronchitis", "Common Cold", "Food Poisoning", "Gastroenteritis", "Influenza", 
                "Migraine", "Pneumonia", "Strep Throat", "Tension Headache", "Asthma", 
                "Sinusitis", "Urinary Tract Infection"]

# Initialize spell checker
spell = SpellChecker()

# Symptom correction and embedding
def correct_spelling(user_input):
    tokens = word_tokenize(user_input.lower())
    corrected_tokens = []
    for token in tokens:
        # Check if the token is misspelled
        if spell.unknown([token]):
            corrected = spell.correction(token)
            corrected_tokens.append(corrected if corrected else token)  # Use correction or original if no correction
        else:
            corrected_tokens.append(token)  # Keep correct words as-is
    corrected_input = " ".join(corrected_tokens)
    return corrected_input

def get_symptom_embedding(user_input):
    corrected_input = correct_spelling(user_input)
    doc = nlp(corrected_input)
    return doc.vector  # 300D embedding

# Diagnosis function
def diagnose(user_input):
    embedding = get_symptom_embedding(user_input)
    vector_array = np.array([embedding])  # Shape: (1, 300)
    probabilities = model.predict(vector_array, verbose=0)[0]
    prediction_idx = np.argmax(probabilities)
    prediction = disease_list[prediction_idx]
    prob_dict = {disease: round(float(prob * 100), 2) for disease, prob in zip(disease_list, probabilities)}
    corrected_input = correct_spelling(user_input)  # For display
    return corrected_input, prediction, prob_dict

# Command-line interface
def main():
    print("\nWelcome I'm Doctor Sathi!")
    print("Enter your symptoms (e.g., 'I have a fever and cough') or type 'exit' to quit:")
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            print("Goodbye! Stay Healthy. ;)")
            break
        symptoms, disease, probs = diagnose(user_input)
        if not symptoms.strip():
            print("Sorry! Please provide some symptoms. :(")
        else:
            print(f"Original Input: {user_input}")
            print(f"Corrected Symptoms: {symptoms}")
            print(f"Predicted Disease: {disease}")
            print(f"Probabilities: {probs}")
            print("\nEnter more symptoms or 'exit' to quit:")

if __name__ == "__main__":
    main()