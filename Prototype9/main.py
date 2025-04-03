import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fuzzywuzzy import fuzz
import re

# Paths to files (adjust as needed)
MODEL_PATH = "biowordvec_diagnosis_model.keras"
DATA_PATH = "DiseaseAndSymptoms.csv"
TOKENIZER_PATH = "symptom_tokenizer.pkl"

# Synonym mapping for common terms
SYMPTOM_SYNONYMS = {
    "loose motion": "diarrhea",
    "tired": "fatigue",
    "fevr": "fever",
    "cof": "cough",
    "continuous": "persistent",
    "continous": "persistent",
    "coughing": "cough",
    "tiredness": "fatigue",
    "loose motions": "diarrhea",
    "feverish": "fever",
    "stomach pain": "abdominal pain",
    "body ache": "muscle pain",
    "body aches": "muscle pain",
    "stomach ache": "abdominal pain",
}

def load_disease_list(data_path):
    """
    Load the list of diseases from the dataset to map predictions back to disease names.
    """
    data = pd.read_csv(data_path)
    data["Disease"] = data["Disease"].replace("Peptic ulcer diseae", "Peptic ulcer disease")
    data["Disease"] = data["Disease"].replace("Dimorphic hemmorhoids(piles)", "Dimorphic hemorrhoids (piles)")
    disease_list = sorted(data["Disease"].unique())
    return disease_list

def load_symptom_list(data_path):
    """
    Load the list of unique symptoms from the dataset for fuzzy matching.
    """
    data = pd.read_csv(data_path)
    data.columns = [col.replace("_", " ") for col in data.columns]
    data = data.apply(lambda x: x.str.replace("_", " ") if x.dtype == "object" else x)
    symptom_cols = [col for col in data.columns if "Symptom" in col]
    data["Symptoms"] = data[symptom_cols].apply(
        lambda row: " ".join(sorted(set([s.strip() for s in row if pd.notna(s)]))), axis=1
    )
    # Extract all unique symptoms
    all_symptoms = set()
    for symptoms in data["Symptoms"]:
        for symptom in symptoms.split():
            all_symptoms.add(symptom)
    return sorted(list(all_symptoms))

def extract_symptoms_from_text(text, known_symptoms):
    """
    Extract symptoms from free text using fuzzy matching, with stopword filtering and synonym mapping.
    """
    # Define stopwords to filter out non-symptom words (all lowercase)
    stopwords = {
        "i", "have", "been", "a", "and", "in", "the", "of", "to", "feel", "has", "with", "on", "at", "for",
        "experiencing", "am", "is", "are", "was", "were", "my", "me", "since", "from", "having", "body"
    }

    # Preprocess the text: lowercase, remove punctuation, split into words
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()

    # Generate bigrams to handle multi-word symptoms (e.g., "loose motion")
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    candidates = words + bigrams

    extracted_symptoms = []
    for candidate in candidates:
        # Skip stopwords for single words
        if len(candidate.split()) == 1 and candidate in stopwords:
            continue

        # Map synonyms if applicable
        candidate = SYMPTOM_SYNONYMS.get(candidate, candidate)

        # Find the best match among known symptoms using fuzzywuzzy
        best_match = None
        best_score = 0
        for symptom in known_symptoms:
            score = fuzz.ratio(candidate, symptom)
            if score > best_score and score > 80:  # Stricter threshold for better matching
                best_score = score
                best_match = symptom
        if best_match:
            extracted_symptoms.append(best_match)

    # Remove duplicates and sort
    extracted_symptoms = sorted(set(extracted_symptoms))
    return " ".join(extracted_symptoms) if extracted_symptoms else None

def predict_disease(symptoms, model, tokenizer, max_len, disease_list, known_symptoms):
    """
    Predict the disease based on input symptoms.
    """
    # Extract symptoms from free text
    processed_symptoms = extract_symptoms_from_text(symptoms, known_symptoms)
    if not processed_symptoms:
        raise ValueError("No valid symptoms detected. Please provide more specific symptoms.")
    print(f"Processed symptoms: {processed_symptoms}")

    # Tokenize and pad the input
    symptom_seq = tokenizer.texts_to_sequences([processed_symptoms])
    symptom_seq = pad_sequences(symptom_seq, maxlen=max_len, padding="post")

    # Predict
    prediction = model.predict(symptom_seq, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = disease_list[predicted_class]
    confidence = prediction[0][predicted_class]

    return predicted_disease, confidence

def main():
    # Set up argument parser for command-line input
    parser = argparse.ArgumentParser(description="Predict disease from free text symptoms using a trained model.")
    parser.add_argument(
        "--symptoms",
        type=str,
        help="Free text describing symptoms (e.g., 'I have a fever and cough')",
        default=None  # Make it optional
    )
    args = parser.parse_args()

    # Load the disease list
    print("Loading disease list...")
    disease_list = load_disease_list(DATA_PATH)
    print(f"Loaded {len(disease_list)} disease classes.")

    # Load the symptom list for fuzzy matching
    print("Loading symptom list...")
    known_symptoms = load_symptom_list(DATA_PATH)
    print(f"Loaded {len(known_symptoms)} unique symptoms.")

    # Load the tokenizer
    print("Loading tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print(f"Loaded tokenizer with vocabulary size: {len(tokenizer.word_index) + 1}")

    # Load the model
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Get max_len from the model input shape
    max_len = model.input_shape[1]  # Shape: (None, max_len)
    print(f"Max sequence length: {max_len}")

    # Iterative prompting loop
    while True:
        # If symptoms are not provided via command line, prompt the user
        if args.symptoms is None:
            print("\nEnter 'quit' to exit.")
            user_input = input("Please enter your symptoms (e.g., 'I have a fever and cough'): ")
            if user_input.lower() == "quit":
                print("Exiting the program. Goodbye!")
                break
            symptoms = user_input
        else:
            symptoms = args.symptoms  # Use command-line input for the first iteration
            args.symptoms = None  # Clear args.symptoms to switch to interactive mode

        # Predict
        print("\nPredicting disease...")
        try:
            predicted_disease, confidence = predict_disease(symptoms, model, tokenizer, max_len, disease_list, known_symptoms)
            print(f"\nPredicted Disease: {predicted_disease}")
            print(f"Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()