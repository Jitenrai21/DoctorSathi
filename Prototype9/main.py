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
    Extract symptoms from free text using fuzzy matching.
    """
    # Preprocess the text: lowercase, remove punctuation, split into words
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()

    extracted_symptoms = []
    for word in words:
        # Find the best match among known symptoms using fuzzywuzzy
        best_match = None
        best_score = 0
        for symptom in known_symptoms:
            score = fuzz.ratio(word, symptom)
            if score > best_score and score > 50:  # Loose threshold for matching
                best_score = score
                best_match = symptom
        if best_match:
            extracted_symptoms.append(best_match)

    # Remove duplicates and sort
    extracted_symptoms = sorted(set(extracted_symptoms))
    return " ".join(extracted_symptoms) if extracted_symptoms else None

def predict_disease(symptoms, model, tokenizer, max_len, disease_list):
    """
    Predict the disease based on input symptoms.
    """
    # If no symptoms were extracted, return an error message
    if not symptoms:
        return None, 0.0, "No valid symptoms detected. Please provide more specific symptoms."

    # Preprocess symptoms (already done in extract_symptoms_from_text, but ensure format)
    processed_symptoms = symptoms
    print(f"Processed symptoms: {processed_symptoms}")

    # Tokenize and pad the input
    symptom_seq = tokenizer.texts_to_sequences([processed_symptoms])
    symptom_seq = pad_sequences(symptom_seq, maxlen=max_len, padding="post")

    # Predict
    prediction = model.predict(symptom_seq, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = disease_list[predicted_class]
    confidence = prediction[0][predicted_class]

    return predicted_disease, confidence, None

def main():
    # Set up argument parser for command-line input
    parser = argparse.ArgumentParser(description="Predict disease from free text symptoms using a trained transformer model.")
    parser.add_argument(
        "--text",
        type=str,
        help="Free text describing symptoms (e.g., 'I have a fever and cough')",
        default=None  # Make it optional
    )
    args = parser.parse_args()

    # If text is not provided, prompt the user
    if args.text is None:
        print("No text provided via command line.")
        args.text = input("Please enter your symptoms (e.g., 'I have a fever and cough'): ")

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

    # Extract symptoms from free text
    print("\nExtracting symptoms from text...")
    processed_symptoms = extract_symptoms_from_text(args.text, known_symptoms)

    # Predict
    print("\nPredicting disease...")
    try:
        predicted_disease, confidence, warning = predict_disease(processed_symptoms, model, tokenizer, max_len, disease_list)
        if warning:
            print(warning)
        else:
            print(f"\nPredicted Disease: {predicted_disease}")
            print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()