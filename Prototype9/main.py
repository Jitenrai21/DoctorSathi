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
    "head ache": "headache",
    "nauseous": "nausea",
    "belly ache": "abdominal pain",
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
    Returns both the full list of known symptoms and a set of single-word symptoms.
    """
    data = pd.read_csv(data_path)
    data.columns = [col.replace("_", " ") for col in data.columns]
    data = data.apply(lambda x: x.str.replace("_", " ") if x.dtype == "object" else x)
    symptom_cols = [col for col in data.columns if "Symptom" in col]
    data["Symptoms"] = data[symptom_cols].apply(
        lambda row: " ".join(sorted(set([s.strip() for s in row if pd.notna(s)]))), axis=1
    )
    # Extract all unique symptoms, including multi-word symptoms
    all_symptoms = set()
    single_word_symptoms = set()
    for symptoms in data["Symptoms"]:
        # Add the full symptom string as a multi-word symptom
        all_symptoms.add(symptoms)
        # Add individual words as single-word symptoms
        for symptom in symptoms.split():
            single_word_symptoms.add(symptom)
    # Add all synonym values to ensure mapped terms are recognized
    all_symptoms.update(SYMPTOM_SYNONYMS.values())
    single_word_symptoms.update(SYMPTOM_SYNONYMS.values())
    known_symptoms = sorted(list(all_symptoms))
    single_word_symptoms = sorted(list(single_word_symptoms))
    return known_symptoms, single_word_symptoms

def extract_symptoms_from_text(text, known_symptoms, single_word_symptoms):
    """
    Extract symptoms from free text using fuzzy matching, with stopword filtering and synonym mapping.
    """
    # Define stopwords to filter out non-symptom words (all lowercase)
    stopwords = {
        "i", "have", "been", "a", "and", "in", "the", "of", "to", "feel", "has", "with", "on", "at", "for",
        "experiencing", "am", "is", "are", "was", "were", "my", "me", "since", "from", "having", "body",
        "feeling", "because", "went", "doctor", "bad", "yesterday"
    }

    # Preprocess the text: lowercase, split into words
    text = text.lower()
    # Split by spaces and clean each word (keep alphabetic parts)
    words = []
    for word in text.split():
        # Extract the alphabetic part (e.g., fever123 → fever)
        alpha_part = "".join([char for char in word if char.isalpha()])
        if alpha_part and alpha_part not in stopwords:  # Filter stopwords early
            words.append(alpha_part)

    # Generate bigrams to handle multi-word symptoms (e.g., "loose motion")
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    candidates = bigrams + words  # Prioritize bigrams over single words

    extracted_symptoms = set()  # Use a set to avoid duplicates early
    used_indices = set()  # Track indices of words already used in bigrams

    # First, try to match bigrams (multi-word symptoms)
    for i, candidate in enumerate(candidates):
        if candidate in bigrams:
            start_idx = i
            # Skip if any word in this bigram has already been used
            if start_idx in used_indices or (start_idx + 1) in used_indices:
                continue

            # Check if the full bigram matches a synonym
            if candidate in SYMPTOM_SYNONYMS:
                mapped_candidate = SYMPTOM_SYNONYMS[candidate]
                extracted_symptoms.add(mapped_candidate)
                used_indices.add(start_idx)
                used_indices.add(start_idx + 1)
                continue

            # If the bigram doesn't match, split into individual terms
            bigram_terms = candidate.split()
            mapped_terms = []
            for term in bigram_terms:
                # Map each term to its synonym
                mapped_term = SYMPTOM_SYNONYMS.get(term, term)
                mapped_terms.append(mapped_term)

            # If the mapped terms form a known multi-word symptom
            full_mapped_candidate = " ".join(mapped_terms)
            if full_mapped_candidate in SYMPTOM_SYNONYMS.values():
                extracted_symptoms.add(full_mapped_candidate)
                used_indices.add(start_idx)
                used_indices.add(start_idx + 1)
                continue

            # Otherwise, add the individual mapped terms if they are valid standalone symptoms
            for mapped_term in mapped_terms:
                # Only add if the term is a valid standalone symptom
                if mapped_term in SYMPTOM_SYNONYMS.values() or mapped_term in single_word_symptoms:
                    extracted_symptoms.add(mapped_term)
                else:
                    # Try fuzzy matching for the mapped term
                    best_match = None
                    best_score = 0
                    for symptom in single_word_symptoms:  # Use single-word symptoms for fuzzy matching
                        score = fuzz.ratio(mapped_term, symptom)
                        if score > best_score and score > 30:  # Threshold for fuzzy matching
                            best_score = score
                            best_match = symptom
                    if best_match:
                        extracted_symptoms.add(mapped_term)

            used_indices.add(start_idx)
            used_indices.add(start_idx + 1)

    # Then, match single words that haven't been used in bigrams
    for i, candidate in enumerate(words):
        if i in used_indices:
            continue

        # Map synonyms if applicable
        mapped_candidate = SYMPTOM_SYNONYMS.get(candidate, candidate)

        # Only add if the term is a valid standalone symptom
        if mapped_candidate in SYMPTOM_SYNONYMS.values() or mapped_candidate in single_word_symptoms:
            extracted_symptoms.add(mapped_candidate)
        else:
            # Try fuzzy matching
            best_match = None
            best_score = 0
            for symptom in single_word_symptoms:  # Use single-word symptoms for fuzzy matching
                score = fuzz.ratio(mapped_candidate, symptom)
                if score > best_score and score > 30:  # Threshold for fuzzy matching
                    best_score = score
                    best_match = symptom
            if best_match:
                extracted_symptoms.add(mapped_candidate)

    # Sort the final symptoms
    final_symptoms = sorted(extracted_symptoms)
    return " ".join(final_symptoms) if final_symptoms else None

def predict_disease(symptoms, model, tokenizer, max_len, disease_list, known_symptoms, single_word_symptoms):
    """
    Predict the disease based on input symptoms.
    """
    # Extract symptoms from free text
    processed_symptoms = extract_symptoms_from_text(symptoms, known_symptoms, single_word_symptoms)
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
    disease_list = load_disease_list(DATA_PATH)

    # Load the symptom list for fuzzy matching
    known_symptoms, single_word_symptoms = load_symptom_list(DATA_PATH)

    # Load the tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Load the model
    model = load_model(MODEL_PATH)

    # Get max_len from the model input shape
    max_len = model.input_shape[1]  # Shape: (None, max_len)

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
            predicted_disease, confidence = predict_disease(
                symptoms, model, tokenizer, max_len, disease_list, known_symptoms, single_word_symptoms
            )
            print(f"\nPredicted Disease: {predicted_disease}")
            print(f"Confidence: {confidence:.4f}")
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()