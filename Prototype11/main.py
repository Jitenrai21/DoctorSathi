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
MODEL_PATH = "diagnosis_model.keras"  # Updated model path
DATA_PATH = "DiseaseAndSymptoms.csv"
TOKENIZER_PATH = "symptom_tokenizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

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
    all_symptoms = set()
    single_word_symptoms = set()
    for symptoms in data["Symptoms"]:
        all_symptoms.add(symptoms)
        for symptom in symptoms.split():
            single_word_symptoms.add(symptom)
    all_symptoms.update(SYMPTOM_SYNONYMS.values())
    single_word_symptoms.update(SYMPTOM_SYNONYMS.values())
    known_symptoms = sorted(list(all_symptoms))
    single_word_symptoms = sorted(list(single_word_symptoms))
    return known_symptoms, single_word_symptoms

def extract_symptoms_from_text(text, known_symptoms, single_word_symptoms):
    """
    Extract symptoms from free text using fuzzy matching, with stopword filtering and synonym mapping.
    """
    stopwords = {
        "i", "have", "been", "a", "and", "in", "the", "of", "to", "feel", "has", "with", "on", "at", "for",
        "experiencing", "am", "is", "are", "was", "were", "my", "me", "since", "from", "having", "body",
        "feeling", "because", "went", "doctor", "bad", "yesterday"
    }

    text = text.lower()
    words = []
    for word in text.split():
        alpha_part = "".join([char for char in word if char.isalpha()])
        if alpha_part and alpha_part not in stopwords:
            words.append(alpha_part)

    bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    candidates = bigrams + words

    extracted_symptoms = set()
    used_indices = set()

    for i, candidate in enumerate(candidates):
        if candidate in bigrams:
            start_idx = i
            if start_idx in used_indices or (start_idx + 1) in used_indices:
                continue

            if candidate in SYMPTOM_SYNONYMS:
                mapped_candidate = SYMPTOM_SYNONYMS[candidate]
                extracted_symptoms.add(mapped_candidate)
                used_indices.add(start_idx)
                used_indices.add(start_idx + 1)
                continue

            bigram_terms = candidate.split()
            mapped_terms = []
            for term in bigram_terms:
                mapped_term = SYMPTOM_SYNONYMS.get(term, term)
                mapped_terms.append(mapped_term)

            full_mapped_candidate = " ".join(mapped_terms)
            if full_mapped_candidate in SYMPTOM_SYNONYMS.values():
                extracted_symptoms.add(full_mapped_candidate)
                used_indices.add(start_idx)
                used_indices.add(start_idx + 1)
                continue

            for mapped_term in mapped_terms:
                if mapped_term in SYMPTOM_SYNONYMS.values() or mapped_term in single_word_symptoms:
                    extracted_symptoms.add(mapped_term)
                else:
                    best_match = None
                    best_score = 0
                    for symptom in single_word_symptoms:
                        score = fuzz.ratio(mapped_term, symptom)
                        if score > best_score and score > 30:
                            best_score = score
                            best_match = symptom
                    if best_match:
                        extracted_symptoms.add(mapped_term)

            used_indices.add(start_idx)
            used_indices.add(start_idx + 1)

    for i, candidate in enumerate(words):
        if i in used_indices:
            continue

        mapped_candidate = SYMPTOM_SYNONYMS.get(candidate, candidate)
        if mapped_candidate in SYMPTOM_SYNONYMS.values() or mapped_candidate in single_word_symptoms:
            extracted_symptoms.add(mapped_candidate)
        else:
            best_match = None
            best_score = 0
            for symptom in single_word_symptoms:
                score = fuzz.ratio(mapped_candidate, symptom)
                if score > best_score and score > 30:
                    best_score = score
                    best_match = symptom
            if best_match:
                extracted_symptoms.add(mapped_candidate)

    final_symptoms = sorted(extracted_symptoms)
    return " ".join(final_symptoms) if final_symptoms else None

def predict_disease(symptoms, model, tokenizer, max_len, disease_list, known_symptoms, single_word_symptoms):
    """
    Predict the disease based on input symptoms.
    """
    processed_symptoms = extract_symptoms_from_text(symptoms, known_symptoms, single_word_symptoms)
    if not processed_symptoms:
        raise ValueError("No valid symptoms detected. Please provide more specific symptoms.")
    print(f"Processed symptoms: {processed_symptoms}")

    symptom_seq = tokenizer.texts_to_sequences([processed_symptoms])
    symptom_seq = pad_sequences(symptom_seq, maxlen=max_len, padding="post")

    prediction = model.predict(symptom_seq, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = disease_list[predicted_class]
    confidence = prediction[0][predicted_class]

    return predicted_disease, confidence

def main():
    parser = argparse.ArgumentParser(description="Predict disease from free text symptoms using a trained model.")
    parser.add_argument(
        "--symptoms",
        type=str,
        help="Free text describing symptoms (e.g., 'I have a fever and cough')",
        default=None
    )
    args = parser.parse_args()

    disease_list = load_disease_list(DATA_PATH)
    known_symptoms, single_word_symptoms = load_symptom_list(DATA_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    disease_list = list(label_encoder.classes_)

    model = load_model(MODEL_PATH)
    max_len = model.input_shape[1]

    while True:
        if args.symptoms is None:
            print("\nEnter 'quit' to exit.")
            user_input = input("Please enter your symptoms (e.g., 'I have a fever and cough'): ")
            if user_input.lower() == "quit":
                print("Exiting the program. Goodbye!")
                break
            symptoms = user_input
        else:
            symptoms = args.symptoms
            args.symptoms = None

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