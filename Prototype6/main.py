import argparse
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths to files (adjust as needed)
MODEL_PATH = "biowordvec_diagnosis_model.h5"
DATA_PATH = "DiseaseAndSymptoms.csv"
EMBEDDING_DICT_PATH = "symptom_embeddings.npy"

def load_disease_list(data_path):
    """
    Load the list of diseases from the dataset to map predictions back to disease names.
    """
    data = pd.read_csv(data_path)
    data["Disease"] = data["Disease"].replace("Peptic ulcer diseae", "Peptic ulcer disease")
    data["Disease"] = data["Disease"].replace("Dimorphic hemmorhoids(piles)", "Dimorphic hemorrhoids (piles)")
    disease_list = sorted(data["Disease"].unique())
    return disease_list

def preprocess_symptoms(symptoms):
    """
    Preprocess symptoms the same way as in the notebook:
    - Removes special characters, trims whitespace.
    - Splits by commas, removes duplicates, sorts, and joins them back.
    """
    # Remove unwanted special characters (except commas for splitting)
    symptoms = re.sub(r"[^a-zA-Z0-9, ]", "", symptoms)
    
    # Split symptoms, strip spaces, remove duplicates, sort
    symptom_list = sorted(set(s.strip().lower() for s in symptoms.split(",") if s.strip()))

    return " ".join(symptom_list)

def load_precomputed_embeddings(embedding_dict_path):
    """
    Load the precomputed symptom embeddings.
    Returns a dictionary with symptoms and their embeddings.
    """
    data = np.load(embedding_dict_path, allow_pickle=True).item()
    return data["symptoms"], data["embeddings"]

def find_closest_symptom_embedding(input_symptoms, precomputed_symptoms, precomputed_embeddings):
    """
    Find the embedding for the input symptoms by matching with precomputed symptoms.
    Uses exact string matching for simplicity.
    """
    processed_symptoms = preprocess_symptoms(input_symptoms)
    print(f"Processed symptoms: {processed_symptoms}")

    # Look for an exact match
    try:
        idx = precomputed_symptoms.index(processed_symptoms)
        embedding = precomputed_embeddings[idx]
        return embedding
    except ValueError:
        raise ValueError(f"No exact match found for symptoms: {processed_symptoms}. Try a different combination.")

def predict_disease(symptoms, model, precomputed_symptoms, precomputed_embeddings, disease_list):
    """
    Predict the disease based on input symptoms using precomputed embeddings.
    """
    embedding = find_closest_symptom_embedding(symptoms, precomputed_symptoms, precomputed_embeddings)
    embedding = np.expand_dims(embedding, axis=0)  # Shape: (1, 200)

    # Predict
    prediction = model.predict(embedding, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = disease_list[predicted_class]
    confidence = prediction[0][predicted_class]

    return predicted_disease, confidence

def main():
    # Set up argument parser for command-line input
    parser = argparse.ArgumentParser(description="Predict disease from symptoms using a trained BioWordVec model.")
    parser.add_argument(
        "--symptoms",
        type=str,
        help="Comma-separated list of symptoms (e.g., 'fever, cough, fatigue')",
        default=None  # Make it optional
    )
    args = parser.parse_args()

    # If symptoms are not provided, prompt the user
    if args.symptoms is None:
        print("No symptoms provided via command line.")
        args.symptoms = input("Please enter symptoms (comma-separated, e.g., 'fever, cough, fatigue'): ")

    # Load the disease list
    print("Loading disease list...")
    disease_list = load_disease_list(DATA_PATH)
    print(f"Loaded {len(disease_list)} disease classes.")

    # Load the precomputed embeddings
    print("Loading precomputed embeddings...")
    precomputed_symptoms, precomputed_embeddings = load_precomputed_embeddings(EMBEDDING_DICT_PATH)
    print(f"Loaded precomputed embeddings with {len(precomputed_symptoms)} samples.")

    # Load the model
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Predict
    print("\nPredicting disease...")
    try:
        predicted_disease, confidence = predict_disease(
            args.symptoms, model, precomputed_symptoms, precomputed_embeddings, disease_list
        )
        print(f"\nPredicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()