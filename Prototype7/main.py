import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

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

def preprocess_symptoms(symptoms):
    """
    Preprocess symptoms:
    - Split by commas, strip whitespace, remove duplicates, sort, join with spaces.
    """
    symptom_list = sorted(set(s.strip() for s in symptoms.split(",")))
    return " ".join(symptom_list)

def predict_disease(symptoms, model, tokenizer, max_len, disease_list):
    """
    Predict the disease based on input symptoms.
    """
    # Preprocess symptoms
    processed_symptoms = preprocess_symptoms(symptoms)
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
    parser = argparse.ArgumentParser(description="Predict disease from symptoms using a trained model with embedding layer.")
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

    # Predict
    print("\nPredicting disease...")
    try:
        predicted_disease, confidence = predict_disease(args.symptoms, model, tokenizer, max_len, disease_list)
        print(f"\nPredicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()