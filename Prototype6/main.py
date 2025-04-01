import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model

# Paths to files (adjust as needed)
MODEL_PATH = "biowordvec_diagnosis_model.keras"  # Updated to .keras
DATA_PATH = "DiseaseAndSymptoms.csv"
EMBEDDING_PATH = r"C:/Users/ACER/Downloads/bio_embedding_extrinsic.bin"

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

def get_symptom_embedding(text, word_vectors):
    """
    Generate the average embedding for a symptom string using BioWordVec embeddings.
    """
    words = text.split()
    vectors = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word_vectors.vector_size)

def predict_disease(symptoms, model, word_vectors, disease_list):
    """
    Predict the disease based on input symptoms.
    """
    # Preprocess symptoms
    processed_symptoms = preprocess_symptoms(symptoms)
    print(f"Processed symptoms: {processed_symptoms}")

    # Generate embedding
    embedding = get_symptom_embedding(processed_symptoms, word_vectors)
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

    # Load the embeddings
    print("Loading BioWordVec embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, binary=True)
    print(f"Loaded embeddings with vector size: {word_vectors.vector_size}")

    # Load the model
    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Predict
    print("\nPredicting disease...")
    try:
        predicted_disease, confidence = predict_disease(args.symptoms, model, word_vectors, disease_list)
        print(f"\nPredicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()