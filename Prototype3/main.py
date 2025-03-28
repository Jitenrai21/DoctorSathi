import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Layer
from fuzzywuzzy import process, fuzz
import nltk
import re
nltk.download('punkt')

# Define BioBERTLayer with serialization support
class BioBERTLayer(Layer):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.biobert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # Return pooler_output (CLS token)

    def get_config(self):
        config = super(BioBERTLayer, self).get_config()
        config.update({"model_name": self.model_name})
        return config

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load the trained model and disease list
model = tf.keras.models.load_model("biobert_diagnosis_model.h5", custom_objects={"BioBERTLayer": BioBERTLayer})
disease_list = np.load("disease_list.npy", allow_pickle=True).tolist()

# Load dataset to extract unique symptoms
data = pd.read_csv("DiseaseAndSymptoms.csv")
data.columns = [col.replace("_", " ") for col in data.columns]
symptom_cols = [col for col in data.columns if "Symptom" in col]
all_symptoms = data[symptom_cols].melt().dropna()["value"].unique().tolist()
all_symptoms = [s.replace("_", " ") for s in all_symptoms]  # Normalize symptom names
print(f"Available symptoms: {all_symptoms[:10]}... (total: {len(all_symptoms)})")

# Define prescriptions
prescriptions = {
    "Fungal infection": "Apply antifungal cream, keep area dry, consult a doctor if persistent.",
    "Allergy": "Avoid allergens, take antihistamines, seek medical advice if severe.",
    "GERD": "Avoid spicy/acidic foods, take antacids, consult a doctor.",
    "Heart attack": "Call emergency services immediately.",
    "Pneumonia": "Seek immediate medical attention, rest, and follow a doctor’s antibiotic treatment.",
    "Common Cold": "Rest, stay hydrated, take over-the-counter cold remedies.",
    "Hepatitis B": "Consult a doctor for antiviral treatment and monitoring.",
    "Hyperthyroidism": "Consult a doctor for thyroid function tests and treatment."
}

def match_symptoms(user_input, symptom_list, threshold=80):
    """Match user input to known symptoms using fuzzywuzzy."""
    # Clean input: remove punctuation, keep words
    cleaned_input = re.sub(r'[^\w\s]', '', user_input.lower())  # Remove all punctuation
    tokens = nltk.word_tokenize(cleaned_input)
    
    matched_symptoms = []
    for token in tokens:
        if len(token) < 3:  # Skip short tokens
            continue
        best_match, score = process.extractOne(token, symptom_list, scorer=fuzz.token_sort_ratio)
        print(f"Token: '{token}' -> Best match: '{best_match}' (Score: {score})")
        if score >= threshold:
            matched_symptoms.append(best_match)
    
    result = " ".join(matched_symptoms) if matched_symptoms else cleaned_input
    print(f"Matched symptoms: {result}")
    return result

def tokenize_input(text, max_length=128):
    """Tokenize input to match model expectations."""
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
    return encoded["input_ids"], encoded["attention_mask"]

def diagnose(user_input):
    """Predict disease from user symptoms with fuzzy matching."""
    matched_input = match_symptoms(user_input, all_symptoms)
    input_ids, attention_mask = tokenize_input(matched_input)
    
    # Predict
    probabilities = model.predict([input_ids, attention_mask], verbose=0)[0]
    prediction_idx = np.argmax(probabilities)
    prediction = disease_list[prediction_idx]
    confidence = round(float(max(probabilities) * 100), 2)
    
    # Probability dictionary
    prob_dict = {disease: round(float(prob * 100), 2) for disease, prob in zip(disease_list, probabilities)}
    
    # Get prescription
    prescription = prescriptions.get(prediction, "Consult a doctor for further evaluation.")
    
    return user_input, matched_input, prediction, prob_dict, confidence, prescription

def main():
    print("\nWelcome, I'm Doctor Sathi!")
    print("Enter your symptoms (e.g., 'I have a fever and cough') or type 'exit' to quit:")
    
    while True:
        user_input = input("> ")
        if user_input.lower() == "exit":
            print("Goodbye! Stay healthy. ;)")
            break
        
        if not user_input.strip():
            print("Sorry! Please provide some symptoms. :(")
            continue
        
        try:
            original_input, matched_input, disease, probs, confidence, prescription = diagnose(user_input)
            print(f"\nOriginal Input: {original_input}")
            print(f"Matched Symptoms: {matched_input}")
            print(f"Predicted Disease: {disease}")
            print(f"Confidence: {confidence}%")
            print(f"Probabilities: {probs}")
            print(f"Recommendation: {prescription}")
            print("\nEnter more symptoms or 'exit' to quit:")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again or type 'exit' to quit:")

if __name__ == "__main__":
    main()