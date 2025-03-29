import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from transformers import BertTokenizer, TFBertModel
from keras.layers import Layer, Dense, Input
from keras.models import Model
from fuzzywuzzy import process, fuzz
import nltk
import re
from nltk.corpus import stopwords # For removing common English stopwords

# Download required NLTK data
nltk.download('punkt') # Tokenizer for splitting text into words
nltk.download('stopwords') # List of common English stopwords
stop_words = set(stopwords.words('english')) # Convert stopwords to a set for efficient lookup

# Define a custom Keras layer to integrate BioBERT
class BioBERTLayer(Layer):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        # Storing the model name for serialization
        self.model_name = model_name
        self.biobert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        # Pass inputs through BioBERT; training=False disables dropout for inference
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # Return pooler_output (CLS token)

    def get_config(self):
        # Provide configuration for serialization (needed for model loading)
        config = super(BioBERTLayer, self).get_config()
        config.update({"model_name": self.model_name})
        return config

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load disease list and number of classes
disease_list = np.load("disease_list.npy", allow_pickle=True).tolist() #pickled objects require allow_pickle=True to load correctly
num_classes = len(disease_list)

# Rebuild the model architecture (matching train_model.py)
def build_model():
    input_ids_layer = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Pass inputs through the custom BioBERT layer
    biobert_output = BioBERTLayer()([input_ids_layer, attention_mask_layer])
    output = Dense(num_classes, activation="softmax")(biobert_output)
    model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Load weights into the rebuilt model
model = build_model()
try:
    model.load_weights("biobert_diagnosis_model.h5")
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Load dataset to extract unique symptoms (for fuzzy matching reference)
data = pd.read_csv("DiseaseAndSymptoms.csv")
data.columns = [col.replace("_", " ") for col in data.columns]
symptom_cols = [col for col in data.columns if "Symptom" in col]

# Extract unique symptoms by melting the DataFrame and dropping NaN values
all_symptoms = data[symptom_cols].melt().dropna()["value"].unique().tolist()
all_symptoms = [s.replace("_", " ") for s in all_symptoms]
# print(f"Available symptoms: {all_symptoms[:10]}... (total: {len(all_symptoms)})")

# Define prescriptions
prescriptions = {
    "Fungal infection": "Apply antifungal cream, keep area dry, consult a doctor if persistent.",
    "Allergy": "Avoid allergens, take antihistamines, seek medical advice if severe.",
    "GERD": "Avoid spicy/acidic foods, take antacids, consult a doctor.",
    "Heart attack": "Call emergency services immediately.",
    "Pneumonia": "Seek immediate medical attention, rest, and follow a doctor’s antibiotic treatment.",
    "Common Cold": "Rest, stay hydrated, take over-the-counter cold remedies.",
    "Hepatitis B": "Consult a doctor for antiviral treatment and monitoring.",
    "Hyperthyroidism": "Consult a doctor for thyroid function tests and treatment.",
    "Peptic ulcer disease": "Take antacids or PPIs, avoid NSAIDs, consult a doctor.",
    "Bronchial Asthma": "Use an inhaler, avoid triggers, seek medical advice if uncontrolled."
    # Add more diseases from disease_list as needed
}

# Defining function to clean and fuzzy-match the full input string
def match_symptoms(user_input, symptom_list, threshold=75):
    """Clean and fuzzy-match the full input string to preserve context."""
    # Clean input: remove punctuation, lowercase, remove stopwords
    cleaned_input = re.sub(r'[^\w\s]', '', user_input.lower())
    # Tokenize and remove stopwords to focus on meaningful words
    tokens = [t for t in nltk.word_tokenize(cleaned_input) if t not in stop_words]
    cleaned_input = " ".join(tokens)
    
    # Fuzzy match the full phrase against symptom list
    best_match, score = process.extractOne(cleaned_input, symptom_list, scorer=fuzz.token_sort_ratio)
    print(f"Input: '{cleaned_input}' -> Best match: '{best_match}' (Score: {score})")
    
    # Use the best match if above threshold, otherwise use cleaned input
    return best_match if score >= threshold else cleaned_input

def tokenize_input(text, max_length=128):
    """Tokenize input to match model expectations."""
    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
    return encoded["input_ids"], encoded["attention_mask"]

def diagnose(user_input):
    """Predict disease from user symptoms with fuzzy matching and confidence threshold."""
    matched_input = match_symptoms(user_input, all_symptoms)
    input_ids, attention_mask = tokenize_input(matched_input)
    
    # Predict
    probabilities = model.predict([input_ids, attention_mask], verbose=0)[0]
    prediction_idx = np.argmax(probabilities) # Find the index of the highest probability
    prediction = disease_list[prediction_idx] # Map index to disease name
    confidence = round(float(max(probabilities) * 100), 2)
    prob_dict = {disease: round(float(prob * 100), 2) for disease, prob in zip(disease_list, probabilities)}
    
    # Apply confidence threshold
    if confidence < 50:
        prediction = f"Possible {prediction} (low confidence)"
        prescription = "Confidence too low; consult a doctor for a proper diagnosis."
    else:
        prescription = prescriptions.get(prediction, "Consult a doctor for further evaluation.")
    
    return user_input, matched_input, prediction, prob_dict, confidence, prescription

# main interactive function
def main():
    print("\nWelcome, I'm Doctor Sathi!")
    print("Enter your symptoms (e.g., 'I have a fever and cough') or type 'exit' to quit.")
    print("Add 'details' (e.g., 'fever details') for more info.")
    
    while True:
        user_input = input("> ").strip() # Get user input and remove leading/trailing whitespace
        if user_input.lower() == "exit":
            print("Goodbye! Stay healthy. ;)")
            break
        
        if not user_input:
            print("Sorry! Please provide some symptoms. :(")
            continue
        
        show_details = "details" in user_input.lower()
        if show_details:
            user_input = user_input.lower().replace("details", "").strip()
        
        try:
            original_input, matched_input, disease, probs, confidence, prescription = diagnose(user_input)
            # Simplified output for regular users
            print(f"\nPredicted Disease: {disease} ({confidence}% confidence)")
            print(f"Recommendation: {prescription}")
            
            # Detailed output if requested
            if show_details:
                print(f"Original Input: {original_input}")
                print(f"Matched Symptoms: {matched_input}")
                print(f"Probabilities: {probs}")
            
            print("\nEnter more symptoms or 'exit' to quit:")
        except ValueError as ve:
            print(f"Input processing error: {ve}")
            print("Please try again or type 'exit' to quit:")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Please try again or type 'exit' to quit:")

if __name__ == "__main__":
    main()