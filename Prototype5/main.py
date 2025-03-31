import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.layers import Layer
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define BioBERTLayer
class BioBERTLayer(Layer):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", trainable=False, **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.trainable = trainable

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        pass

    def get_config(self):
        config = super(BioBERTLayer, self).get_config()
        config.update({"model_name": self.model_name, "trainable": self.trainable})
        return config

def load_model_and_predict(symptoms, model_path="biobert_diagnosis_model.h5", disease_list_path="disease_list.npy", temperature=0.5):
    # Check model file
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(model_path)
    file_size_mb = os.path.getsize(model_path) / 1024 / 1024
    logging.info(f"Model file found: {model_path}, size: {file_size_mb:.2f} MB")
    if file_size_mb < 50:  # BioBERT should be ~400 MB
        logging.error(f"Model file too small ({file_size_mb:.2f} MB). Expected ~400 MB. File may be corrupt.")
        raise ValueError("Model file is too small to be a valid BioBERT model.")

    logging.info("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"BioBERTLayer": BioBERTLayer})
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    # Check disease list
    if not os.path.exists(disease_list_path):
        logging.error(f"Disease list file not found: {disease_list_path}")
        raise FileNotFoundError(disease_list_path)
    logging.info("Loading disease list...")
    disease_list = np.load(disease_list_path, allow_pickle=True)
    logging.info("Disease list loaded.")

    logging.info("Tokenizing symptoms...")
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    encoded = tokenizer(symptoms, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    logging.info("Symptoms tokenized.")

    logging.info("Running prediction...")
    preds = model.predict([encoded["input_ids"], encoded["attention_mask"]])[0]
    logging.info("Prediction completed.")

    # Apply temperature scaling
    logging.info("Applying temperature scaling...")
    logits = np.log(preds + 1e-10)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits)
    scaled_probs = exp_logits / np.sum(exp_logits)

    # Get top-3 predictions
    top_3_idx = np.argsort(scaled_probs)[-3:][::-1]
    top_3_probs = scaled_probs[top_3_idx]
    return [(disease_list[idx], prob) for idx, prob in zip(top_3_idx, top_3_probs)]

if __name__ == "__main__":
    prompt = (
        'Enter symptoms (space-separated, e.g., "itching skin rash dischromic patches"):\n'
        'Please provide 3-5 specific symptoms for best results. The model will predict the top-3 likely diseases with confidence scores.\n'
    )
    symptoms = input(prompt)
    try:
        predictions = load_model_and_predict(symptoms, temperature=0.5)
        print("\nTop-3 Predictions (Confidence):")
        for disease, prob in predictions:
            print(f"{disease}: {prob:.4f} ({prob*100:.1f}%)")
    except Exception as e:
        logging.error(f"Program failed: {e}")
        sys.exit(1)