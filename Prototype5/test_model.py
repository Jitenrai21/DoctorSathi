import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split

# Define BioBERTLayer to match the training definition
class BioBERTLayer(Layer):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", trainable=False, **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.trainable = trainable
        # Don’t initialize TFBertModel here; weights will be loaded from the saved model
        self.biobert = None

    def build(self, input_shape):
        # If not loaded, initialize (but this won’t run since weights are restored later)
        if self.biobert is None:
            self.biobert = TFBertModel.from_pretrained(self.model_name, from_pt=True)
        self.biobert.trainable = self.trainable

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        # If biobert isn’t set (shouldn’t happen with loaded model), this would fail
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # CLS token embedding

    def get_config(self):
        config = super(BioBERTLayer, self).get_config()
        config.update({"model_name": self.model_name, "trainable": self.trainable})
        return config

def load_and_test(model_path="biobert_diagnosis_model.h5", disease_list_path="disease_list.npy", data_file="DiseaseAndSymptoms.csv"):
    # Load model and disease list
    model = tf.keras.models.load_model(model_path, custom_objects={"BioBERTLayer": BioBERTLayer})
    disease_list = np.load(disease_list_path, allow_pickle=True)

    # Load and preprocess data
    data = pd.read_csv(data_file)
    data.columns = [col.replace("_", " ") for col in data.columns]
    data = data.apply(lambda x: x.str.replace("_", " ") if x.dtype == "object" else x)
    data["Disease"] = data["Disease"].replace("Peptic ulcer diseae", "Peptic ulcer disease")
    data["Disease"] = data["Disease"].replace("Dimorphic hemmorhoids(piles)", "Dimorphic hemorrhoids (piles)")
    symptom_cols = [col for col in data.columns if "Symptom" in col]
    data["Symptoms"] = data[symptom_cols].apply(lambda row: " ".join([s for s in row if pd.notna(s)]), axis=1)

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    def encode_symptoms(texts, max_length=128):
        encoded = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
        return encoded["input_ids"], encoded["attention_mask"]

    input_ids, attention_masks = encode_symptoms(data["Symptoms"])
    labels = tf.keras.utils.to_categorical([disease_list.tolist().index(d) for d in data["Disease"]], num_classes=len(disease_list))

    # Split (same random_state as training)
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids.numpy(), attention_masks.numpy(), labels, test_size=0.2, random_state=42
    )

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate([X_test_ids, X_test_mask], y_test, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Sample predictions
    val_preds = model.predict([X_test_ids[:5], X_test_mask[:5]])
    print("Sample predictions (top 5 validation samples):")
    for i, pred in enumerate(val_preds):
        pred_idx = np.argmax(pred)
        true_idx = np.argmax(y_test[i])
        print(f"Sample {i}: Predicted: {disease_list[pred_idx]} ({pred[pred_idx]:.4f}), True: {disease_list[true_idx]}")

    # Top-3 predictions for insight
    print("\nTop-3 predictions for first 5 samples:")
    for i, pred in enumerate(val_preds):
        top_3_idx = np.argsort(pred)[-3:][::-1]
        top_3_probs = pred[top_3_idx]
        print(f"Sample {i}: {[(disease_list[idx], f'{prob:.4f}') for idx, prob in zip(top_3_idx, top_3_probs)]}")

    print("\nFull probabilities for Sample 2:")
    pred = val_preds[2]
    for idx, prob in enumerate(pred):
        print(f"{disease_list[idx]}: {prob:.4f}")

    # Add after loading model and disease_list
    custom_symptoms = "itching skin rash dischromic patches"
    encoded = tokenizer(custom_symptoms, padding="max_length", truncation=True, max_length=128, return_tensors="tf")
    custom_pred = model.predict([encoded["input_ids"], encoded["attention_mask"]])
    pred_idx = np.argmax(custom_pred[0])
    print(f"\nCustom Input Prediction: {disease_list[pred_idx]} ({custom_pred[0][pred_idx]:.4f})")

if __name__ == "__main__":
    load_and_test()