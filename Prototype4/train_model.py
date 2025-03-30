import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Define custom Keras layer to integrate BioBERT
class BioBERTLayer(Layer):
    # Initialize with BioBERT model name and optional arguments
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", trainable=False, **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.trainable = trainable  # Add trainable flag to control freezing
        # Load pre-trained BioBERT, converting from PyTorch to TensorFlow
        self.biobert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.biobert.trainable = self.trainable  # Freeze or unfreeze BioBERT

    # Define forward pass
    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs  # Unpack inputs
        # Pass through BioBERT, training flag controls dropout
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # Return pooler output (CLS token) for classification

    # Enable serialization
    def get_config(self):
        config = super(BioBERTLayer, self).get_config()
        config.update({"model_name": self.model_name, "trainable": self.trainable})
        return config

def train_model(input_file="DiseaseAndSymptoms.csv", model_output="biobert_diagnosis_model.h5"):
    # Load and preprocess data
    data = pd.read_csv(input_file)
    data.columns = [col.replace("_", " ") for col in data.columns]
    data = data.apply(lambda x: x.str.replace("_", " ") if x.dtype == "object" else x)
    data["Disease"] = data["Disease"].replace("Peptic ulcer diseae", "Peptic ulcer disease")
    data["Disease"] = data["Disease"].replace("Dimorphic hemmorhoids(piles)", "Dimorphic hemorrhoids (piles)")
    symptom_cols = [col for col in data.columns if "Symptom" in col]
    data["Symptoms"] = data[symptom_cols].apply(lambda row: " ".join([s for s in row if pd.notna(s)]), axis=1)
    
    # Extract disease list
    disease_list = sorted(data["Disease"].unique())
    num_classes = len(disease_list)

    # Initialize BioBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Tokenize symptoms
    def encode_symptoms(texts, max_length=128):
        encoded = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
        return encoded["input_ids"], encoded["attention_mask"]

    input_ids, attention_masks = encode_symptoms(data["Symptoms"])
    labels = tf.keras.utils.to_categorical([disease_list.index(d) for d in data["Disease"]], num_classes=num_classes)

    # Define model inputs
    input_ids_layer = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Use BioBERT layer with frozen weights (transfer learning)
    biobert_output = BioBERTLayer(trainable=False)(inputs=[input_ids_layer, attention_mask_layer])
    # Add dropout for regularization
    dropout = Dropout(0.3)(biobert_output)
    # Add classification head
    output = Dense(num_classes, activation="softmax")(dropout)

    # Build and compile model
    model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Split data
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids.numpy(), attention_masks.numpy(), labels, test_size=0.2, random_state=42
    )

    # Train with early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(
        [X_train_ids, X_train_mask], y_train,
        validation_data=([X_test_ids, X_test_mask], y_test),
        epochs=10,
        batch_size=8,
        callbacks=[early_stopping]
    )

    # Optional: Fine-tune by unfreezing BioBERT (uncomment if needed after initial training)
    # model.layers[2].biobert.trainable = True  # Unfreeze BioBERT
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    # model.fit([X_train_ids, X_train_mask], y_train, validation_data=([X_test_ids, X_test_mask], y_test), epochs=5, batch_size=8)

    # Save model and disease list
    model.save(model_output)
    np.save("disease_list.npy", disease_list)
    print(f"Model saved to {model_output}")
    print(f"Disease list saved to disease_list.npy")

if __name__ == "__main__":
    train_model()