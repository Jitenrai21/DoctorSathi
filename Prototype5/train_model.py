import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Custom BioBERT layer
class BioBERTLayer(Layer):
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", trainable=False, **kwargs):
        super(BioBERTLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.trainable = trainable
        self.biobert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.biobert.trainable = self.trainable

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # CLS token embedding

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
    
    # Verify data
    print(f"Number of rows: {len(data)}")
    disease_list = sorted(data["Disease"].unique())
    num_classes = len(disease_list)
    print(f"Number of classes: {num_classes}")
    print("Class distribution:\n", data["Disease"].value_counts())

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    def encode_symptoms(texts, max_length=128):
        encoded = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
        return encoded["input_ids"], encoded["attention_mask"]

    input_ids, attention_masks = encode_symptoms(data["Symptoms"])
    labels = tf.keras.utils.to_categorical([disease_list.index(d) for d in data["Disease"]], num_classes=num_classes)

    # Split data (80% train, 20% validation)
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids.numpy(), attention_masks.numpy(), labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {X_train_ids.shape[0]}, Validation samples: {X_test_ids.shape[0]}")  # Expect ~3936 train, ~984 val

    # Define model
    input_ids_layer = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    biobert_output = BioBERTLayer(trainable=False)(inputs=[input_ids_layer, attention_mask_layer])
    dropout = Dropout(0.3)(biobert_output)
    output = Dense(num_classes, activation="softmax")(dropout)
    model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train head (frozen BioBERT)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history_head = model.fit(
        [X_train_ids, X_train_mask], y_train,
        validation_data=([X_test_ids, X_test_mask], y_test),
        epochs=15,  # More epochs for 4920 rows
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Fine-tune (unfreeze BioBERT)
    print("\nFine-tuning BioBERT...")
    model.layers[2].biobert.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    history_finetune = model.fit(
        [X_train_ids, X_train_mask], y_train,
        validation_data=([X_test_ids, X_test_mask], y_test),
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    train_loss, train_acc = model.evaluate([X_train_ids, X_train_mask], y_train, verbose=0)
    # print(f"Final Training Loss: {train_loss:.4f}, Final Training Accuracy: {train_acc:.4f}")
    # Evaluate
    val_loss, val_accuracy = model.evaluate([X_test_ids, X_test_mask], y_test, verbose=0)
    # print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_accuracy:.4f}")

    # Sample predictions
    val_preds = model.predict([X_test_ids[:5], X_test_mask[:5]])
    print("Sample predictions (top 5 validation samples):")
    for i, pred in enumerate(val_preds):
        pred_idx = np.argmax(pred)
        true_idx = np.argmax(y_test[i])
        print(f"Sample {i}: Predicted: {disease_list[pred_idx]} ({pred[pred_idx]:.4f}), True: {disease_list[true_idx]}")

    # Save model and disease list
    model.save(model_output)
    np.save("disease_list.npy", disease_list)
    print(f"Model saved to {model_output}")
    print(f"Disease list saved to disease_list.npy")

if __name__ == "__main__":
    train_model()