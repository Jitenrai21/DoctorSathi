import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

#defining custom Keras layer to integrate BioBERT into the model
class BioBERTLayer(Layer):
    # Initialize the layer with the specified BioBERT model name and any additional arguments
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", **kwargs):
        #initializing the layer with specified BioBERT model name and additional arguments
        super(BioBERTLayer, self).__init__(**kwargs)
        #loading the pre-trained model, converting from pytorch to tf (from_pt=True)
        self.biobert = TFBertModel.from_pretrained(model_name, from_pt=True)

    def call(self, inputs, training=False):
        #defining how the layer processes inputs during forward pass
        input_ids, attention_mask = inputs #unpacking inputs
        #passing inputs through BioBERT, training flag controls dropout behavior
        outputs = self.biobert(input_ids, attention_mask=attention_mask, training=training)
        return outputs[1]  # Return pooler_output (CLS token) for classification

def train_model(input_file="DiseaseAndSymptoms.csv", model_output="biobert_diagnosis_model.h5"):
    # Load and preprocess data
    data = pd.read_csv(input_file)
    data.columns = [col.replace("_", " ") for col in data.columns]
    data = data.apply(lambda x: x.str.replace("_", " ") if x.dtype == "object" else x)
    data["Disease"] = data["Disease"].replace("Peptic ulcer diseae", "Peptic ulcer disease")
    data["Disease"] = data["Disease"].replace("Dimorphic hemmorhoids(piles)", "Dimorphic hemorrhoids (piles)")
    symptom_cols = [col for col in data.columns if "Symptom" in col]
    # Combine all non-null symptoms into a single "Symptoms" column for each row 
    data["Symptoms"] = data[symptom_cols].apply(lambda row: " ".join([s for s in row if pd.notna(s)]), axis=1)
    
    # Extract disease list
    disease_list = sorted(data["Disease"].unique())

    # Initialize BioBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Tokenize symptoms with explicit padding to max_length=128
    def encode_symptoms(texts, max_length=128):
        # Convert pandas Series to list and tokenize with padding and truncation
        encoded = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")
        return encoded["input_ids"], encoded["attention_mask"] 

    input_ids, attention_masks = encode_symptoms(data["Symptoms"])
    # Convert disease names to one-hot encoded labels based on their index in disease_list
    labels = tf.keras.utils.to_categorical([disease_list.index(d) for d in data["Disease"]], num_classes=len(disease_list))

    # Define model inputs (must match tokenized sequence length)
    input_ids_layer = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Use custom BioBERT layer
    biobert_output = BioBERTLayer()(inputs=[input_ids_layer, attention_mask_layer])
    output = Dense(len(disease_list), activation="softmax")(biobert_output)

    # Build and compile model
    model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Split data
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids.numpy(), attention_masks.numpy(), labels, test_size=0.2, random_state=42
    )
    # Train
    model.fit(
        [X_train_ids, X_train_mask], y_train,
        validation_data=([X_test_ids, X_test_mask], y_test),
        epochs=10, batch_size=8
    )

    # Save model and disease list
    model.save(model_output)
    np.save("disease_list.npy", disease_list)
    print(f"Model saved to {model_output}")
    print(f"Disease list saved to disease_list.npy")

if __name__ == "__main__":
    train_model()