import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import spacy

# Load spaCy model (using en_core_web_lg with 300D embeddings)
nlp = spacy.load("en_core_web_lg")

# Small free-text dataset
data = [
    ("I have a fever and cough", "Influenza"),
    ("Sore throat and fatigue", "Common Cold"),
    ("Headache and nausea", "Migraine"),
    ("Fever and chills", "Influenza"),
    ("Cough and shortness of breath", "Bronchitis"),
    ("Nausea and fatigue", "Gastroenteritis"),
    ("Sore throat and fever", "Strep Throat"),
    ("Headache all day", "Tension Headache"),
    ("Shortness of breath and cough", "Pneumonia"),
    ("Fever and nausea", "Food Poisoning"),
    ("Cough and sore throat", "Common Cold"),
    ("Fatigue and headache", "Migraine"),
    ("Chills and fever", "Influenza"),
    ("Shortness of breath", "Bronchitis"),
    ("Nausea and fever", "Gastroenteritis"),("Wheezing and shortness of breath", "Asthma"),
    ("Runny nose and sneezing", "Common Cold"),
    ("Fever and chest pain", "Pneumonia"),
    ("Dizziness and headache", "Migraine"),
    ("Burning when I pee", "Urinary Tract Infection"),
    ("Rash and fever", "Food Poisoning"),
    ("Cough with phlegm", "Bronchitis"),
    ("Sore throat and swollen glands", "Strep Throat"),
    ("Fatigue and chills", "Influenza"),
    ("Nausea and vomiting", "Gastroenteritis"),
    ("Headache and stiff neck", "Tension Headache"),
    ("Shortness of breath and wheezing", "Asthma"),
    ("Fever and body aches", "Influenza"),
    ("Cough and runny nose", "Common Cold"),
    ("Painful urination and fever", "Urinary Tract Infection"),
    ("Headache and sensitivity to light", "Migraine"),
    ("Chest tightness and cough", "Pneumonia"),
    ("Sore throat and difficulty swallowing", "Strep Throat"),
    ("Nausea and diarrhea", "Food Poisoning"),
    ("Wheezing and chest pain", "Asthma"),
    ("Fever and nasal congestion", "Sinusitis"),
    ("Fatigue and dizziness", "Gastroenteritis"),
    ("Cough and fever", "Bronchitis"),
    ("Headache and eye strain", "Tension Headache"),
    ("Shortness of breath and fatigue", "Pneumonia"),
    ("Sore throat and hoarse voice", "Common Cold"),
    ("Fever and rash", "Influenza"),
    ("Nausea and abdominal pain", "Gastroenteritis"),
    ("Wheezing and difficulty breathing", "Asthma"),
    ("Sinus pressure and headache", "Sinusitis"),
    ("Burning urination and lower back pain", "Urinary Tract Infection"),
    ("Cough and chest discomfort", "Bronchitis"),
    ("Fever and sore throat", "Strep Throat"),
    ("Headache and fatigue", "Migraine"),
]

# Disease list
disease_list = sorted(set(d[1] for d in data))
print(f"Diseases: {disease_list}")
# Convert symptoms to embeddings and prepare data
X = []
y = []
for symptoms, disease in data:
    doc = nlp(symptoms) # Process symptom text into a spaCy document
    embedding = doc.vector  # 300D embedding
    X.append(embedding)
    y.append(disease_list.index(disease))

X = np.array(X)  # Shape: (num_samples, 300)
y = keras.utils.to_categorical(y, num_classes=len(disease_list))  # Shape: (num_samples, num_diseases)

# Build Keras model
model = Sequential([
    Dense(64, activation='relu', input_shape=(300,)),  # 300D embedding input
    Dense(32, activation='relu'),
    Dense(len(disease_list), activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=4, verbose=1)

# Save model
model.save("diagnosis_model.h5")
print("Model trained and saved as 'diagnosis_model.h5'")