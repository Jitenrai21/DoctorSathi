import pandas as pd
data = pd.read_csv(r"C:\Users\ACER\gitClones\DoctorSathi\DataPreprocessing\DiseaseAndSymptoms.csv")  # Replace with your full file path
print(f"Number of rows: {len(data)}")
print(f"Number of unique diseases: {len(data['Disease'].unique())}")
print("Class distribution:\n", data["Disease"].value_counts())