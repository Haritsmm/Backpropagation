# predictor.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("model.h5")

# Load dataset
data = pd.read_csv("Dataset/data_siswa_seimbang.csv")

# Load Label Encoder
le = LabelEncoder()
le.fit(data['Potensi'])

# Fitur yang digunakan
FEATURE_COLUMNS = [
    'Usia',
    'Nilai Matematika',
    'Nilai IPA',
    'Nilai IPS',
    'Nilai Bahasa Indonesia',
    'Nilai Bahasa Inggris',
    'Nilai TIK',
    'Minat Sains',
    'Minat Bahasa',
    'Minat Sosial',
    'Minat Teknologi'
]

def predict_potensi(input_dict):
    # Ubah input ke DataFrame
    X = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    
    # Prediksi
    y_pred_prob = model.predict(X)
    y_pred_num = np.argmax(y_pred_prob, axis=1)
    potensi_pred = le.inverse_transform(y_pred_num)[0]
    
    return potensi_pred, y_pred_prob[0]
