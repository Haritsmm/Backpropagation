# predictor.py

"""Utility functions for predicting a student's academic potential."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

MODEL_PATH = Path("model.h5")
DATA_PATH = Path("Dataset/data_siswa_seimbang.csv")


@lru_cache(maxsize=1)
def _load_model():
    """Load the trained Keras model only once."""
    return load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_label_encoder() -> LabelEncoder:
    """Load and fit the label encoder from the dataset."""
    data = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    le.fit(data["Potensi"])
    return le

# Fitur yang digunakan
FEATURE_COLUMNS = [
    "Usia",
    "Nilai Matematika",
    "Nilai IPA",
    "Nilai IPS",
    "Nilai Bahasa Indonesia",
    "Nilai Bahasa Inggris",
    "Nilai TIK",
    "Minat Sains",
    "Minat Bahasa",
    "Minat Sosial",
    "Minat Teknologi",
]

def predict_potensi(input_dict: Dict[str, float]) -> Tuple[str, np.ndarray]:
    """Predict the academic potential.

    Parameters
    ----------
    input_dict : dict
        Mapping of feature names to values.

    Returns
    -------
    tuple
        The predicted label and the raw probability array.
    """

    # Convert input into the required DataFrame structure
    X = pd.DataFrame([input_dict])[FEATURE_COLUMNS]

    # Perform prediction
    model = _load_model()
    le = _load_label_encoder()
    y_pred_prob = model.predict(X, verbose=0)
    y_pred_num = np.argmax(y_pred_prob, axis=1)
    potensi_pred = le.inverse_transform(y_pred_num)[0]

    return potensi_pred, y_pred_prob[0]
