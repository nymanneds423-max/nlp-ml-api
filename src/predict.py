import joblib
import os
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Auto-train if missing
if not os.path.exists(MODEL_PATH):
    from src.train import train_model
    train_model()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def predict(text: str):
    text = clean_text(text)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    
    return "Positive" if pred == 1 else "Negative"