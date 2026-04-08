from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    texts = [
        "I love this product",
        "This is amazing",
        "Very bad experience",
        "Worst service ever",
        "I am very happy",
        "I hate this"
    ]
    
    labels = [1, 1, 0, 0, 1, 0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    # 🔥 Correct path handling
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "models")

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    print("✅ Model saved at:", model_dir)

if __name__ == "__main__":
    train_model()