from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "NLP API is running 🚀"}

@app.post("/predict")
def get_prediction(data: InputText):
    result = predict(data.text)
    return {"sentiment": result}