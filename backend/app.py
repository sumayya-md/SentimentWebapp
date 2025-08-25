from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
import os

# Initialize FastAPI app
app = FastAPI(title="Sentiment & Emotion Analyzer")

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained sentiment model
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Home route
@app.get("/")
def home():
    return {"message": "Sentiment & Emotion Analyzer is running!"}

# Prediction route
@app.post("/predict")
def predict_sentiment(data: dict):
    text = data.get("text")
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is required"})
    
    # Transform text using the vectorizer
    vectorized_text = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(vectorized_text)[0]
    
    return {"text": text, "sentiment": prediction}
