import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline

# ------------------------
# Paths for Models & Frontend
# ------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# ------------------------
# Load Reddit-trained model
# ------------------------
reddit_model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ------------------------
# HuggingFace pipelines
# ------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Data Model
# ------------------------
class TextData(BaseModel):
    text: str

# ------------------------
# Routes
# ------------------------
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/predict_sentiment/")
def predict_sentiment(data: TextData):
    # HuggingFace sentiment
    result = sentiment_pipeline(data.text)[0]

    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    sentiment = label_map.get(result["label"], "neutral")
    score = float(result["score"])  # confidence (0.0–1.0)

    # Emotion analysis
    emotions = emotion_pipeline(data.text, top_k=6)
    emotions = [{"label": r["label"].lower(), "score": float(r["score"])} for r in emotions]
    top_emotion = max(emotions, key=lambda x: x["score"])

    # Hybrid override
    if sentiment == "neutral" and top_emotion["score"] >= 0.70:
        if top_emotion["label"] in ["disgust", "anger", "sadness", "fear"]:
            sentiment = "negative"
        elif top_emotion["label"] in ["joy", "love", "surprise"]:
            sentiment = "positive"

    return {
        "text": data.text,
        "sentiment": sentiment,
        "confidence": score,
        "top_emotion": top_emotion,
        "emotions": emotions
    }

@app.post("/predict_reddit_sentiment/")
def predict_reddit_sentiment(data: TextData):
    # Reddit-trained model sentiment
    X = vectorizer.transform([data.text])
    pred = reddit_model.predict(X)[0]
    sentiment = str(pred).lower()

    # Emotion analysis
    emotions = emotion_pipeline(data.text, top_k=6)
    emotions = [{"label": r["label"].lower(), "score": float(r["score"])} for r in emotions]
    top_emotion = max(emotions, key=lambda x: x["score"])

    # Hybrid override
    if sentiment == "neutral" and top_emotion["score"] >= 0.70:
        if top_emotion["label"] in ["disgust", "anger", "sadness", "fear"]:
            sentiment = "negative"
        elif top_emotion["label"] in ["joy", "love", "surprise"]:
            sentiment = "positive"

    return {
        "text": data.text,
        "sentiment": sentiment,
        "top_emotion": top_emotion,
        "emotions": emotions
    }

@app.post("/predict_emotion/")
def predict_emotion(data: TextData):
    results = emotion_pipeline(data.text, top_k=6)
    emotions = [{"label": r["label"], "score": float(r["score"])} for r in results]
    top_emotion = max(emotions, key=lambda x: x["score"])

    if top_emotion["score"] < 0.30:
        return {"text": data.text, "emotions": [{"label": "neutral", "score": 1.0}]}

    return {"text": data.text, "emotions": emotions}
