import gradio as gr
import pickle
import os

# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your trained sentiment model
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Prediction function
def predict_sentiment(text):
    if not text:
        return "Please enter some text."
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return prediction

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."),
    outputs="text",
    title="Sentiment & Emotion Analyzer",
    description="Enter text and get sentiment prediction (positive/negative/neutral)."
)

# Launch the app (this will be automatically run by Hugging Face Spaces)
iface.launch()
