import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Sample dataset (you can replace with your own)
data = {
    "text": [
        "I love this movie!",
        "This was a fantastic experience.",
        "Horrible acting, I hated it.",
        "What a waste of time.",
        "Absolutely wonderful!",
        "I will never watch this again.",
        "It was okay, not great.",
        "Really enjoyed it.",
        "So boring and bad.",
        "Brilliant! Highly recommended."
    ],
    "label": [1, 1, 0, 0, 1, 0, 0, 1, 0, 1]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

# 2. Preprocess and train
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 3. Prediction function
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    proba = model.predict_proba(text_vec)[0][pred]
    label = "Positive" if pred == 1 else "Negative"
    return f"Sentiment: {label} (Confidence: {proba:.2f})"

# 4. Gradio Interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter a review or sentence"),
    outputs=gr.Text(label="Sentiment Prediction"),
    title="Sentiment Analysis using Logistic Regression",
    description="Type a sentence and see if it's Positive or Negative!"
)

# 5. Run locally
interface.launch()
