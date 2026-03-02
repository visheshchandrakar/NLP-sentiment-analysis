import streamlit as st
import joblib
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.title("Sentiment Analysis App")

st.write("Enter a sentence and the model will predict the sentiment.")

# User input
text = st.text_input("Enter text")

if text:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)

    if prediction[0] == 1:
        st.success("Positive Sentiment 😀")
    else:
        st.error("Negative Sentiment 😞")