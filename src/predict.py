import joblib
from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Example input
text = input("Enter a sentence: ")

# Clean text
cleaned = clean_text(text)

# Convert to vector
vec = vectorizer.transform([cleaned])

# Predict sentiment
prediction = model.predict(vec)

if prediction[0] == 1:
    print("Positive sentiment")
else:
    print("Negative sentiment")