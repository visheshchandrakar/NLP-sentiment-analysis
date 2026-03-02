import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess import clean_text
import joblib

# Load dataset
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Clean text
train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

# Convert text to numerical features
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train["clean_text"])
X_test = vectorizer.transform(test["clean_text"])

y_train = train["label"]
y_test = test["label"]

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully.")