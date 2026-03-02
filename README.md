LINK : https://nlpsentimentanalysis.streamlit.app/
# NLP Sentiment Analysis

This project implements a Sentiment Analysis system using classical Natural Language Processing (NLP) techniques and machine learning.

The model predicts whether a given sentence expresses positive or negative sentiment.

##Demo
<img width="950" height="892" alt="Screenshot 2026-03-03 004136" src="https://github.com/user-attachments/assets/54167ed7-66c8-443d-a50c-d7ab3c85333d" />


## Project Overview

The pipeline includes:

1. Text preprocessing
2. Feature extraction using TF-IDF
3. Training a Logistic Regression classifier
4. Model evaluation
5. Interactive prediction using a Streamlit web app

## Tech Stack

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- HuggingFace Datasets

## Project Structure

nlp-sentiment-analysis
│
├── data
│   ├── train.csv
│   └── test.csv
│
├── model
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── src
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── app.py
│
├── requirements.txt
└── README.md

## Dataset

The project uses the IMDB movie reviews dataset.

It contains 50,000 labeled reviews for binary sentiment classification.

Dataset source:
https://huggingface.co/datasets/imdb

## Installation

Clone the repository:

git clone https://github.com/visheshchandrakar/NLP-sentiment-analysis.git

cd NLP-sentiment-analysis

Create a virtual environment:

python -m venv venv

Activate it (Windows):

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Download Dataset

python src/download_data.py

## Train the Model

python src/train_model.py

This will train the sentiment classifier and save the model.

## Run Prediction Script

python src/predict.py

Example:

Enter a sentence: This movie was amazing  
Positive sentiment

## Run the Web App

streamlit run src/app.py

Then open:

http://localhost:8501

You can enter text and the model will predict sentiment.

## Future Improvements

- Transformer-based sentiment model (BERT)
- Model deployment on cloud
- Docker containerization
- Real-time API using FastAPI

## Author

Vishesh Chandrakar 
