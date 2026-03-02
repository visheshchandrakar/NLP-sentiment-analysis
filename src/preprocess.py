import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # convert to lowercase
    text = text.lower()

    # remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # split into words
    words = text.split()

    # remove stopwords
    words = [word for word in words if word not in stop_words]

    # join words back
    text = " ".join(words)

    return text