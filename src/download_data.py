from datasets import load_dataset
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

dataset = load_dataset("imdb")

train = pd.DataFrame(dataset["train"])
test = pd.DataFrame(dataset["test"])

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Dataset saved in data folder")