import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

# Load dataset
dataset_url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
data = pd.read_csv(dataset_url)

# Select relevant columns
data = data[['tweet', 'class']]

# Map class labels (0: hate speech, 1: offensive, 2: neither) for simplicity
data['class'] = data['class'].map({0: 0, 1: 1, 2: 1}) 

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['tweet'], data['class'], test_size=0.2, random_state=42
)


# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Save tokenized data
torch.save((train_encodings, train_labels), "train_data.pt")
torch.save((test_encodings, test_labels), "test_data.pt")
