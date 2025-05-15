# %%
# Import packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import spacy

# %%
# Load and split speeches into sentences (toggle with `run_preprocessing`)
run_preprocessing = False

if run_preprocessing:
    df = pd.read_csv('../data/TildesSuperData/klima_speeches.csv')
    nlp = spacy.load("da_core_news_sm")

    def split_sentences(text):
        """Split text into sentences using spaCy."""
        return [sent.text.strip() for sent in nlp(text).sents]

    # Expand speeches into individual sentences
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = split_sentences(row['speech'])
        for sent in sentences:
            rows.append({
                'year': row['year'],
                'meeting_number': row['meeting_number'],
                'speaker': row['speaker'],
                'sentence': sent
            })

    df_expanded = pd.DataFrame(rows)
    df_expanded.to_csv('../data/TildesSuperData/expanded_speeches.csv', index=False)

# %%
# Load the preprocessed data
df = pd.read_csv('../data/TildesSuperData/expanded_speeches.csv')

# %%
# Filter for sentences that mention 'klima'
filtered_df = df[df['sentence'].str.contains('klima', case=False, na=False)]
filtered_df.head()

# %%
# Initialize the SENTENCE EMBEDDER model
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# %%
# Generate sentence embeddings
embedding_rows = []

for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Embedding rows"):
    sentence = row['sentence']
    speaker = row['speaker']
    year = row['year']

    # Format input for INSTRUCTOR model
    model_input = [sentence]
    inputs = tokenizer(model_input, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)[0].numpy()  # Average pooling

    embedding_rows.append({
        "Speaker_name": speaker,
        "text": sentence,
        "Year": year,
        **{f"dim_{i}": val for i, val in enumerate(embedding)}
    })

# %%
# Convert to DataFrame and optionally save
embedding_df = pd.DataFrame(embedding_rows)

save_embeddings = True
if save_embeddings:
    embedding_df.to_csv('../data/embeddings/Sentence_Klima.csv', index=False)
