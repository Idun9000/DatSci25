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
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-small')
input_texts = [f"query: {s}" for s in filtered_df['sentence']]
embeddings = model.encode(
    input_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# %%
# Convert to DataFrame and optionally save
# Concatenate embeddings with selected metadata columns
embedding_df = pd.concat(
    [pd.DataFrame(embeddings), filtered_df[['year', 'sentence', 'meeting_number', 'speaker', 'party']].reset_index(drop=True)],
    axis=1
)
save_embeddings = True
if save_embeddings:
    embedding_df.to_csv('../data/embeddings/Sentence_Klima.csv', index=False)


# %%