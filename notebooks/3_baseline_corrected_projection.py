#%%
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
# TODO: Embed and find baseline of average politician utterance
# Load and split speeches into sentences (toggle with `run_preprocessing`)
run_preprocessing = True

if run_preprocessing:
    df = pd.read_csv('../data/TildesSuperData/speeches_subset.csv')
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
                'party' : row['party'],
                'sentence': sent
            })

    df_expanded = pd.DataFrame(rows)
    df_expanded.to_csv('../data/TildesSuperData/baseline_sentences.csv', index=False)

# %%
# Load the preprocessed data
baseline_df = pd.read_csv('../data/TildesSuperData/baseline_sentences.csv')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('intfloat/multilingual-e5-small')
input_texts = [f"query: {s}" for s in baseline_df['sentence']]
embeddings = model.encode(
    input_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
# Convert to DataFrame and optionally save
# Concatenate embeddings with selected metadata columns
embedding_df = pd.concat(
    [pd.DataFrame(embeddings), df_expanded[['year', 'sentence', 'meeting_number', 'speaker', 'party']].reset_index(drop=True)],
    axis=1
)
save_embeddings = True
if save_embeddings:
    embedding_df.to_csv('../data/embeddings/baseline_embedding.csv', index=False)

# %%
# TODO: Subtract this baseline from 'klima sentences'
baseline_embedding = embedding_df.iloc[:,:-5].mean(axis=0).to_numpy()

# TODO: Rerun analysis.






# %%
