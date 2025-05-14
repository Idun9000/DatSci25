# %%
# Activate packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # For train-test split
from sklearn.metrics import classification_report # For classification report
from sklearn.decomposition import PCA # For PCA
import transformers # For BERT model
from tqdm import tqdm # For progress bar  
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
# %%
### Change to True to run the code below: 
if False:
    # Load and seperate sentences of dataframe the CSV file
    df = pd.read_csv('../data/TildesSuperData/klima_speeches.csv')
    nlp = spacy.load("da_core_news_sm")

    def split_sentences(text):
        return [sent.text.strip() for sent in nlp(text).sents]
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
    # Save df_expanded to CSV
    df_expanded.to_csv('../data/TildesSuperData/expanded_speeches.csv', index=False)
# %%
df_expanded = pd.read_csv('../data/TildesSuperData/expanded_speeches.csv')
df=df_expanded
df
#%%
# Filter rows containing the word "klima" in the 'text' column
filtered_df = df[df['sentence'].str.contains('klima', case=False, na=False)]

# Display the filtered rows
filtered_df.head()

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")

# Sample 200 rows and reset index
# sampled_df = filtered_df.sample(n=2000, random_state=42).reset_index(drop=True)

# Define target tokens
target_tokens = ["klima", "klimaet"]
target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

# Store embeddings
rows = []

# %%
# Embedding with BERT and progress bar
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Embedding rows"):
    sentence = row['sentence']
    speaker_name = row['speaker']
    year = row['year']
    
    # Tokenize with truncation
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state[0]

    # Save embeddings for relevant tokens
    for idx, token_id in enumerate(input_ids):
        if token_id.item() in target_ids:
            embedding = last_hidden_state[idx].numpy()
            token_str = tokens[idx]
            rows.append({
                "Speaker_name": speaker_name,
                "text": sentence,
                "Year": year,
                "token": token_str,
                **{f"dim_{i}": val for i, val in enumerate(embedding)}
            })

# Build final DataFrame
df = pd.DataFrame(rows)

write_to_csv = False
if write_to_csv:
    # Save the DataFrame to a CSV file
    df.to_csv('../data/embeddings/DanishBertBotxo_Klima.csv', index=False)


# %%








