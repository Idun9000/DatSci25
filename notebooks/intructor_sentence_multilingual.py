# %%
# Activate packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # For train-test split
from sklearn.metrics import classification_report # For classification report
from sklearn.decomposition import PCA # For PCA
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm # For progress bar  
import matplotlib.pyplot as plt
import spacy

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
df = df_expanded
df

#%%
# Filter rows containing the word "klima" in the 'text' column
filtered_df = df[df['sentence'].str.contains('klima', case=False, na=False)]

# Display the filtered rows
filtered_df.head()

# Initialize INSTRUCTOR model
model_name = "hkunlp/instructor-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define instruction in Danish
instruction = "Beskriv hvilken mening ordet klima har i denne tekst"

# Store embeddings
rows = []

# %%
# Embedding with INSTRUCTOR and progress bar
for index, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Embedding rows"):
    sentence = row['sentence']
    speaker_name = row['speaker']
    year = row['year']
    
    # Format input for INSTRUCTOR model
    model_input = [[instruction, sentence]]
    
    # Tokenize with truncation
    inputs = tokenizer(model_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        embedding = embeddings[0].numpy()  # Get the sentence embedding
    
    # Save embeddings
    rows.append({
        "Speaker_name": speaker_name,
        "text": sentence,
        "Year": year,
        **{f"dim_{i}": val for i, val in enumerate(embedding)}
    })

# Build final DataFrame
df = pd.DataFrame(rows)

write_to_csv = False
if write_to_csv:
    # Save the DataFrame to a CSV file
    df.to_csv('../data/embeddings/Instructor_Klima.csv', index=False)


# %%








