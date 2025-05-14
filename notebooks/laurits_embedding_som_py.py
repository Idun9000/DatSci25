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
# %%
# Load the CSV file
# df = pd.read_csv('../data/PBramsCleaned/clean_parlamint3.csv')
df = pd.read_csv('../data/TildesSuperData/klima_speeches.csv')
df
#%%
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
sampled_df = filtered_df.sample(n=2000, random_state=42).reset_index(drop=True)

# Define target tokens
target_tokens = ["klima", "klimaet"]
target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

# Store embeddings
rows = []
#%%
sampled_df


# %%
# Embedding with BERT and progress bar
for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Embedding rows"):
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

#%%
df
# %%
# Perform PCA on the embeddings and polot first two components
# Select embeddings
embeddings = df.iloc[:, 4:]

# Perform PCA with 2 components
pca = PCA(n_components=10)
pca_result = pca.fit_transform(embeddings)

# Get variance explained
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1 and PC2: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}")

# Create new DataFrame with original first 4 columns and PCA results
pca_df = df.iloc[:, :4].copy()
pca_df['PC1'] = pca_result[:, 0]
pca_df['PC2'] = pca_result[:, 1]

# Show the result
pca_df

# Sort by Month, then Year
pca_df_sorted = pca_df.sort_values(by=['Year']).reset_index(drop=True)

# Show the result
pca_df_sorted

# %%
# Plot the average PCA results by year
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the first two PCA dimensions (PC1 vs PC2), coloring by 'Year'
sns.scatterplot(x=pca_df_sorted['PC1'], y=pca_df_sorted['PC2'], hue=pca_df_sorted['Year'], palette='viridis', s=100)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Embeddings (Colored by Year)')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.savefig('pca_plot.png', bbox_inches='tight')
plt.show()

# %%
# Group by Year and compute mean PC1 and PC2
mean_pca_by_year = pca_df_sorted.groupby('Year')[['PC1', 'PC2']].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mean_pca_by_year, x='PC1', y='PC2', hue='Year', palette='viridis', s=150)

# Annotate with year labels
for _, row in mean_pca_by_year.iterrows():
    plt.text(row['PC1'] + 0.01, row['PC2'], str(row['Year']), fontsize=10)

# Labels and title
plt.xlabel('Mean Principal Component 1')
plt.ylabel('Mean Principal Component 2')
plt.title('Mean PCA Embedding per Year')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
