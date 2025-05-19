# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer
# %%
# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
sentimentLoader = CorpusLoader(text_col="text", label_col="label")
sentimentLoader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
sentimentLoader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42)
sentimentMPNET = MultiLingMPNET.embed(sentimentLoader.train_texts, cache_path="../data/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
sentimentMPNET["label"] = sentimentLoader.train_labels

# %%
klimaLoader = CorpusLoader(path="../data/TildesSuperData/expanded_speeches.csv", text_col="sentence")
klimaMPNET = MultiLingMPNET.embed(klimaLoader.df, cache_path="../data/embeddings/Sentence_Klima.csv")
filtered = klimaMPNET.iloc[:, 3:].copy()
filtered["extra_column"] = klimaMPNET.iloc[:, 2].values
analyzer = ProjectionAnalyzer(matrix_concept=sentimentMPNET, matrix_project=filtered)
analyzer.project()

# Add sentiment:
result = klimaMPNET.iloc[:, :3].copy()
result["sentiment"] = analyzer.projected_in_1D

# %%
# Extract name and party from Speaker_name
result[["Speaker", "Party"]] = result["Speaker_name"].str.extract(r"^(.*?) \(([A-Z]{1,3})\)$")
unique_parties = result["Party"].unique()
print(unique_parties)
party_map = {
    "S": "Socialdemokratiet",
    "V": "Venstre",
    "DD": "Danmarksdemokraterne",
    "SF": "Socialistisk Folkeparti",
    "LA": "Liberal Alliance",
    "M": "Moderaterne",
    "KF": "Det Konservative Folkeparti",
    "EL": "Enhedslisten",
    "DF": "Dansk Folkeparti",
    "RV": "Radikale Venstre",
    "ALT": "Alternativet",
    "BP": "Borgernes Parti",
    "N": "Naleraq",
    "IA": "Inuit Ataqatigiit",
    "SP": "Sambandsflokkurin",
    "JF": "Javnaðarflokkurin",
    "NB": "Nye Borgerlige",
    "KRF": "Kristeligt Folkeparti",
    None: None
}
result["Party_full"] = result["Party"].map(party_map).fillna("Ukendt")
result

# %%
# Filter rows where Party_full is "Ukendt"
unknown_speakers = result[result["Party_full"] == "Ukendt"]

# Count unique Speaker_name occurrences
speaker_counts = unknown_speakers["Speaker_name"].value_counts()
# As DataFrame
speaker_counts_df = speaker_counts.reset_index()
speaker_counts_df.columns = ["Speaker_name", "Count"]
speaker_counts_df

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.lineplot(data=result, x="Year", y="sentiment", marker='o')
plt.title("Sentiment Over Time")
plt.xlabel("Year")
plt.ylabel("Sentiment Score")
plt.grid(True)
plt.tight_layout()
plt.show()



# %%
