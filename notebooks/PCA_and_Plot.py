#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#%%
# read csv file
# df = pd.read_csv('../data/embeddings/DanishBertBotxo_Klima.csv')
# df = pd.read_csv('../data/embeddings/Instructor_Klima.csv')
df = pd.read_csv('../data/embeddings/Sentence_Klima.csv')

# %%
# Perform PCA on the embeddings and polot first two components
# Select embeddings
embeddings = df.iloc[:, 4:]

# Perform PCA with 2 components
pca = PCA(n_components=10)
pca_result = pca.fit_transform(embeddings)

# Get variance explained
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by PC1, PC2 and PC3: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}, {explained_variance[2]:.2%}")

# Create new DataFrame with original first 4 columns and PCA results
pca_df = df.iloc[:, :4].copy()
pca_df['PC1'] = pca_result[:, 0]
pca_df['PC2'] = pca_result[:, 1]
pca_df['PC3'] = pca_result[:, 2]

# Show the result
pca_df

# Sort by Month, then Year
pca_df_sorted = pca_df.sort_values(by=['Year']).reset_index(drop=True)

# Show the result
pca_df_sorted



# %%
### Plot the average PCA results by year ### 
# Group by Year and compute mean PC1 and PC2
mean_pca_by_year = pca_df_sorted.groupby('Year')[['PC1', 'PC2']].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mean_pca_by_year, x='PC1', y='PC2', hue='Year', palette='viridis', s=150)

# Labels and title
plt.xlabel('Mean Principal Component 1')
plt.ylabel('Mean Principal Component 2')
plt.title('Mean PCA Embedding per Year')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
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
# Calculate rolling window means (5-year window)
rolling_means = []
years = sorted(pca_df_sorted['Year'].unique())

for i in range(len(years) - 4):  # -4 to ensure we have 5 years for each window
    window_years = years[i:i+5]
    window_data = pca_df_sorted[pca_df_sorted['Year'].isin(window_years)]
    mean_pc1 = window_data['PC1'].mean()
    mean_pc2 = window_data['PC2'].mean()
    rolling_means.append({
        'Years': f"{window_years[0]}-{window_years[-1]}",
        'PC1': mean_pc1,
        'PC2': mean_pc2
    })

rolling_df = pd.DataFrame(rolling_means)

# Create plot with both individual years and rolling means
plt.figure(figsize=(12, 8))

# Plot individual years
sns.scatterplot(data=mean_pca_by_year, x='PC1', y='PC2', 
                hue='Year', palette='viridis', s=150)

# Plot rolling means line
plt.plot(rolling_df['PC1'], rolling_df['PC2'], 
         'r-', linewidth=2, label='5-year rolling mean')

# Add annotations
# for _, row in mean_pca_by_year.iterrows():
#    plt.text(row['PC1'] + 0.01, row['PC2'], str(row['Year']), fontsize=10)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Embedding: Yearly Means with 5-Year Rolling Average')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




# %%
