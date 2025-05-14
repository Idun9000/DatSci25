#%%
import pandas as pd
# read csv file
read_csv = pd.read_csv('../data/embeddingsDanishBertBotxo_Klima.csv')
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

### Plot the average PCA results by year ### 

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



