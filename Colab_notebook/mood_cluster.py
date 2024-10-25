import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import joblib
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer




df = pd.read_csv('/content/spotify_tracks_dataset-new.csv')

print("Shape of the dataset:", df.shape)
df.head()

print("Data Types:\n")
df.dtypes

print("Descriptive Statistics:\n")
df.describe()

mood_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness']
df_mood = df[mood_features]

df_mood.isnull().sum()

df_mood.duplicated().sum()

df_mood = df_mood.drop_duplicates(keep='first')
df = df.loc[df_mood.index]

print(f'Number of duplicates removed from df_mood: {df_mood.shape[0] - df.shape[0]}')
print(f'Shape of df_mood_cleaned: {df_mood.shape}')
print(f'Shape of df: {df.shape}')

correlation_matrix = df_mood.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

for feature in mood_features:
    plt.figure(figsize=(6, 4))
    df_mood[feature].hist(bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

for feature in mood_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_mood[feature])
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.show()

def remove_outliers(df_mood, df, features):
    df_filter_mood = df_mood.copy()
    df_filter_other = df.copy()

    for feature in features:
        Q1 = df_mood[feature].quantile(0.25)
        Q3 = df_mood[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_mood[(df_mood[feature] < lower_bound) | (df_mood[feature] > upper_bound)]
        print(f'Number of outliers in {feature}: {len(outliers)}')

        df_filter_mood = df_filter_mood[(df_filter_mood[feature] >= lower_bound) & (df_filter_mood[feature] <= upper_bound)]
        df_filter_other = df_filter_other.loc[df_filter_mood.index]


    return df_filter_mood, df_filter_other

df_mood_cleaned, df_cleaned = remove_outliers(df_mood, df, mood_features)

print("Shape before removing outliers in df_mood:", df_mood.shape)
print("Shape before removing outliers in df:", df.shape)
print("Shape after removing outliers in df_mood_cleaned:", df_mood_cleaned.shape)
print("Shape after removing outliers in df_cleaned:", df_cleaned.shape)

for feature in mood_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_mood_cleaned[feature])
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.show()

for feature in mood_features:
  skewness = df_mood_cleaned[feature].skew()
  print(f'Skewness of {feature} : {skewness}')

skewed_features = ['instrumentalness', 'liveness', 'acousticness', 'loudness']
for feature in skewed_features:
    if (df_mood_cleaned[feature] <= 0).any():
        shift_value = abs(df_mood_cleaned[feature].min()) + 1
        transformed_data = df_mood_cleaned[feature] + shift_value
    else:
        transformed_data = df_mood_cleaned[feature]

    boxcox_transformed, _ = boxcox(transformed_data)

    boxcox_skew = pd.Series(boxcox_transformed).skew()

    print(f"Skewness for {feature}:")
    print(f"  Original: {df_mood_cleaned[feature].skew()}")
    print(f"  After Box-Cox Transformation: {boxcox_skew}")
    print()

quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=42)
df_mood_cleaned['instrumentalness_quantile'] = quantile_transformer.fit_transform(df_mood_cleaned[['instrumentalness']])
print("Skewness after Quantile transformation:", pd.Series(df_mood_cleaned['instrumentalness_quantile']).skew())

plt.figure(figsize=(8,6))
sns.kdeplot(x=df['danceability'], y=df['energy'], cmap="Blues", fill=True)
plt.title('Density Plot of Danceability vs Energy')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['loudness'], y=df['valence'], hue=df['energy'], palette="coolwarm")
plt.title('Loudness vs Valence (Emotion)')
plt.xlabel('Loudness (dB)')
plt.ylabel('Valence (Emotional Positivity)')
plt.show()

plt.figure(figsize=(8,6))
sns.histplot(df['tempo'], bins=30, kde=True, color='purple')
plt.title('Distribution of Track Tempo (BPM)')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['acousticness'], y=df['instrumentalness'], hue=df['liveness'], palette="viridis")
plt.title('Instrumentalness vs Acousticness')
plt.xlabel('Acousticness')
plt.ylabel('Instrumentalness')
plt.legend(loc='upper right')
plt.show()

df['liveness_bin'] = pd.cut(df['liveness'], bins=[0, 0.3, 0.6, 1], labels=['Low', 'Medium', 'High'])

plt.figure(figsize=(8,6))
sns.boxplot(x='liveness_bin', y='danceability', data=df)
plt.title('Danceability by Liveness (Live Audience Presence)')
plt.xlabel('Liveness (Low/Medium/High)')
plt.ylabel('Danceability')
plt.show()

df_mood_cleaned.to_csv('Spotify_preprocessed.csv')

pca = PCA(n_components=2)
df_mood_cleaned_pca = pca.fit_transform(df_mood_cleaned)

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, random_state=42)
kmeans.fit(df_mood_cleaned_pca)

cluster_labels = kmeans.labels_

silhouette_avg = silhouette_score(df_mood_cleaned, cluster_labels)

print(f'Silhouette Score: {silhouette_avg:.4f}')

df_mood_cleaned['Cluster'] = cluster_labels

df_mood_cleaned['Cluster'].value_counts()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_mood_cleaned['danceability'],
                y=df_mood_cleaned['energy'],
                hue=cluster_labels,
                palette='viridis',
                alpha=0.7)
plt.title('Clusters in Danceability vs Energy')
plt.xlabel('Danceability')
plt.ylabel('Energy')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()

selected_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
df_pair = df_mood_cleaned[selected_features].copy()
df_pair['Cluster'] = cluster_labels

sns.pairplot(df_pair, hue='Cluster', palette='viridis', markers='o', diag_kind='kde')
plt.suptitle('Pair Plot of Features Colored by Cluster', y=1.02)
plt.show()

pca = PCA(n_components=3)
pca_components = pca.fit_transform(df_mood_cleaned[mood_features])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], pca_components[:, 2],
                     c=df_mood_cleaned['Cluster'], cmap='viridis')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D Clusters Visualization')

cbar = fig.colorbar(scatter, ax=ax, label='Cluster')

plt.show()

pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_mood_cleaned[mood_features])

plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df_mood_cleaned['Cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualization')
plt.colorbar(label='Cluster')
plt.show()

cluster_summary = df_mood_cleaned.groupby('Cluster').mean()
cluster_summary

mood_labels = {
    0: 'Calm/Relaxed',
    1: 'Happy/Energetic',
    2: 'Sad/Reflective',
    3: 'Angry/Intense'
}

df_mood_cleaned['Mood'] = df_mood_cleaned['Cluster'].map(mood_labels)

df_cleaned[['Cluster', 'Mood']] = df_mood_cleaned[['Cluster', 'Mood']]

df_cleaned.describe()

joblib.dump(pca, 'mood_clustering_pca.pkl')
joblib.dump(kmeans, 'mood_clustering_kmeans.pkl')

mood_features = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness']

pca = PCA(n_components=2)
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, random_state=42)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('pca', pca),
    ('kmeans', kmeans)
])

pipeline.fit(df_mood_cleaned[mood_features])
joblib.dump(pipeline, 'mood_clustering_pipeline.pkl')
