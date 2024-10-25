# 🎵 Music Mood Clustering

## 📚 Table of Contents
1. [📋 Project Overview](#project-overview)
2. [📌 Introduction](#introduction)
3. [🎯 Objective](#objective)
4. [📊 Dataset Overview](#dataset-overview)
5. [🕵️ Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [📝 Methodology](#methodology)
7. [⚙️ Model Development](#model-development)
8. [📈 Evaluation](#evaluation)
9. [🏆 Results](#results)
10. [🔚 Conclusion](#conclusion)
11. [🚀 Future Work](#future-work)

## 📋 Project Overview
The goal of this project is to develop a machine learning model that clusters music tracks based on their mood using key audio features. This enables personalized music recommendations and playlists for users, enhancing their listening experience.

#### 🚀 Key Benefits:
- **🎧 Personalized Playlists**: Tailors music suggestions based on user emotions, improving engagement.
- **🗂️ Content Curation**: Helps platforms organize and discover music faster.
- **💆 Music Therapy**: Supports wellness applications by offering mood-based playlists.
- **🎯 Targeted Marketing**: Enables mood-driven advertising for better audience targeting.
- **🔍 Creative Insights**: Provides artists and labels with valuable audience mood trends.

## 📌 Introduction
Music is deeply connected to human emotions, often influencing and reflecting our moods. With access to rich metadata from streaming platforms like Spotify, we can analyze songs based on their musical features to discover patterns and group them by mood. This project uses unsupervised learning to cluster songs based on their attributes, providing insight into how various musical qualities like energy, danceability, and valence relate to different moods.

## 🎯 Objective
The goal of this project is to use unsupervised clustering, specifically K-Means, to group songs from the Spotify dataset based on their musical features.

## Main Objectives:
- **🔍 Perform Exploratory Data Analysis (EDA)** on the dataset to understand the distribution and relationships between various features, such as danceability, energy, and valence.
- **🛠️ Preprocess the dataset** by handling missing values, standardizing features, and addressing skewness where necessary.
- **⚙️ Apply clustering algorithms** like K-Means to group songs into distinct clusters based on musical attributes.
- **📊 Analyze and interpret** the resulting clusters by examining the feature distribution within each cluster and associating them with potential musical moods (e.g., happy, sad, energetic, calm).
- **📈 Evaluate the clustering results** and refine the model by adjusting hyperparameters and handling class imbalances, if necessary.
- **💡 Draw insights and conclusions** from the clusters to provide a better understanding of how various musical features contribute to the mood or emotional content of a song.

## 📊Dataset Overview
The dataset is sourced from Hugging Face's open datasets platform and can be accessed [here](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).
Or, Access from my [Google Drive](https://drive.google.com/file/d/10JsSQ74dAH-J9J2Yuq9-ZksYTgYRD1dl/view) and contains key audio features for each track, such as:

- **🕺 Danceability**
- **⚡ Energy**
- **😄 Valence**
- **🕒 Tempo**
- **🎸 Acousticness**
- **🔊 Loudness**
- **🎤 Speechiness**
More details about the dataset can be found in the `data` folder in the repository.

## 🕵️ Exploratory Data Analysis (EDA)
In this phase, we explore the dataset to understand the relationships between various features, identify patterns, and prepare the dataset for model development. EDA steps include:

- **📈 Distribution Analysis**: Visualize distributions of key features like energy, danceability, and valence.
- **🔗 Correlation Analysis**: Understand feature relationships.
- **⚠️ Outlier Detection**: Identify and handle outliers if necessary.

## 📝 Methodology
- **1.📊 Data Preprocessing**: Handle missing values, standardize features, and address skewness.
- **2.⚙️ Clustering with K-Means**: Group songs into clusters using K-Means clustering.
- **3.📈 Evaluation**: Analyze cluster quality and adjust hyperparameters as needed.
- **4.🎨 Mood Classification**: Assign mood labels based on cluster characteristics.

## ⚙️ Model Development
The model used for clustering is the K-Means algorithm. Key steps include:

-**🏗️ Model Fitting and Prediction**: Fitting the K-Means model to the standardized dataset and assigning clusters.

## 📈 Evaluation
The quality of clustering is evaluated using metrics such as:

- **🖋️ Silhouette Score**: To measure how similar each point is to its own cluster compared to other clusters.
- **📊 Cluster Distribution Analysis**: Analyzing the spread and density of clusters.

## 🏆 Results
After clustering, songs are grouped into distinct mood-based clusters. Each cluster is analyzed to identify common patterns in features like energy, danceability, and valence. These insights can be used to create mood-based playlists or drive recommendation algorithms.

## 🔚 Conclusion
The project successfully demonstrates the application of unsupervised learning to cluster songs based on key audio features, thereby providing personalized and mood-based music recommendations. By leveraging these insights, music streaming platforms can enhance user experiences and cater to individual preferences.

## 🚀 Future Work
- **📚 Incorporating More Features**: Add additional musical features or metadata for better clustering accuracy.
- **🧠 Advanced Clustering Algorithms**: Experiment with other unsupervised learning algorithms like DBSCAN or Gaussian Mixture Models.
- **😄 Mood Prediction**: Train a supervised learning model to predict the mood of a new song based on its features.


