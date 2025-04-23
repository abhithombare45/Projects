import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def cluster_posts(df, n_clusters=5):
    """
    Cluster posts based on TF-IDF vectors using KMeans.
    df: DataFrame with 'Title_Clean' column
    n_clusters: Number of clusters to form
    """
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df["Title_Clean"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    df["Cluster"] = clusters
    return df


if __name__ == "__main__":
    df = pd.read_csv("../processed_posts.csv")
    df = cluster_posts(df, n_clusters=5)
    df.to_csv("../clustered_posts.csv", index=False)
    print("Clustering complete, saved to clustered_posts.csv")
