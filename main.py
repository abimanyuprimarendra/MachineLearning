import streamlit as st
import pandas as pd
import numpy as np
import heapq
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

@st.cache_data(show_spinner=False)
def load_data():
    file_path = '/content/drive/MyDrive/Semester 6/Machine Learning 2025/Netflix.csv'
    df = pd.read_csv(file_path)
    features = ['title', 'listed_in'] + (['description'] if 'description' in df.columns else [])
    df[features] = df[features].fillna('')
    df['combined'] = df[features].agg(' '.join, axis=1)
    return df

@st.cache_data(show_spinner=False)
def build_tfidf_and_knn(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined']).astype(np.float32)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn_model.fit(tfidf_matrix)
    return tfidf_matrix, knn_model

df = load_data()
tfidf_matrix, knn_model = build_tfidf_and_knn(df)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_content_based_recommendations(title, num_recommendations=10):
    if title not in indices:
        return "Judul tidak ditemukan di dataset."
    idx = indices[title]
    cosine_sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten().astype(np.float32)
    sim_scores = list(enumerate(cosine_sim_row))
    top_sim_scores = heapq.nlargest(num_recommendations + 1, sim_scores, key=lambda x: x[1])
    top_sim_scores = [item for item in top_sim_scores if item[0] != idx][:num_recommendations]
    recommended = [f"{df['title'].iloc[i]} (cosine similarity: {score:.4f})" for i, score in top_sim_scores]
    return recommended

def get_knn_recommendations(title, num_recommendations=10):
    if title not in indices:
        return "Judul tidak ditemukan di dataset."
    idx = indices[title]
    distances, neighbors = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
    distances = distances.flatten()[1:]
    neighbors = neighbors.flatten()[1:]
    recommended = []
    for i, dist in zip(neighbors, distances):
        similarity = 1 - dist
        recommended.append(f"{df['title'].iloc[i]} (similarity: {similarity:.4f})")
    return recommended

st.title("Rekomendasi Film Netflix")

title_input = st.text_input("Masukkan judul film:", "Dick Johnson Is Dead")
num_recs = st.slider("Jumlah rekomendasi:", 1, 20, 10)

if title_input:
    start = time.time()
    content_recs = get_content_based_recommendations(title_input, num_recs)
    end = time.time()
    st.write(f"Waktu eksekusi Content-Based (Cosine Similarity): {end - start:.5f} detik")

    start = time.time()
    knn_recs = get_knn_recommendations(title_input, num_recs)
    end = time.time()
    st.write(f"Waktu eksekusi KNN: {end - start:.5f} detik")

    st.subheader(f"Rekomendasi berdasarkan Content-Based untuk '{title_input}':")
    if isinstance(content_recs, list):
        for rec in content_recs:
            st.write(rec)
    else:
        st.write(content_recs)

    st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title_input}':")
    if isinstance(knn_recs, list):
        for rec in knn_recs:
            st.write(rec)
    else:
        st.write(knn_recs)
