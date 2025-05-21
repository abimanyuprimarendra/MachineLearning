import streamlit as st
import pandas as pd
import numpy as np
import heapq
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# 1. Load dataset
@st.cache_data
def load_data():
    file_path = '/content/drive/MyDrive/Semester 6/Machine Learning 2025/Netflix.csv'  # ganti jika perlu
    df = pd.read_csv(file_path)
    features = ['title', 'listed_in'] + (['description'] if 'description' in df.columns else [])
    df[features] = df[features].fillna('')
    df['combined'] = df[features].agg(' '.join, axis=1)
    return df

df = load_data()

# 2. TF-IDF vectorization
@st.cache_resource
def create_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined']).astype(np.float32)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn_model.fit(tfidf_matrix)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return tfidf_matrix, cosine_sim, knn_model, indices

tfidf_matrix, cosine_sim, knn_model, indices = create_model(df)

# 3. Content-Based Recommendation
def get_content_based_recommendations(title, num_recommendations=10):
    if title not in indices:
        return ["Judul tidak ditemukan di dataset."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    top_sim_scores = heapq.nlargest(num_recommendations + 1, sim_scores, key=lambda x: x[1])
    top_sim_scores = [item for item in top_sim_scores if item[0] != idx][:num_recommendations]

    return [f"{df['title'].iloc[i]} (Cosine Similarity: {score:.4f})" for i, score in top_sim_scores]

# 4. KNN Recommendation
def get_knn_recommendations(title, num_recommendations=10):
    if title not in indices:
        return ["Judul tidak ditemukan di dataset."]
    
    idx = indices[title]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=num_recommendations + 1)
    
    recommended = []
    for i, dist in zip(indices_knn.flatten()[1:], distances.flatten()[1:]):
        similarity = 1 - dist
        recommended.append(f"{df['title'].iloc[i]} (Similarity: {similarity:.4f})")

    return recommended

# 5. Streamlit UI
st.set_page_config(page_title="Sistem Rekomendasi Film Netflix", layout="centered")
st.title("🎬 Sistem Rekomendasi Film Netflix")
st.write("Sistem ini menggunakan **Content-Based Filtering** dengan algoritma **Cosine Similarity** dan **KNN**.")

judul_film = st.text_input("Masukkan judul film (case-sensitive)", "")

if judul_film:
    st.subheader("📌 Rekomendasi Berdasarkan Cosine Similarity:")
    for rec in get_content_based_recommendations(judul_film):
        st.write("•", rec)

    st.subheader("📌 Rekomendasi Berdasarkan KNN:")
    for rec in get_knn_recommendations(judul_film):
        st.write("•", rec)
