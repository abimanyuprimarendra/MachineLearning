import streamlit as st
import pandas as pd
import numpy as np
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# 1. Load dataset dari Google Drive
@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1lto09pdlh825Gv0TfBUkgk1e2JVQW19c"
    df = pd.read_csv(csv_url)
    features = ['title', 'listed_in'] + (['description'] if 'description' in df.columns else [])
    df[features] = df[features].fillna('')
    df['combined'] = df[features].agg(' '.join, axis=1)
    return df

df = load_data_from_drive()

# 2. TF-IDF + Model Cosine & KNN
@st.cache_resource
def create_models(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined']).astype(np.float32)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn_model.fit(tfidf_matrix)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    return tfidf_matrix, cosine_sim, knn_model, indices

tfidf_matrix, cosine_sim, knn_model, indices = create_models(df)

# 3. Rekomendasi Content-Based
def get_content_based_recommendations(title, num_recommendations=10):
    if title not in indices:
        return ["❌ Judul tidak ditemukan di dataset."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    top_sim_scores = heapq.nlargest(num_recommendations + 1, sim_scores, key=lambda x: x[1])
    top_sim_scores = [item for item in top_sim_scores if item[0] != idx][:num_recommendations]

    return [f"{df['title'].iloc[i]} (Cosine Similarity: {score:.4f})" for i, score in top_sim_scores]

# 4. Rekomendasi KNN
def get_knn_recommendations(title, num_recommendations=10):
    if title not in indices:
        return ["❌ Judul tidak ditemukan di dataset."]
    
    idx = indices[title]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=num_recommendations + 1)
    
    recommended = []
    for i, dist in zip(indices_knn.flatten()[1:], distances.flatten()[1:]):
        similarity = 1 - dist
        recommended.append(f"{df['title'].iloc[i]} (Similarity: {similarity:.4f})")

    return recommended

# 5. Tampilan Streamlit
st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")
st.title("🎬 Sistem Rekomendasi Film Netflix")
st.markdown("Model ini menggunakan **Content-Based Filtering** dengan algoritma **Cosine Similarity** dan **K-Nearest Neighbors (KNN)** berdasarkan deskripsi dan genre film.")

judul_film = st.text_input("Masukkan judul film (perhatikan huruf kapital):", "")

if judul_film:
    st.subheader("📌 Rekomendasi Berdasarkan Cosine Similarity:")
    for rec in get_content_based_recommendations(judul_film):
        st.write("•", rec)

    st.subheader("📌 Rekomendasi Berdasarkan KNN:")
    for rec in get_knn_recommendations(judul_film):
        st.write("•", rec)
