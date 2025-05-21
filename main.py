import streamlit as st
import pandas as pd
import numpy as np
import time
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# SET PAGE CONFIG HARUS DI BARIS PERTAMA SEBELUM KODE LAIN
st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")

@st.cache_data
def load_data():
    # Ganti url ini sesuai file kamu di Google Drive (pastikan shared link bisa diakses publik)
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    df = pd.read_csv(csv_url)

    features = ['title', 'listed_in'] + (['description'] if 'description' in df.columns else [])
    df[features] = df[features].fillna('')
    df['combined'] = df[features].agg(' '.join, axis=1)

    return df

@st.cache_data
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df['combined']).astype(np.float32)

@st.cache_resource
def create_knn_model(tfidf_matrix):
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn.fit(tfidf_matrix)
    return knn

def get_content_based_recommendations(title, df, tfidf_matrix, indices, num_recommendations=10):
    if title not in indices:
        return "Judul tidak ditemukan di dataset."

    idx = indices[title]
    cosine_sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten().astype(np.float32)
    sim_scores = list(enumerate(cosine_sim_row))
    top_sim_scores = heapq.nlargest(num_recommendations + 1, sim_scores, key=lambda x: x[1])
    top_sim_scores = [item for item in top_sim_scores if item[0] != idx][:num_recommendations]

    recommended = [(df['title'].iloc[i], score) for i, score in top_sim_scores]
    return recommended

def get_knn_recommendations(title, df, tfidf_matrix, knn_model, indices, num_recommendations=10):
    if title not in indices:
        return "Judul tidak ditemukan di dataset."

    idx = indices[title]
    distances, neighbors = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
    distances = distances.flatten()[1:]
    neighbors = neighbors.flatten()[1:]

    recommended = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(neighbors, distances)]
    return recommended

def measure_avg_time(func, title, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        func(title)
        end = time.time()
        times.append(end - start)
    return sum(times) / runs

def plot_similarity_bar_chart(recommendations, method_name):
    titles = [rec[0] for rec in recommendations]
    scores = [rec[1] for rec in recommendations]

    fig, ax = plt.subplots()
    ax.barh(titles[::-1], scores[::-1], color='skyblue')
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top {len(recommendations)} Rekomendasi berdasarkan {method_name}")
    plt.tight_layout()
    st.pyplot(fig)

# MAIN STREAMLIT APP
st.title("Sistem Rekomendasi Film Netflix")

# Load dataset dan buat tfidf matrix + knn model
df = load_data()
tfidf_matrix = create_tfidf_matrix(df)
knn_model = create_knn_model(tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

title = st.selectbox("Pilih judul film:", options=df['title'].sort_values().unique())

if title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_cosine = measure_avg_time(lambda t=title: get_content_based_recommendations(t, df, tfidf_matrix, indices), title)
        avg_time_knn = measure_avg_time(lambda t=title: get_knn_recommendations(t, df, tfidf_matrix, knn_model, indices), title)

        st.write(f"Rata-rata waktu eksekusi Content-Based (Cosine Similarity): **{avg_time_cosine:.5f} detik**")
        st.write(f"Rata-rata waktu eksekusi KNN: **{avg_time_knn:.5f} detik**")

        cosine_recs = get_content_based_recommendations(title, df, tfidf_matrix, indices)
        knn_recs = get_knn_recommendations(title, df, tfidf_matrix, knn_model, indices)

        st.subheader(f"Rekomendasi berdasarkan Content-Based untuk '{title}':")
        for rec_title, score in cosine_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title}':")
        for rec_title, score in knn_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        st.subheader("Visualisasi Similarity Scores (Content-Based):")
        plot_similarity_bar_chart(cosine_recs, "Content-Based (Cosine Similarity)")

        st.subheader("Visualisasi Similarity Scores (KNN):")
        plot_similarity_bar_chart(knn_recs, "KNN")
