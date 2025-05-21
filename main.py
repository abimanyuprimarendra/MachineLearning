import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# SET PAGE CONFIG HARUS DI BARIS PERTAMA SEBELUM KODE LAIN
st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    if 'description' in data.columns:
        data['description'] = data['description'].fillna('')
    else:
        data['description'] = ''
    data['combined'] = data['title'] + " " + data['listed_in'] + " " + data['description']
    return data

@st.cache_data(show_spinner=False)
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix

def cosine_distance(vec1, vec2):
    # vec1 dan vec2 adalah numpy arrays
    return 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def get_content_based_recommendations_with_scores(title, cosine_sim, df, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = [(df['title'].iloc[i], score) for i, score in sim_scores]
    return recommended

def get_knn_recommendations_with_scores_manual(title, df, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    target_vector = tfidf_matrix[idx].toarray()[0]  # convert sparse to dense

    distances = []
    for i in range(tfidf_matrix.shape[0]):
        if i == idx:
            continue
        vec = tfidf_matrix[i].toarray()[0]
        dist = cosine_distance(target_vector, vec)
        distances.append((i, dist))
    distances = sorted(distances, key=lambda x: x[1])  # kecil = dekat

    top_neighbors = distances[:top_n]
    # Ubah distance jadi similarity dengan (1 - dist)
    recommended = [(df['title'].iloc[i], 1 - dist) for i, dist in top_neighbors]
    return recommended

def measure_avg_time(func, title, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        func(title)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / runs
    return avg_time

def plot_similarity_bar_chart(recommendations, method_name):
    titles = [rec[0] for rec in recommendations]
    scores = [rec[1] for rec in recommendations]

    fig, ax = plt.subplots()
    ax.barh(titles[::-1], scores[::-1], color='skyblue')
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top 5 Rekomendasi berdasarkan {method_name}")
    plt.tight_layout()
    st.pyplot(fig)

st.title("Sistem Rekomendasi Film Netflix")

# Load data dan buat tfidf matrix
df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Pilihan film dari dataset
title = st.selectbox("Pilih judul film untuk direkomendasikan:", options=df['title'].sort_values().unique())

if title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_cosine = measure_avg_time(lambda t=title: get_content_based_recommendations_with_scores(t, cosine_sim, df), title)
        avg_time_knn_manual = measure_avg_time(lambda t=title: get_knn_recommendations_with_scores_manual(t, df, tfidf_matrix), title)

        st.write(f"Rata-rata waktu eksekusi Cosine Similarity: **{avg_time_cosine:.5f} detik**")
        st.write(f"Rata-rata waktu eksekusi KNN Manual: **{avg_time_knn_manual:.5f} detik**")

        cosine_recs = get_content_based_recommendations_with_scores(title, cosine_sim, df)
        knn_manual_recs = get_knn_recommendations_with_scores_manual(title, df, tfidf_matrix)

        st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{title}':")
        for rec_title, score in cosine_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        st.subheader(f"Rekomendasi berdasarkan KNN Manual (Cosine Distance) untuk '{title}':")
        for rec_title, score in knn_manual_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        # Visualisasi bar chart
        st.subheader(f"Visualisasi Similarity Scores berdasarkan Cosine Similarity:")
        plot_similarity_bar_chart(cosine_recs, "Cosine Similarity")

        st.subheader(f"Visualisasi Similarity Scores berdasarkan KNN Manual:")
        plot_similarity_bar_chart(knn_manual_recs, "KNN Manual")
