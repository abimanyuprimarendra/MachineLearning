import streamlit as st
import pandas as pd
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")
os.environ['STREAMLIT_ENV'] = 'development'

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    df = pd.read_csv(csv_url)

    # Isi kosong diganti string kosong
    df['title'] = df['title'].fillna('').astype(str)
    df['listed_in'] = df['listed_in'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str) if 'description' in df.columns else ''
    
    # Hapus duplikat dan gabungkan
    df.drop_duplicates(subset='title', inplace=True)
    df['combined'] = df['title'] + " " + df['listed_in'] + " " + df['description']
    return df

@st.cache_data(show_spinner=False)
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix

@st.cache_resource(show_spinner=False)
def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

def get_content_based_recommendations_with_scores(title, cosine_sim, df, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended = [(df.iloc[i]['title'], score) for i, score in sim_scores]
    return recommended

def get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended = [(df.iloc[i]['title'], 1 - dist) for i, dist in zip(indices.flatten()[1:], distances.flatten()[1:])]
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
    ax.set_title(f"Top 5 Rekomendasi berdasarkan {method_name}")
    plt.tight_layout()
    st.pyplot(fig)

# Judul aplikasi
st.title("🎬 Sistem Rekomendasi Film Netflix")

# Load data
df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix)
knn_model = create_knn_model(tfidf_matrix)

# Pilih judul
title = st.selectbox("Pilih judul film untuk direkomendasikan:", options=df['title'].sort_values().unique())

if title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_cosine = measure_avg_time(lambda t=title: get_content_based_recommendations_with_scores(t, cosine_sim, df), title)
        avg_time_knn = measure_avg_time(lambda t=title: get_knn_recommendations_with_scores(t, knn_model, df, tfidf_matrix), title)

        st.write(f"⏱️ Rata-rata waktu eksekusi *Cosine Similarity*: **{avg_time_cosine:.5f} detik**")
        st.write(f"⏱️ Rata-rata waktu eksekusi *KNN*: **{avg_time_knn:.5f} detik**")

        cosine_recs = get_content_based_recommendations_with_scores(title, cosine_sim, df)
        knn_recs = get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix)

        st.subheader(f"🔍 Rekomendasi berdasarkan Cosine Similarity untuk '{title}':")
        for rec_title, score in cosine_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        st.subheader(f"🤖 Rekomendasi berdasarkan KNN untuk '{title}':")
        for rec_title, score in knn_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        # Visualisasi
        st.subheader(f"📊 Visualisasi Similarity Scores (Cosine Similarity):")
        plot_similarity_bar_chart(cosine_recs, "Cosine Similarity")

        st.subheader(f"📊 Visualisasi Similarity Scores (KNN):")
        plot_similarity_bar_chart(knn_recs, "KNN")
