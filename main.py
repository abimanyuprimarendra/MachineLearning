import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['description'] = data['description'].fillna('') if 'description' in data.columns else ''
    data['combined'] = data['title'] + " " + data['listed_in'] + " " + data['description']
    return data

@st.cache_data
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix

@st.cache_resource
def create_knn_model():
    return NearestNeighbors(metric='cosine', algorithm='brute')

def get_content_based_recommendations(title, cosine_sim, df, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    return [(df['title'].iloc[i], score) for i, score in sim_scores]

def get_knn_recommendations(title, knn_model, df, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    return [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]

def plot_similarity_comparison(recs1, recs2, title1, title2, method_name):
    # Gabungkan kedua rekomendasi berdasar judul film
    df1 = pd.DataFrame(recs1, columns=['title', f'similarity_{title1}'])
    df2 = pd.DataFrame(recs2, columns=['title', f'similarity_{title2}'])
    merged = pd.merge(df1, df2, on='title', how='outer').fillna(0)

    plt.figure(figsize=(8,4))
    sns.barplot(data=merged.melt(id_vars='title'), x='title', y='value', hue='variable')
    plt.title(f'Perbandingan Skor Similarity Rekomendasi ({method_name})')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Similarity Score')
    plt.xlabel('Film Rekomendasi')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

st.title("Perbandingan Rekomendasi Film untuk Dua Pilihan Film")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix)

knn_model = create_knn_model()
knn_model.fit(tfidf_matrix)

title1 = st.selectbox("Pilih film pertama:", options=df['title'].sort_values().unique(), key='title1')
title2 = st.selectbox("Pilih film kedua:", options=df['title'].sort_values().unique(), key='title2')

if title1 and title2:
    if title1 == title2:
        st.warning("Silakan pilih dua film yang berbeda untuk dibandingkan.")
    else:
        cosine_recs1 = get_content_based_recommendations(title1, cosine_sim, df)
        cosine_recs2 = get_content_based_recommendations(title2, cosine_sim, df)
        
        knn_recs1 = get_knn_recommendations(title1, knn_model, df, tfidf_matrix)
        knn_recs2 = get_knn_recommendations(title2, knn_model, df, tfidf_matrix)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Rekomendasi Cosine Similarity untuk '{title1}':")
            for t, s in cosine_recs1:
                st.write(f"- {t} (similarity: {s:.4f})")

            st.subheader(f"Rekomendasi KNN untuk '{title1}':")
            for t, s in knn_recs1:
                st.write(f"- {t} (similarity: {s:.4f})")

        with col2:
            st.subheader(f"Rekomendasi Cosine Similarity untuk '{title2}':")
            for t, s in cosine_recs2:
                st.write(f"- {t} (similarity: {s:.4f})")

            st.subheader(f"Rekomendasi KNN untuk '{title2}':")
            for t, s in knn_recs2:
                st.write(f"- {t} (similarity: {s:.4f})")

        st.subheader("Film yang direkomendasikan oleh kedua film (Cosine Similarity):")
        common_cosine = set([t for t, _ in cosine_recs1]) & set([t for t, _ in cosine_recs2])
        if common_cosine:
            for film in common_cosine:
                st.write(f"- {film}")
        else:
            st.write("Tidak ada rekomendasi yang sama untuk Cosine Similarity.")

        st.subheader("Film yang direkomendasikan oleh kedua film (KNN):")
        common_knn = set([t for t, _ in knn_recs1]) & set([t for t, _ in knn_recs2])
        if common_knn:
            for film in common_knn:
                st.write(f"- {film}")
        else:
            st.write("Tidak ada rekomendasi yang sama untuk KNN.")

        # Visualisasi perbandingan skor similarity
        plot_similarity_comparison(cosine_recs1, cosine_recs2, title1, title2, "Cosine Similarity")
        plot_similarity_comparison(knn_recs1, knn_recs2, title1, title2, "KNN")

        # Kelebihan masing-masing metode
        st.subheader("Kelebihan Metode Cosine Similarity")
        st.write("""
        - Menghitung kesamaan secara langsung menggunakan representasi vektor TF-IDF.
        - Cepat dan efisien untuk dataset dengan fitur teks.
        - Memberikan skor similarity yang mudah diinterpretasikan (nilai antara 0 dan 1).
        - Lebih sederhana, cocok untuk konten dengan teks kaya seperti deskripsi film.
        """)

        st.subheader("Kelebihan Metode KNN")
        st.write("""
        - Menggunakan algoritma yang lebih fleksibel, dapat digunakan untuk berbagai jenis data.
        - Memperhitungkan jarak tetangga terdekat secara eksplisit.
        - Cocok untuk dataset dengan ukuran besar jika dioptimalkan.
        - Mudah untuk dikembangkan ke metode lain seperti k-means clustering atau rekomendasi berbasis pengguna.
        """)

