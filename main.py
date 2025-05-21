import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")

@st.cache_data
def load_data():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    df = pd.read_csv(csv_url)
    df['listed_in'] = df['listed_in'].fillna('')
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('')
    else:
        df['description'] = ''
    df['combined'] = df['title'] + " " + df['listed_in'] + " " + df['description']
    return df

@st.cache_resource(show_spinner=False)
def create_tfidf_and_knn(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined']).astype(np.float32)
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return tfidf_matrix, knn_model

@st.cache_resource(show_spinner=False)
def compute_cosine_sim_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

def get_content_based_recommendations(title, cosine_sim, df, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [(df['title'].iloc[i], score) for i, score in sim_scores]

def get_knn_recommendations(title, knn_model, df, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df.index[df['title'] == title][0]
    distances, indices_knn = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    return [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]

def measure_avg_time(func, title, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        func(title)
        times.append(time.time() - start)
    return np.mean(times)

def plot_bar_chart(recommendations, method):
    titles = [rec[0] for rec in recommendations]
    scores = [rec[1] for rec in recommendations]
    fig, ax = plt.subplots()
    ax.barh(titles[::-1], scores[::-1], color='skyblue')
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top {len(recommendations)} Rekomendasi berdasarkan {method}")
    plt.tight_layout()
    st.pyplot(fig)

# Main UI
df = load_data()
tfidf_matrix, knn_model = create_tfidf_and_knn(df)
cosine_sim = compute_cosine_sim_matrix(tfidf_matrix)

title = st.selectbox("Pilih judul film untuk direkomendasikan:", options=sorted(df['title'].unique()))

if title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_cosine = measure_avg_time(lambda t=title: get_content_based_recommendations(t, cosine_sim, df), title)
        avg_time_knn = measure_avg_time(lambda t=title: get_knn_recommendations(t, knn_model, df, tfidf_matrix), title)

        st.write(f"Rata-rata waktu eksekusi Cosine Similarity: **{avg_time_cosine:.5f} detik**")
        st.write(f"Rata-rata waktu eksekusi KNN: **{avg_time_knn:.5f} detik**")

        cosine_recs = get_content_based_recommendations(title, cosine_sim, df)
        knn_recs = get_knn_recommendations(title, knn_model, df, tfidf_matrix)

        if not cosine_recs or not knn_recs:
            st.warning("Judul film tidak ditemukan dalam dataset.")
        else:
            st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{title}':")
            for rec_title, score in cosine_recs:
                st.write(f"- {rec_title} (similarity: {score:.4f})")

            st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title}':")
            for rec_title, score in knn_recs:
                st.write(f"- {rec_title} (similarity: {score:.4f})")

            st.subheader(f"Visualisasi Similarity Scores berdasarkan Cosine Similarity:")
            plot_bar_chart(cosine_recs, "Cosine Similarity")

            st.subheader(f"Visualisasi Similarity Scores berdasarkan KNN:")
            plot_bar_chart(knn_recs, "KNN")
