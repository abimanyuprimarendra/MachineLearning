import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Harus diletakkan di paling awal, sebelum perintah Streamlit lain seperti st.title, st.write, dsb.
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

def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

def get_content_based_recommendations_with_scores(title, cosine_sim, df, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recommended = [(df['title'].iloc[i], score) for i, score in sim_scores]
    return recommended

def get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    recommended = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]
    return recommended

def measure_avg_time(func, title, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(title)
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / runs
    return avg_time

st.title("Sistem Rekomendasi Film Netflix")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
knn_model = create_knn_model(tfidf_matrix)

title = st.selectbox("Pilih judul film untuk direkomendasikan:", options=df['title'].sort_values().unique())

if title:
    with st.spinner("Menghitung rekomendasi..."):
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        avg_time_cosine = measure_avg_time(
            lambda t=title: get_content_based_recommendations_with_scores(t, cosine_sim, df, top_n=1), title)
        avg_time_knn = measure_avg_time(
            lambda t=title: get_knn_recommendations_with_scores(t, knn_model, df, tfidf_matrix, top_n=1), title)

        st.write(f"Rata-rata waktu eksekusi Cosine Similarity (1 rekomendasi): **{avg_time_cosine:.5f} detik**")
        st.write(f"Rata-rata waktu eksekusi KNN (1 rekomendasi): **{avg_time_knn:.5f} detik**")

        cosine_recs = get_content_based_recommendations_with_scores(title, cosine_sim, df, top_n=1)
        knn_recs = get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix, top_n=1)

        st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{title}':")
        for rec_title, score in cosine_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

        st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title}':")
        for rec_title, score in knn_recs:
            st.write(f"- {rec_title} (similarity: {score:.4f})")

    st.markdown("""
    ---
    ### Perbandingan Metode Rekomendasi

    Kedua metode menggunakan data dan film input yang sama, namun berbeda dalam cara pengukuran dan efisiensi:

    - **Cosine Similarity** menghitung skor kesamaan secara global antara semua film. Ini memberi gambaran menyeluruh, namun bisa lambat jika data sangat besar.

    - **K-Nearest Neighbors (KNN)** fokus pada pencarian tetangga terdekat untuk film yang dipilih. Ini lebih efisien untuk aplikasi real-time dan dataset besar.

    Hasil rekomendasi keduanya biasanya mirip, namun KNN lebih scalable dan praktis untuk sistem rekomendasi.
    """)
