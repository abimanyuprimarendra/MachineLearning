import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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

@st.cache_resource
def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

def get_content_based_recommendations_with_scores(title, cosine_sim, df):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # 5 rekomendasi
    recommended = [(df['title'].iloc[i], score) for i, score in sim_scores]
    return recommended

def get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix, n_neighbors=5):
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=n_neighbors + 1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    recommended = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]
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

def explain_methods():
    st.markdown("## Perbandingan Metode Rekomendasi")
    st.markdown("""
    **Cosine Similarity**:
    - Menghitung kesamaan langsung antar fitur (contoh: deskripsi film).
    - Efisien untuk dataset kecil hingga sedang.
    - Mudah dipahami dan cepat dihitung.
    - Kurang optimal untuk data sangat besar tanpa optimasi.

    **K-Nearest Neighbors (KNN)**:
    - Menggunakan algoritma pencarian tetangga terdekat berdasarkan jarak (cosine distance).
    - Lebih fleksibel untuk berbagai tipe data.
    - Bisa lebih lambat karena perlu pencarian tetangga dalam ruang fitur.
    - Memungkinkan penggunaan parameter K yang bisa di-tuning untuk hasil optimal.

    **Kesimpulan:**
    - Cosine similarity cocok untuk sistem rekomendasi sederhana dan cepat.
    - KNN cocok untuk dataset besar dan kompleks, tapi perlu optimasi dan tuning.
    """)

st.title("Sistem Rekomendasi Film Netflix")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
knn_model = create_knn_model(tfidf_matrix)

title = st.selectbox("Pilih judul film untuk direkomendasikan:", options=df['title'].sort_values().unique())

if title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_cosine = measure_avg_time(lambda t=title: get_content_based_recommendations_with_scores(t, cosine_sim, df), title)
        avg_time_knn = measure_avg_time(lambda t=title: get_knn_recommendations_with_scores(t, knn_model, df, tfidf_matrix), title)

    st.write(f"Rata-rata waktu eksekusi Cosine Similarity: **{avg_time_cosine:.5f} detik**")
    st.write(f"Rata-rata waktu eksekusi KNN: **{avg_time_knn:.5f} detik**")

    cosine_recs = get_content_based_recommendations_with_scores(title, cosine_sim, df)
    knn_recs = get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix)

    st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{title}':")
    for rec_title, score in cosine_recs:
        st.write(f"- {rec_title} (similarity: {score:.4f})")

    st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title}':")
    for rec_title, score in knn_recs:
        st.write(f"- {rec_title} (similarity: {score:.4f})")

    # Visualisasi perbandingan skor
    rec_titles = [rec[0] for rec in cosine_recs]
    cosine_scores = [rec[1] for rec in cosine_recs]
    knn_scores = [score for _, score in knn_recs]

    compare_df = pd.DataFrame({
        'Film': rec_titles,
        'Cosine Similarity': cosine_scores,
        'KNN Similarity': knn_scores
    }).set_index('Film')

    st.subheader("Perbandingan Similarity Score Rekomendasi")
    st.bar_chart(compare_df)

    # Penjelasan metode
    explain_methods()
