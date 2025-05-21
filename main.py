import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time
import os

# Load dari Google Drive menggunakan gdown atau requests
@st.cache_data
def load_data_from_drive():
    import os
    import pandas as pd
    file_id = "1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "netflix_data.csv"

    data = None

    try:
        import gdown
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        data = pd.read_csv(output)
    except Exception as e:
        try:
            # fallback via requests
            from io import BytesIO
            import requests
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url)
            if response.status_code == 200:
                data = pd.read_csv(BytesIO(response.content))
        except:
            pass

    if data is None or not isinstance(data, pd.DataFrame):
        st.error("❌ Gagal memuat data. Periksa URL Google Drive dan koneksi internet.")
        return pd.DataFrame(columns=["title", "listed_in", "description", "combined"])

    # Periksa kolom penting
    for col in ['title', 'listed_in', 'description']:
        if col not in data.columns:
            data[col] = ""

    # Bersihkan missing value
    data['listed_in'] = data['listed_in'].fillna('')
    data['description'] = data['description'].fillna('')
    data['title'] = data['title'].fillna('')

    # Gabungkan
    data['combined'] = data['title'] + " " + data['listed_in'] + " " + data['description']

    return data

# Inisialisasi model
@st.cache_data
def prepare_models(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)

    return tfidf_matrix, cosine_sim, indices, knn_model

# Rekomendasi Content-Based
def get_content_recommendations(title, cosine_sim, data, indices, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [(data['title'].iloc[i], score) for i, score in sim_scores]

# Rekomendasi KNN
def get_knn_recommendations(title, knn_model, data, tfidf_matrix, indices, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    vector = tfidf_matrix[idx]
    distances, neighbors = knn_model.kneighbors(vector, n_neighbors=n+1)
    return [(data['title'].iloc[i], 1 - dist) for i, dist in zip(neighbors.flatten()[1:], distances.flatten()[1:])]

# Ukur waktu rata-rata eksekusi
def avg_execution_time(func, title, runs=5):
    durations = []
    for _ in range(runs):
        start = time.time()
        func(title)
        durations.append(time.time() - start)
    return sum(durations) / runs

# ================================
# Streamlit UI
# ================================

st.title("🎬 Sistem Rekomendasi Film Netflix")
st.markdown("Gunakan model Content-Based Filtering & KNN berbasis TF-IDF")

data = load_data_from_drive()

if data is not None:
    tfidf_matrix, cosine_sim, indices, knn_model = prepare_models(data)

    judul = st.selectbox("🎞️ Pilih Judul Film:", data['title'].sort_values())

    if st.button("🔍 Tampilkan Rekomendasi"):
        time_cosine = avg_execution_time(lambda x: get_content_recommendations(x, cosine_sim, data, indices), judul)
        time_knn = avg_execution_time(lambda x: get_knn_recommendations(x, knn_model, data, tfidf_matrix, indices), judul)

        content_recs = get_content_recommendations(judul, cosine_sim, data, indices)
        knn_recs = get_knn_recommendations(judul, knn_model, data, tfidf_matrix, indices)

        st.subheader("📌 Rekomendasi Content-Based Filtering:")
        for title, score in content_recs:
            st.write(f"- {title} (cosine similarity: {score:.4f})")

        st.subheader("📌 Rekomendasi KNN:")
        for title, score in knn_recs:
            st.write(f"- {title} (similarity: {score:.4f})")

        st.markdown("---")
        st.write(f"⏱️ Rata-rata waktu Content-Based: `{time_cosine:.5f}` detik")
        st.write(f"⏱️ Rata-rata waktu KNN: `{time_knn:.5f}` detik")

        st.markdown("""
        ### 📘 Keterangan:
        - **Content-Based Filtering**: menghitung kemiripan antar film menggunakan cosine similarity dari TF-IDF.
        - **KNN (Nearest Neighbors)**: mencari film mirip menggunakan pendekatan tetangga terdekat berbasis TF-IDF.
        - KNN cenderung lebih lambat karena pencarian dilakukan real-time.
        """)
else:
    st.error("Gagal memuat data.")
