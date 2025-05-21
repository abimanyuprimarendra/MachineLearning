import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

# Load dataset dari Google Drive
@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['description'] = data['description'].fillna('') if 'description' in data.columns else ''
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

# Fungsi rekomendasi Content-Based
def get_content_recommendations(title, cosine_sim, data, indices, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [(data['title'].iloc[i], score) for i, score in sim_scores]

# Fungsi rekomendasi KNN
def get_knn_recommendations(title, knn_model, data, tfidf_matrix, indices, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    vector = tfidf_matrix[idx]
    distances, neighbors = knn_model.kneighbors(vector, n_neighbors=n+1)
    return [(data['title'].iloc[i], 1 - dist) for i, dist in zip(neighbors.flatten()[1:], distances.flatten()[1:])]

# Fungsi ukur waktu rata-rata
def avg_execution_time(func, title, runs=5):
    durations = []
    for _ in range(runs):
        start = time.time()
        func(title)
        durations.append(time.time() - start)
    return sum(durations) / runs

# Streamlit Interface
st.title("Sistem Rekomendasi Film Netflix")

data = load_data_from_drive()
tfidf_matrix, cosine_sim, indices, knn_model = prepare_models(data)

judul = st.selectbox("Pilih Judul Film:", data['title'].sort_values())

if st.button("Tampilkan Rekomendasi"):
    time_cosine = avg_execution_time(lambda x: get_content_recommendations(x, cosine_sim, data, indices), judul)
    time_knn = avg_execution_time(lambda x: get_knn_recommendations(x, knn_model, data, tfidf_matrix, indices), judul)

    content_recs = get_content_recommendations(judul, cosine_sim, data, indices)
    knn_recs = get_knn_recommendations(judul, knn_model, data, tfidf_matrix, indices)

    st.subheader("Rekomendasi Content-Based Filtering:")
    for title, score in content_recs:
        st.write(f"- {title} (cosine similarity: {score:.4f})")

    st.subheader("Rekomendasi KNN:")
    for title, score in knn_recs:
        st.write(f"- {title} (similarity: {score:.4f})")

    st.markdown("---")
    st.write(f"⏱️ Rata-rata waktu Content-Based: `{time_cosine:.5f}` detik")
    st.write(f"⏱️ Rata-rata waktu KNN: `{time_knn:.5f}` detik")

    st.markdown("""
    #### Keterangan:
    - Model **Content-Based** menggunakan cosine similarity langsung antar film.
    - Model **KNN** menggunakan pendekatan tetangga terdekat berbasis TF-IDF.
    - Eksekusi KNN lebih lambat karena pencarian tetangga dilakukan saat runtime.
    """)
