import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['combined'])
    return matrix

@st.cache_resource(show_spinner=False)
def create_knn_model(tfidf_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return model

def get_content_based_recommendations(title, df, tfidf_matrix, top_n=5):
    # Cari index film
    if title not in df['title'].values:
        return []
    idx = df.index[df['title'] == title][0]
    
    # Hitung cosine similarity dari film ke semua film
    cosine_sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Dapatkan top_n rekomendasi tertinggi (kecuali film itu sendiri)
    top_indices = cosine_sim_row.argsort()[-top_n-1:][::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]
    
    results = [(df['title'].iloc[i], cosine_sim_row[i]) for i in top_indices]
    return results

def get_knn_recommendations(title, df, knn_model, tfidf_matrix, top_n=5):
    if title not in df['title'].values:
        return []
    idx = df.index[df['title'] == title][0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    indices = indices.flatten()[1:]  # exclude itself
    distances = distances.flatten()[1:]
    results = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(indices, distances)]
    return results

def measure_avg_time(func, title, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        func(title)
        times.append(time.time() - start)
    return sum(times) / runs

def plot_bar_chart(recommendations, method_name):
    titles = [rec[0] for rec in recommendations]
    scores = [rec[1] for rec in recommendations]
    fig, ax = plt.subplots()
    ax.barh(titles[::-1], scores[::-1], color='skyblue')
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top {len(recommendations)} Rekomendasi berdasarkan {method_name}")
    plt.tight_layout()
    st.pyplot(fig)

# MAIN APP
st.title("Sistem Rekomendasi Film Netflix")

df = load_data()
tfidf_matrix = create_tfidf_matrix(df)
knn_model = create_knn_model(tfidf_matrix)

selected_title = st.selectbox("Pilih judul film:", df['title'].sort_values().unique())

if selected_title:
    with st.spinner("Menghitung rekomendasi..."):
        avg_time_content = measure_avg_time(lambda t=selected_title: get_content_based_recommendations(t, df, tfidf_matrix), selected_title)
        avg_time_knn = measure_avg_time(lambda t=selected_title: get_knn_recommendations(t, df, knn_model, tfidf_matrix), selected_title)

        st.write(f"Rata-rata waktu eksekusi Content-Based (Cosine Similarity): **{avg_time_content:.5f} detik**")
        st.write(f"Rata-rata waktu eksekusi KNN: **{avg_time_knn:.5f} detik**")

        content_recs = get_content_based_recommendations(selected_title, df, tfidf_matrix)
        knn_recs = get_knn_recommendations(selected_title, df, knn_model, tfidf_matrix)

        st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{selected_title}':")
        for title, score in content_recs:
            st.write(f"- {title} (similarity: {score:.4f})")

        st.subheader(f"Rekomendasi berdasarkan KNN untuk '{selected_title}':")
        for title, score in knn_recs:
            st.write(f"- {title} (similarity: {score:.4f})")

        st.subheader("Visualisasi Similarity Scores (Cosine Similarity):")
        plot_bar_chart(content_recs, "Cosine Similarity")

        st.subheader("Visualisasi Similarity Scores (KNN):")
        plot_bar_chart(knn_recs, "KNN")
