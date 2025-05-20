import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Netflix Movie Recommender", layout="wide")

@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['description'] = data.get('description', '').fillna('')
    data['combined'] = data['title'] + " " + data['listed_in'] + " " + data['description']
    return data

@st.cache_data(show_spinner=False)
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(df['combined'])

@st.cache_resource
def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

def get_recommendations(title, df, cosine_sim, knn_model, tfidf_matrix, n=10):
    if title not in df['title'].values:
        return [], []
    idx = df[df['title'] == title].index[0]

    # Cosine similarity
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    cosine_recs = [(df['title'].iloc[i], score) for i, score in sim_scores]

    # KNN
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    knn_recs = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(indices.flatten()[1:], distances.flatten()[1:])]

    return cosine_recs, knn_recs

def plot_similarity_scores(recommendations, method_name):
    titles = [x[0] for x in recommendations]
    scores = [x[1] for x in recommendations]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=scores, y=titles, palette="viridis", ax=ax)
    ax.set_xlabel('Similarity Score')
    ax.set_title(f'Top {len(titles)} Recommendations ({method_name})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    st.pyplot(fig)

# --- Main App ---

st.title("🎬 Netflix Movie Recommender")
st.markdown("""
Aplikasi rekomendasi film Netflix menggunakan metode TF-IDF, Cosine Similarity, dan K-Nearest Neighbors (KNN).
Pilih film dari dropdown untuk melihat rekomendasi berdasarkan dua metode berbeda.
""")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
knn_model = create_knn_model(tfidf_matrix)

# Pilihan film dengan selectbox, disusun alfabetis
selected_title = st.selectbox(
    "Pilih judul film untuk mendapatkan rekomendasi:",
    options=df['title'].sort_values().unique()
)

if selected_title:
    # Hitung rekomendasi dan waktu eksekusi untuk Cosine Similarity
    start_time = time.time()
    cosine_recs, knn_recs = get_recommendations(selected_title, df, cosine_sim, knn_model, tfidf_matrix)
    cosine_time = time.time() - start_time

    # Waktu eksekusi KNN sudah diukur bersamaan di fungsi di atas

    # Buat pilihan rekomendasi yang berbeda untuk Cosine dan KNN
    cosine_titles = set([rec[0] for rec in cosine_recs])
    knn_titles = set([rec[0] for rec in knn_recs])

    # Filter agar rekomendasi berbeda, prioritaskan cosine di kiri, knn di kanan
    cosine_filtered = [rec for rec in cosine_recs if rec[0] not in knn_titles]
    knn_filtered = [rec for rec in knn_recs if rec[0] not in cosine_titles]

    st.markdown(f"### Rekomendasi film mirip dengan **{selected_title}**")

    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Rekomendasi Berdasarkan Cosine Similarity")
        st.write(f"Waktu eksekusi: {cosine_time:.4f} detik")
        for title, score in cosine_filtered[:10]:
            st.write(f"- {title} (score: {score:.4f})")

    with col2:
        st.subheader("Rekomendasi Berdasarkan KNN")
        # Waktu eksekusi KNN pakai timer juga (meskipun sudah dihitung di atas)
        start_time_knn = time.time()
        # ulang hitung knn (bisa dioptimalkan)
        distances, indices = knn_model.kneighbors(tfidf_matrix[df[df['title'] == selected_title].index[0]], n_neighbors=11)
        knn_time = time.time() - start_time_knn
        st.write(f"Waktu eksekusi: {knn_time:.4f} detik")
        for title, score in knn_filtered[:10]:
            st.write(f"- {title} (score: {score:.4f})")

    # Visualisasi distribusi genre film
    st.markdown("---")
    st.subheader("📊 Distribusi 10 Genre Film Teratas di Dataset")
    genre_counts = df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="coolwarm", ax=ax)
    ax.set_xlabel("Jumlah Film")
    ax.set_ylabel("Genre")
    ax.set_title("10 Genre Film Terpopuler di Netflix Dataset")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, v in enumerate(genre_counts.values):
        ax.text(v + 3, i, str(v), color='black', va='center')
    st.pyplot(fig)

    # Visualisasi jumlah film per tahun jika ada kolom 'release_year'
    if 'release_year' in df.columns:
        st.subheader("📅 Jumlah Film Per Tahun Rilis")
        year_counts = df['release_year'].value_counts().sort_index()

        fig2, ax2 = plt.subplots(figsize=(12,5))
        sns.barplot(x=year_counts.index, y=year_counts.values, palette="viridis", ax=ax2)
        ax2.set_xlabel("Tahun Rilis")
        ax2.set_ylabel("Jumlah Film")
        ax2.set_title("Distribusi Jumlah Film per Tahun di Netflix Dataset")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for i, v in enumerate(year_counts.values):
            ax2.text(i, v + max(year_counts.values)*0.01, str(v), ha='center', fontsize=8)
        st.pyplot(fig2)
