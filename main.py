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

    plt.figure(figsize=(10, 4))
    sns.barplot(x=scores, y=titles, palette="viridis")
    plt.xlabel('Similarity Score')
    plt.title(f'Top {len(titles)} Recommendations ({method_name})')
    st.pyplot(plt.gcf())

# --- Main App ---

st.title("🎬 Netflix Movie Recommender")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
knn_model = create_knn_model(tfidf_matrix)

# Pilihan film dengan selectbox, disusun alfabetis
selected_title = st.selectbox(
    "Pilih judul film untuk mendapatkan rekomendasi:",
    options=df['title'].sort_values().unique(),
    index=0
)

if selected_title:
    with st.spinner("Menghitung rekomendasi..."):
        cosine_recs, knn_recs = get_recommendations(selected_title, df, cosine_sim, knn_model, tfidf_matrix)

    st.markdown(f"### Rekomendasi film mirip dengan **{selected_title}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rekomendasi Berdasarkan Cosine Similarity")
        for title, score in cosine_recs:
            st.write(f"- {title} (score: {score:.4f})")
        plot_similarity_scores(cosine_recs, "Cosine Similarity")

    with col2:
        st.subheader("Rekomendasi Berdasarkan KNN")
        for title, score in knn_recs:
            st.write(f"- {title} (score: {score:.4f})")
        plot_similarity_scores(knn_recs, "KNN")

    # Visualisasi tambahan: distribusi genre film dataset
    st.markdown("---")
    st.subheader("Distribusi Genre Film di Dataset")
    genre_counts = df['listed_in'].str.split(',').explode().str.strip().value_counts()
    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(x=genre_counts.values[:15], y=genre_counts.index[:15], palette="magma", ax=ax)
    ax.set_xlabel("Jumlah Film")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

    # Visualisasi tambahan: jumlah film per tahun jika ada kolom 'release_year'
    if 'release_year' in df.columns:
        st.subheader("Jumlah Film per Tahun Rilis")
        year_counts = df['release_year'].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(12,4))
        sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', ax=ax2)
        ax2.set_xlabel("Tahun Rilis")
        ax2.set_ylabel("Jumlah Film")
        st.pyplot(fig2) ini perbaiki yang salah saja
