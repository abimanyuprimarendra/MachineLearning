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
def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

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

# Streamlit UI
st.title("Perbandingan Rekomendasi Film: KNN vs Cosine Similarity")

df = load_data_from_drive()
tfidf_matrix = create_tfidf_matrix(df)
cosine_sim = cosine_similarity(tfidf_matrix)
knn_model = create_knn_model(tfidf_matrix)

title = st.selectbox("Pilih judul film:", options=df['title'].sort_values().unique())

if title:
    cosine_recs = get_content_based_recommendations(title, cosine_sim, df)
    knn_recs = get_knn_recommendations(title, knn_model, df, tfidf_matrix)

    # Tampilkan tabel berdampingan
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cosine Similarity")
        for i, (rec_title, score) in enumerate(cosine_recs, 1):
            st.write(f"{i}. {rec_title} (similarity: {score:.4f})")

    with col2:
        st.subheader("KNN Recommendation")
        for i, (rec_title, score) in enumerate(knn_recs, 1):
            st.write(f"{i}. {rec_title} (similarity: {score:.4f})")

    # Tampilkan visualisasi
    st.subheader("Visualisasi Perbandingan Skor Similarity")
    combined_data = pd.DataFrame({
        'Film': [rec[0] for rec in cosine_recs] + [rec[0] for rec in knn_recs],
        'Similarity': [rec[1] for rec in cosine_recs] + [rec[1] for rec in knn_recs],
        'Metode': ['Cosine'] * len(cosine_recs) + ['KNN'] * len(knn_recs)
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(data=combined_data, x='Similarity', y='Film', hue='Metode')
    st.pyplot(plt.gcf())

    # Tampilkan irisan rekomendasi yang sama
    st.subheader("Film yang Direkomendasikan oleh Keduanya")
    cosine_titles = set([title for title, _ in cosine_recs])
    knn_titles = set([title for title, _ in knn_recs])
    common_titles = list(cosine_titles & knn_titles)

    if common_titles:
        for i, common in enumerate(common_titles, 1):
            st.write(f"{i}. {common}")
    else:
        st.write("Tidak ada rekomendasi yang sama antara kedua metode.")
