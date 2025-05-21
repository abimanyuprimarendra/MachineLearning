import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- Load data dari Google Drive ---
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

df = load_data_from_drive()

# --- TF-IDF matrix ---
@st.cache_data(show_spinner=False)
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix

tfidf_matrix = create_tfidf_matrix(df)

# --- Cosine similarity matrix ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- KNN model ---
@st.cache_resource
def create_knn_model(tfidf_matrix):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)
    return knn_model

knn_model = create_knn_model(tfidf_matrix)

# --- Fungsi rekomendasi ---
def get_content_based_recommendations_with_scores(title, cosine_sim, df, n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # exclude itself
    recommended = [(df['title'].iloc[i], score) for i, score in sim_scores]
    return recommended

def get_knn_recommendations_with_scores(title, knn_model, df, tfidf_matrix, n=5):
    if title not in df['title'].values:
        return "Judul tidak ditemukan di dataset."
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=n+1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    recommended = [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]
    return recommended

# --- Visualisasi Bar Chart ---
def plot_similarity_scores(recs, title):
    if isinstance(recs, str):
        st.write(recs)
        return
    titles = [x[0] for x in recs]
    scores = [x[1] for x in recs]

    fig, ax = plt.subplots()
    ax.barh(titles, scores, color='skyblue')
    ax.set_xlabel('Similarity Score')
    ax.set_title(title)
    ax.invert_yaxis()
    st.pyplot(fig)

# --- Visualisasi WordCloud ---
def plot_wordcloud(recs, df):
    if isinstance(recs, str):
        st.write(recs)
        return
    texts = ""
    for title, _ in recs:
        row = df[df['title'] == title]
        if not row.empty:
            texts += " ".join(row['combined'].values) + " "
    if texts.strip():
        wc = WordCloud(width=400, height=200, background_color='white').generate(texts)
        st.image(wc.to_array())
    else:
        st.write("Tidak ada data untuk WordCloud")

# --- Tabel rekomendasi ---
def display_recs_table(recs, df):
    if isinstance(recs, str):
        st.write(recs)
        return
    rec_data = []
    for title, score in recs:
        row = df[df['title'] == title].iloc[0]
        rec_data.append({
            "Title": title,
            "Similarity": f"{score:.4f}",
            "Genre": row['listed_in'],
            "Description": (row['description'][:100] + '...') if len(row['description']) > 100 else row['description']
        })
    st.table(pd.DataFrame(rec_data))

# --- Streamlit UI ---
st.set_page_config(page_title="🎬 Rekomendasi Film Netflix", layout="centered")
st.title("Sistem Rekomendasi Film Netflix")

col1, col2 = st.columns(2)

with col1:
    title_cosine = st.selectbox("Pilih film untuk rekomendasi Cosine Similarity:", options=df['title'].sort_values().unique())

with col2:
    title_knn = st.selectbox("Pilih film untuk rekomendasi KNN:", options=df['title'].sort_values().unique())

if title_cosine and title_knn:
    with st.spinner("Menghitung rekomendasi..."):
        cosine_recs = get_content_based_recommendations_with_scores(title_cosine, cosine_sim, df, n=5)
        knn_recs = get_knn_recommendations_with_scores(title_knn, knn_model, df, tfidf_matrix, n=5)

    # Cosine Similarity
    st.subheader(f"Rekomendasi berdasarkan Cosine Similarity untuk '{title_cosine}':")
    display_recs_table(cosine_recs, df)
    plot_similarity_scores(cosine_recs, f"Cosine Similarity Scores for '{title_cosine}'")
    st.subheader("Word Cloud dari Film Rekomendasi Cosine Similarity")
    plot_wordcloud(cosine_recs, df)

    # KNN
    st.subheader(f"Rekomendasi berdasarkan KNN untuk '{title_knn}':")
    display_recs_table(knn_recs, df)
    plot_similarity_scores(knn_recs, f"KNN Similarity Scores for '{title_knn}'")
    st.subheader("Word Cloud dari Film Rekomendasi KNN")
    plot_wordcloud(knn_recs, df)
