import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import requests
import io

# ===============================
# Load dataset dari Google Drive
# ===============================
@st.cache_data
def load_data_from_gdrive():
    file_id = "1s9bogJ0B0rhGgYvTNUjaobEbxL4UfkhZ"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Gagal mengambil data dari Google Drive.")
        return None
    return pd.read_csv(io.BytesIO(response.content))

# ===============================
# Preprocessing teks
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join(text.split())

# ===============================
# Fungsi Rekomendasi
# ===============================
def get_recommendations_verbose(title, df, tfidf_matrix, knn_model, n=10):
    try:
        idx = df[df['movie title'].str.lower() == title.lower()].index[0]
        distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

        results = []
        for i in range(1, len(indices[0])):
            film_index = indices[0][i]
            distance = distances[0][i]
            similarity = 1 - distance

            film_title = df.iloc[film_index]['movie title']
            rating = df.iloc[film_index].get('Rating', '-')
            genre = df.iloc[film_index]['Generes']
            overview = df.iloc[film_index].get('Overview', '')

            results.append({
                'Rank': i,
                'Judul': film_title,
                'Genre': genre,
                'Rating': rating,
                'Deskripsi': overview,
                'Similarity': round(similarity, 4)
            })

        return results

    except IndexError:
        return None

# ===============================
# App Streamlit
# ===============================
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")

menu = st.sidebar.radio("📁 Navigasi", ["Rekomendasi Film", "Visualisasi Dataset"])
df = load_data_from_gdrive()
if df is None:
    st.stop()

# Preprocessing
features = ['movie title', 'Generes', 'Director', 'Writer']
for f in features:
    df[f] = df[f].fillna('')
df['combined_features'] = df.apply(lambda row: ' '.join(row[f] for f in features), axis=1)
df['clean_text'] = df['combined_features'].apply(preprocess_text)

# TF-IDF dan KNN
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# ===============================
# Halaman Rekomendasi
# ===============================
if menu == "Rekomendasi Film":
    st.subheader("🔍 Rekomendasi Berdasarkan Judul Film")

    judul_input = st.text_input("Masukkan Judul Film", value="Spider-Man")
    if st.button("Tampilkan Rekomendasi"):
        rekomendasi = get_recommendations_verbose(judul_input, df, tfidf_matrix, knn_model)

        if rekomendasi:
            for film in rekomendasi:
                with st.container():
                    st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; margin-bottom:10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                        <h5>🎞️ {film['Judul']}</h5>
                        <p><strong>Genre:</strong> {film['Genre']} | <strong>Rating:</strong> {film['Rating']}</p>
                        <p style="color:gray;"><em>{film['Deskripsi'][:250]}...</em></p>
                        <p><strong>Similarity:</strong> {film['Similarity']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"Film '{judul_input}' tidak ditemukan dalam dataset.")

# ===============================
# Halaman Visualisasi
# ===============================
elif menu == "Visualisasi Dataset":
    st.subheader("📊 Visualisasi Data Film")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎭 Genre Film Terbanyak")
        st.markdown("Visualisasi 10 genre film yang paling sering muncul dalam dataset.")
        genres = df['Generes'].dropna().astype(str).str.replace(r'[\[\]\'\"]', '', regex=True).str.split(', ')
        genre_counts = genres.explode().value_counts().head(10)

        fig1, ax1 = plt.subplots()
        genre_counts.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_xlabel("Jumlah Film")
        ax1.set_ylabel("Genre")
        ax1.set_title("Top 10 Genre Terpopuler")
        ax1.invert_yaxis()
        st.pyplot(fig1)

    with col2:
        st.markdown("### ⭐ Distribusi Rating Film")
        st.markdown("Distribusi jumlah film berdasarkan nilai rating.")
        if 'Rating' in df.columns:
            try:
                df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
                fig2, ax2 = plt.subplots()
                sns.histplot(df['Rating'].dropna(), bins=20, kde=True, ax=ax2, color='salmon')
                ax2.set_xlabel("Rating")
                ax2.set_ylabel("Jumlah Film")
                ax2.set_title("Distribusi Rating Film")
                st.pyplot(fig2)
            except:
                st.warning("Rating tidak bisa divisualisasikan.")
        else:
            st.warning("Kolom Rating tidak tersedia.")
