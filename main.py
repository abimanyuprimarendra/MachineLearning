import streamlit as st
import pandas as pd
import re
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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
        for i in range(1, len(indices[0])):  # skip indeks ke-0 (judul itu sendiri)
            film_index = indices[0][i]
            distance = distances[0][i]

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
                'Jarak': distance
            })

        return results

    except IndexError:
        return None

# ===============================
# App Streamlit
# ===============================
st.set_page_config(page_title="Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")

# Pilih Mode Navigasi
mode = st.radio("Pilih Mode Tampilan:", ["Rekomendasi", "Visualisasi"])

# Load dataset
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

# Dropdown untuk memilih film
judul_pilihan = sorted(df['movie title'].unique())
judul_dipilih = st.selectbox("📽️ Pilih Judul Film", judul_pilihan, index=judul_pilihan.index("Spider-Man") if "Spider-Man" in judul_pilihan else 0)

# ==========================
# MODE 1: Tampilkan Rekomendasi
# ==========================
if mode == "Rekomendasi":
    if st.button("Tampilkan Rekomendasi"):
        rekomendasi = get_recommendations_verbose(judul_dipilih, df, tfidf_matrix, knn_model)

        if rekomendasi:
            st.subheader(f"Hasil Rekomendasi Mirip '{judul_dipilih}'")
            for film in rekomendasi:
                with st.container():
                    st.markdown(f"""
                    <div style="background-color:#f9f9f9; padding:15px; border-radius:10px; margin-bottom:10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                        <h5>🎞️ {film['Judul']}</h5>
                        <p><strong>Genre:</strong> {film['Genre']} | <strong>Rating:</strong> {film['Rating']}</p>
                        <p style="color:gray;"><em>{film['Deskripsi']}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning(f"Film '{judul_dipilih}' tidak ditemukan dalam dataset.")

# ==========================
# MODE 2: Visualisasi Jarak
# ==========================
elif mode == "Visualisasi":
    rekomendasi = get_recommendations_verbose(judul_dipilih, df, tfidf_matrix, knn_model)
    if rekomendasi:
        rekomendasi_df = pd.DataFrame(rekomendasi)
        rekomendasi_df = rekomendasi_df.sort_values(by='Jarak', ascending=True)

        st.subheader(f"Visualisasi Jarak Cosine Film Mirip '{judul_dipilih}'")

        plt.figure(figsize=(10, 4))
        sns.barplot(x='Judul', y='Jarak', data=rekomendasi_df, palette='magma')
        plt.title('Jarak Cosine Distance')
        plt.xlabel('Judul Film')
        plt.ylabel('Jarak (Semakin kecil = Semakin Mirip)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(plt)
    else:
        st.warning(f"Film '{judul_dipilih}' tidak ditemukan dalam dataset.")
