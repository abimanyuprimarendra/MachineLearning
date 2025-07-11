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
# Streamlit App
# ===============================
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.title("🎥 Navigasi")
    mode = st.radio("Pilih Halaman", ["Rekomendasi Film", "Visualisasi Jarak"])
    df_sample = load_data_from_gdrive()
    judul_pilihan = sorted(df_sample['movie title'].unique())
    judul_dipilih = st.selectbox("🎬 Pilih Judul Film", judul_pilihan, index=judul_pilihan.index("Spider-Man") if "Spider-Man" in judul_pilihan else 0)

# Load dataset & model
df = load_data_from_gdrive()
if df is None:
    st.stop()

features = ['movie title', 'Generes', 'Director', 'Writer']
for f in features:
    df[f] = df[f].fillna('')
df['combined_features'] = df.apply(lambda row: ' '.join(row[f] for f in features), axis=1)
df['clean_text'] = df['combined_features'].apply(preprocess_text)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# ===============================
# HALAMAN: Rekomendasi Film (5 data, horizontal)
# ===============================
if mode == "Rekomendasi Film":
    st.title("🎬 Rekomendasi Film Berdasarkan Judul")
    rekomendasi = get_recommendations_verbose(judul_dipilih, df, tfidf_matrix, knn_model)

    if rekomendasi:
        st.subheader(f"5 Film Mirip '{judul_dipilih}'")
        cols = st.columns(5)  # 5 kolom dalam satu baris

        for i, film in enumerate(rekomendasi[:5]):  # hanya 5 film ditampilkan
            with cols[i % 5]:
                st.markdown(f"""
                <div style="background-color:#fdfdfd; padding:12px; border-radius:10px; margin-bottom:15px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
                    <h5 style="font-size:16px;">🎞️ {film['Judul']}</h5>
                    <p style="font-size:13px;"><strong>Genre:</strong> {film['Genre']}</p>
                    <p style="font-size:13px;"><strong>Rating:</strong> {film['Rating']}</p>
                    <p style="font-size:12px; color:gray;"><em>{film['Deskripsi']}</em></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning(f"Film '{judul_dipilih}' tidak ditemukan dalam dataset.")

# ===============================
# HALAMAN: Visualisasi Jarak
# ===============================
elif mode == "Visualisasi Jarak":
    st.title("📊 Visualisasi Jarak Cosine Film Serupa")
    rekomendasi = get_recommendations_verbose(judul_dipilih, df, tfidf_matrix, knn_model)

    if rekomendasi:
        rekomendasi_df = pd.DataFrame(rekomendasi)
        rekomendasi_df = rekomendasi_df.sort_values(by='Jarak', ascending=True)

        plt.figure(figsize=(10, 4))
        sns.barplot(x='Judul', y='Jarak', data=rekomendasi_df, palette='magma')
        plt.title(f'Jarak Cosine Distance untuk Film Mirip "{judul_dipilih}"')
        plt.xlabel('Judul Film')
        plt.ylabel('Jarak (semakin kecil = semakin mirip)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(plt)
    else:
        st.warning(f"Film '{judul_dipilih}' tidak ditemukan dalam dataset.")
