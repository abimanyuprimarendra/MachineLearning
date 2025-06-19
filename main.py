import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# 📦 Load Dataset dari Google Drive
# ================================
@st.cache_data
def load_data_from_drive(n_rows=1000):
    url = "https://drive.google.com/uc?id=1tHMyi7TRCapR6_UbHDd-Wbfr5GL_dE6x"
    df = pd.read_csv(url).head(n_rows)
    return df

df = load_data_from_drive()

# ================================
# 🧹 Pra-pemrosesan Data
# ================================
required_cols = ['title', 'genres', 'releaseYear', 'imdbAverageRating', 'imdbNumVotes']
df = df.dropna(subset=required_cols)

# Simpan versi asli untuk dropdown (tanpa lowercase)
df['title_original'] = df['title']

df['title'] = df['title'].str.lower().str.strip()
df['genres'] = df['genres'].str.lower().str.strip()
df['releaseYear'] = df['releaseYear'].astype(int)
df['imdbAverageRating'] = pd.to_numeric(df['imdbAverageRating'], errors='coerce')
df['imdbNumVotes'] = pd.to_numeric(df['imdbNumVotes'], errors='coerce')
df = df.dropna().reset_index(drop=True)

df['combined_features'] = df['title'] + ' ' + df['genres'] + ' ' + df['releaseYear'].astype(str)

# ================================
# 🔍 TF-IDF dan Cosine Similarity
# ================================
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ================================
# 🎯 Fungsi Rekomendasi
# ================================
def recommend(title, n_recommendations=5):
    title_lower = title.lower().strip()
    if title_lower not in indices:
        return [], f"Film '{title}' tidak ditemukan di dataset."

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx].toarray().flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    seen_titles = set()
    recommendations = []

    for i, _ in sim_scores:
        film_title = df.loc[i, 'title']
        if film_title not in seen_titles:
            seen_titles.add(film_title)
            recommendations.append(i)
        if len(recommendations) == n_recommendations:
            break

    selected = df.iloc[recommendations][['title_original', 'genres', 'releaseYear', 'imdbAverageRating']]
    selected.columns = ['Judul Film', 'Genre', 'Tahun Rilis', 'Rating']
    return selected.reset_index(drop=True), None

# ================================
# 🎛️ Streamlit UI
# ================================
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎬 Sistem Rekomendasi Film</h1>
    <p style='text-align: center;'>Berbasis <b>Content-Based Filtering</b> menggunakan <b>TF-IDF</b> dan <b>Cosine Similarity</b>.</p>
""", unsafe_allow_html=True)

# Dropdown: tampilkan pilihan berdasarkan title_original (tanpa lowercase)
st.markdown("## 🎞️ Pilih Film Referensi")
film_list = sorted(df['title_original'].unique())
selected_film = st.selectbox("Pilih judul film:", film_list)

# Tombol: Klik untuk tampilkan rekomendasi
if st.button("🎯 Tampilkan Rekomendasi"):
    result_df, error = recommend(selected_film)
    if error:
        st.warning(error)
    else:
        st.success(f"✅ Rekomendasi film untuk: **{selected_film}**")
        st.markdown("### 🎥 Daftar Rekomendasi:")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
