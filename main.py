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

# Simpan versi asli untuk dropdown
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
# 🎯 Fungsi Kategori Kemiripan
# ================================
def get_similarity_level(score):
    if score >= 0.80:
        return "🟢 Sangat Mirip"
    elif score >= 0.60:
        return "🟡 Mirip"
    elif score >= 0.40:
        return "🟠 Cukup Mirip"
    else:
        return "🔴 Sedikit Mirip"

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

    for i, score in sim_scores:
        film_title = df.loc[i, 'title']
        if film_title not in seen_titles:
            seen_titles.add(film_title)
            recommendations.append((i, score))
        if len(recommendations) == n_recommendations:
            break

    selected_rows = []
    for i, score in recommendations:
        row = {
            'Judul Film': df.loc[i, 'title_original'],
            'Genre': df.loc[i, 'genres'],
            'Tahun Rilis': df.loc[i, 'releaseYear'],
            'Rating': df.loc[i, 'imdbAverageRating'],
            'Kemiripan (%)': round(score * 100, 1),
            'Tingkat': get_similarity_level(score)
        }
        selected_rows.append(row)

    result_df = pd.DataFrame(selected_rows)
    return result_df, None

# ================================
# 🎛️ Streamlit UI
# ================================
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎬 Sistem Rekomendasi Film</h1>
    <p style='text-align: center;'>Berbasis <b>Content-Based Filtering</b> menggunakan <b>TF-IDF</b> dan <b>Cosine Similarity</b>.</p>
    <p style='text-align: center; font-size: 14px;'>Nilai kemiripan dihitung berdasarkan genre, judul, dan tahun rilis. Di atas 40% dianggap relevan.</p>
""", unsafe_allow_html=True)

# Dropdown
st.markdown("## 🎞️ Pilih Film Referensi")
film_list = sorted(df['title_original'].unique())
selected_film = st.selectbox("Pilih judul film:", film_list)

# Tombol
if st.button("🎯 Tampilkan Rekomendasi"):
    result_df, error = recommend(selected_film)
    if error:
        st.warning(error)
    else:
        st.success(f"✅ Rekomendasi film untuk: **{selected_film}**")
        st.markdown("### 🎥 Daftar Rekomendasi Berdasarkan Kemiripan")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
