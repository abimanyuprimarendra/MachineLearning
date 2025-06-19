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

df['title'] = df['title'].str.lower().str.strip()
df['genres'] = df['genres'].str.lower().str.strip()
df['releaseYear'] = df['releaseYear'].astype(int)
df['imdbAverageRating'] = pd.to_numeric(df['imdbAverageRating'], errors='coerce')
df['imdbNumVotes'] = pd.to_numeric(df['imdbNumVotes'], errors='coerce')
df = df.dropna().reset_index(drop=True)

df['combined_features'] = df['title'] + ' ' + df['genres'] + ' ' + df['releaseYear'].astype(str)

# ================================
# 🔍 TF-IDF dan Cosine Similarity (Efisien)
# ================================
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Batasi kata agar hemat RAM
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ================================
# 🎯 Fungsi Rekomendasi
# ================================
def recommend(title, n_recommendations=5):
    title = title.lower().strip()
    if title not in indices:
        return [], f"Film '{title}' tidak ditemukan di dataset."

    idx = indices[title]
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

    return df.iloc[recommendations][['title', 'genres', 'releaseYear', 'imdbAverageRating']], None

# ================================
# 🎛️ Streamlit UI
# ================================
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="centered")
st.title("🎬 Sistem Rekomendasi Film")
st.markdown("Berbasis **Content-Based Filtering** menggunakan TF-IDF dan Cosine Similarity.")

user_input = st.text_input("Masukkan judul film (contoh: The Dark Knight)", "")

if user_input:
    result_df, error = recommend(user_input)
    if error:
        st.warning(error)
    else:
        st.success(f"Rekomendasi film untuk: **{user_input.title()}**")
        st.dataframe(result_df)
