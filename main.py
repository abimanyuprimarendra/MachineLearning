# app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# 1. Load Data dari Google Drive
# -----------------------
@st.cache_data
def load_data():
    # Gunakan link yang bisa langsung diakses
    csv_url = "https://drive.google.com/uc?id=1tHMyi7TRCapR6_UbHDd-Wbfr5GL_dE6x"
    
    # Baca CSV
    df = pd.read_csv(csv_url)

    # Normalisasi nama kolom agar aman
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename jika perlu
    if 'release year' in df.columns:
        df.rename(columns={'release year': 'releaseyear'}, inplace=True)

    # Validasi kolom
    required_cols = ['title', 'genres', 'releaseyear']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Kolom '{col}' tidak ditemukan di dataset.")
            return pd.DataFrame()  # Kosong agar tidak crash

    # Pra-pemrosesan
    df = df.dropna(subset=required_cols)
    df['title'] = df['title'].str.lower().str.strip()
    df['genres'] = df['genres'].str.lower().str.strip()
    df['releaseyear'] = df['releaseyear'].astype(int)

    # Gabungkan fitur konten
    df['combined_features'] = (
        df['title'] + ' ' + df['genres'] + ' ' + df['releaseyear'].astype(str)
    )

    return df

# Load data
df = load_data()

# -----------------------
# 2. TF-IDF dan Cosine Similarity
# -----------------------
@st.cache_data
def compute_similarity(df):
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df)

# -----------------------
# 3. Fungsi Rekomendasi
# -----------------------
def recommend(title, n=5):
    title = title.lower().strip()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    seen = set()
    results = []
    for i, _ in sim_scores:
        film_title = df.loc[i, 'title']
        if film_title not in seen:
            seen.add(film_title)
            results.append(i)
        if len(results) == n:
            break
    return df[['title', 'genres', 'releaseyear']].iloc[results]

# -----------------------
# 4. Tampilan Streamlit
# -----------------------
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="centered")
st.title("🎬 Sistem Rekomendasi Film")
st.caption("Content-Based Filtering | Data dari Google Drive")

if df.empty:
    st.stop()

# Dropdown agar user lebih mudah pilih film
film_options = df['title'].drop_duplicates().sort_values().str.title().tolist()
selected_title = st.selectbox("Pilih judul film", film_options)

if selected_title:
    st.markdown(f"**Hasil rekomendasi untuk:** `{selected_title}`")
    result_df = recommend(selected_title)
    if result_df is not None:
        st.dataframe(result_df.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("❌ Film tidak ditemukan atau tidak ada rekomendasi.")
