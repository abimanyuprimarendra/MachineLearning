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
    csv_url = "https://drive.google.com/uc?id=1ix27-hPzSIjBrZGI5fl3HP5QFlJlDY0K&export=download"
    df = pd.read_csv(csv_url)

    # Pra-pemrosesan
    df = df.dropna(subset=['title', 'genres', 'releaseYear'])
    df['title'] = df['title'].str.lower().str.strip()
    df['genres'] = df['genres'].str.lower().str.strip()
    df['releaseYear'] = df['releaseYear'].astype(int)

    # Gabungkan fitur konten
    df['combined_features'] = df['title'] + ' ' + df['genres'] + ' ' + df['releaseYear'].astype(str)
    
    return df

df = load_data()

# -----------------------
# 2. TF-IDF dan Cosine Similarity
# -----------------------
@st.cache_data
def compute_similarity(df):
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
    return df[['title', 'genres', 'releaseYear']].iloc[results]

# -----------------------
# 4. Streamlit UI
# -----------------------
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="centered")
st.title("🎬 Sistem Rekomendasi Film")
st.caption("Berbasis Content-Based Filtering dari Google Drive")

input_title = st.text_input("Masukkan judul film", placeholder="Contoh: inception")

if input_title:
    if input_title.lower().strip() not in indices:
        st.warning("❌ Film tidak ditemukan dalam data.")
    else:
        st.success(f"Hasil rekomendasi untuk: {input_title.title()}")
        result_df = recommend(input_title, n=5)
        st.table(result_df)
