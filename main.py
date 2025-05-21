import streamlit as st
import pandas as pd
import re
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import log1p
import difflib

# Ambil API key dari environment variable
api_key = os.getenv("OMDB_API_KEY")
if not api_key:
    st.error("API key tidak ditemukan. Pastikan sudah diatur di environment variable STREAMLIT_OMDB_API_KEY.")
    st.stop()

# Load dataset dari Google Drive
@st.cache_data
def load_data():
    csv_url = "https://drive.google.com/uc?id=1ix27-hPzSIjBrZGI5fl3HP5QFlJlDY0K"
    try:
        df = pd.read_csv(csv_url)

        # Bersihkan votes dan rating
        df['votes'] = df['votes'].fillna('0').str.replace(',', '', regex=False).astype(int)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

        # Pembersihan teks
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        for col in ['genre', 'description', 'stars']:
            df[col] = df[col].fillna('').apply(clean_text)

        # Gabungkan fitur
        df['combined_features'] = (
            (df['genre'] + ' ') * 2 +
            (df['stars'] + ' ') * 2 +
            df['description']
        )
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# Siapkan model TF-IDF dan KNN
@st.cache_resource
def prepare_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, knn

# Fungsi dapatkan poster dari OMDb API
def get_movie_poster(title):
    url = f"http://www.omdbapi.com/?apikey={api_key}&t={title}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            poster_url = data.get("Poster", None)
            if poster_url and poster_url != "N/A":
                return poster_url
    except:
        pass
    return None

# Fungsi rekomendasi
def recommend(title, n_recommendations=5, min_rating=7, min_votes=1000):
    idx_list = df.index[df['title'].str.lower() == title.lower()]
    if len(idx_list) == 0:
        closest_matches = difflib.get_close_matches(title, df['title'], n=3)
        return f"Film tidak ditemukan! Coba: {', '.join(closest_matches)}" if closest_matches else "Film tidak ditemukan!", None

    idx = idx_list[0]
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=50)

    recommendations = []
    added_titles = set()

    for i in range(1, len(indices[0])):
        rec_idx = indices[0][i]
        rec_title = df.iloc[rec_idx]['title']

        if rec_title.lower() == title.lower() or rec_title.lower() in added_titles:
            continue

        rating = df.iloc[rec_idx]['rating']
        votes = df.iloc[rec_idx]['votes']
        similarity = 1 - distances[0][i]

        if rating >= min_rating and votes >= min_votes:
            score = (similarity * 0.5) + (rating / 10 * 0.3) + (log1p(votes) / 10 * 0.2)
            recommendations.append((
                rec_title,                                  # Title
                df.iloc[rec_idx]['genre'],                  # Genre
                round(similarity, 3),                        # Similarity
                rating,                                     # Rating
                round(score, 4),                            # Score
                df.iloc[rec_idx]['description'][:150] + '...',  # Description
                f"{votes:,}"                                # Votes formatted
            ))
            added_titles.add(rec_title.lower())

        if len(recommendations) == n_recommendations:
            break

    if not recommendations:
        return "Tidak ada film yang direkomendasikan berdasarkan kriteria.", None

    recommendations = sorted(recommendations, key=lambda x: x[4], reverse=True)
    df_result = pd.DataFrame(recommendations, columns=[
        'Title', 'Genre', 'Similarity', 'Rating', 'Score', 'Description', 'Votes'
    ])
    return None, df_result


# =======================
# Streamlit App
# =======================
st.title("🎬 Sistem Rekomendasi Film dengan Poster")

df = load_data()
if not df.empty:
    vectorizer, tfidf_matrix, knn = prepare_model(df)

    title_input = st.text_input("Masukkan judul film", "Cobra Kai")
    n = st.slider("Jumlah rekomendasi", 1, 20, 5)
    min_rating = st.slider("Minimal rating", 0.0, 10.0, 7.0)
    min_votes = st.number_input("Minimal jumlah votes", min_value=0, value=1000)

    if st.button("Rekomendasikan"):
        error_msg, hasil = recommend(title_input, n, min_rating, min_votes)
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"Berikut adalah {len(hasil)} film mirip '{title_input}' 🎉")

            # Tampilkan dataframe hasil rekomendasi
            st.dataframe(hasil.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)

            # Visualisasi poster film max 5 rekomendasi
            st.markdown("### Poster Film Rekomendasi:")
            cols = st.columns(min(len(hasil), 5))

            for idx, col in enumerate(cols):
                title_rec = hasil.iloc[idx]['Title']
                poster_url = get_movie_poster(title_rec)
                with col:
                    st.markdown(f"**{title_rec}**")
                    if poster_url:
                        st.image(poster_url, use_column_width=True)
                    else:
                        st.write("Poster tidak tersedia.")
else:
    st.stop()
