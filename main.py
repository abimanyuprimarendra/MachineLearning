import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import log1p

# Load dataset dari Google Drive
@st.cache_data
def load_data():
    csv_url = "https://drive.google.com/uc?id=1ix27-hPzSIjBrZGI5fl3HP5QFlJlDY0K"
    df = pd.read_csv(csv_url)

    # Bersihkan votes dan rating
    df['votes'] = df['votes'].fillna('0').str.replace(',', '', regex=False).astype(int)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

    # Fungsi pembersihan teks
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Bersihkan kolom genre, description, stars
    for col in ['genre', 'description', 'stars']:
        df[col] = df[col].fillna('').apply(clean_text)

    # Gabungkan fitur untuk TF-IDF
    df['combined_features'] = (
        (df['genre'] + ' ') * 2 +
        (df['stars'] + ' ') * 2 +
        df['description']
    )

    return df

# Load data
df = load_data()

# TF-IDF dan KNN
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_matrix)

# Fungsi rekomendasi
def recommend(title, n_recommendations=5, min_rating=7, min_votes=1000):
    idx_list = df.index[df['title'].str.lower() == title.lower()]
    if len(idx_list) == 0:
        return "Film tidak ditemukan!"
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
                rec_title,
                round(similarity, 3),
                rating,
                f"{votes:,}",
                round(score, 4),
                df.iloc[rec_idx]['description'][:150] + '...'
            ))
            added_titles.add(rec_title.lower())

        if len(recommendations) == n_recommendations:
            break

    if not recommendations:
        return "Tidak ada film yang direkomendasikan berdasarkan kriteria."

    recommendations = sorted(recommendations, key=lambda x: x[4], reverse=True)

    return pd.DataFrame(recommendations, columns=[
        'Title', 'Similarity', 'Rating', 'Votes', 'Score', 'Description'
    ])

# Streamlit UI
st.title("🎬 Sistem Rekomendasi Film")

judul_input = st.text_input("Masukkan judul film", "Cobra Kai")
n = st.slider("Jumlah rekomendasi", 1, 20, 10)
min_rating = st.slider("Minimal rating", 0.0, 10.0, 7.0)
min_votes = st.number_input("Minimal jumlah votes", min_value=0, value=1000)

if st.button("Rekomendasikan"):
    hasil = recommend(judul_input, n, min_rating, min_votes)

    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.success(f"Berikut adalah {len(hasil)} film yang mirip dengan '{judul_input}' 🎉")
        st.dataframe(hasil, use_container_width=True)
