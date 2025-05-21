import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import log1p
import difflib

# Load dataset dan cek kolom
@st.cache_data
def load_data():
    csv_url = "https://drive.google.com/uc?id=1ix27-hPzSIjBrZGI5fl3HP5QFlJlDY0K"
    df = pd.read_csv(csv_url)
    return df

df = load_data()

# Tampilkan nama kolom dan beberapa baris awal untuk cek struktur data
st.write("Kolom pada dataset:", df.columns.tolist())
st.write("Contoh data:")
st.write(df.head())

# Pastikan kolom votes ada, kalau tidak buat kolom votes default 0
if 'votes' in df.columns:
    df['votes'] = df['votes'].fillna('0').astype(str)
    df['votes'] = df['votes'].str.replace(',', '', regex=False)
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
else:
    st.warning("Kolom 'votes' tidak ditemukan di dataset. Membuat kolom votes dengan nilai 0.")
    df['votes'] = 0

# Pastikan kolom rating ada, kalau tidak buat kolom rating default 0
if 'rating' in df.columns:
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
else:
    st.warning("Kolom 'rating' tidak ditemukan di dataset. Membuat kolom rating dengan nilai 0.")
    df['rating'] = 0

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for col in ['genre', 'description', 'stars']:
    if col in df.columns:
        df[col] = df[col].fillna('').apply(clean_text)
    else:
        st.warning(f"Kolom '{col}' tidak ditemukan di dataset. Membuat kolom kosong.")
        df[col] = ''

df['combined_features'] = (
    (df['genre'] + ' ') * 2 +
    (df['stars'] + ' ') * 2 +
    df['description']
)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(tfidf_matrix)

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
        return "Tidak ada film yang direkomendasikan berdasarkan kriteria.", None

    recommendations = sorted(recommendations, key=lambda x: x[4], reverse=True)
    df_result = pd.DataFrame(recommendations, columns=[
        'Title', 'Similarity', 'Rating', 'Votes', 'Score', 'Description'
    ])
    return None, df_result

st.title("🎬 Sistem Rekomendasi Film")

if df.empty:
    st.error("Data gagal dimuat atau kosong.")
    st.stop()

film_list = df['title'].dropna().sort_values().unique()
default_idx = 0
if "Cobra Kai" in film_list:
    default_idx = film_list.tolist().index("Cobra Kai")

title_input = st.selectbox("Pilih judul film", film_list, index=default_idx)

n = st.slider("Jumlah rekomendasi", 1, 20, 10)
min_rating = st.slider("Minimal rating", 0.0, 10.0, 7.0)
min_votes = st.number_input("Minimal jumlah votes", min_value=0, value=1000)

if st.button("Rekomendasikan"):
    error_msg, hasil = recommend(title_input, n, min_rating, min_votes)
    if error_msg:
        st.warning(error_msg)
    else:
        st.success(f"Berikut adalah {len(hasil)} film mirip '{title_input}' 🎉")
        st.dataframe(hasil.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
