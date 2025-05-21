import streamlit as st
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import log1p
import difflib

# Load dataset
@st.cache_data
def load_data():
    csv_url = "https://drive.google.com/uc?id=1ix27-hPzSIjBrZGI5fl3HP5QFlJlDY0K"
    try:
        df = pd.read_csv(csv_url)
        df['votes'] = df['votes'].fillna('0').str.replace(',', '', regex=False).astype(int)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        for col in ['genre', 'description', 'stars']:
            df[col] = df[col].fillna('').apply(clean_text)

        df['combined_features'] = (
            (df['genre'] + ' ') * 2 +
            (df['stars'] + ' ') * 2 +
            df['description']
        )
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# Fetch poster using OMDb API
def fetch_poster_url(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("Response") == "True":
            return data.get("Poster", "")
        else:
            return ""
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        return ""

# Prepare model
@st.cache_resource
def prepare_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    return vectorizer, tfidf_matrix, knn

# Recommendation logic
def recommend(title, n_recommendations=5, min_rating=7, min_votes=1000, api_key=None):
    idx_list = df.index[df['title'].str.lower() == title.lower()]
    if len(idx_list) == 0:
        closest_matches = difflib.get_close_matches(title, df['title'], n=3)
        return f"Film tidak ditemukan! Coba: {', '.join(closest_matches)}", None

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
            poster_url = fetch_poster_url(rec_title, api_key) if api_key else ""
            recommendations.append({
                'Title': rec_title,
                'Genre': df.iloc[rec_idx]['genre'],
                'Similarity': round(similarity, 3),
                'Rating': rating,
                'Score': round((similarity * 0.5) + (rating / 10 * 0.3) + (log1p(votes) / 10 * 0.2), 4),
                'Description': df.iloc[rec_idx]['description'][:150] + '...',
                'Votes': f"{votes:,}",
                'Poster': poster_url
            })
            added_titles.add(rec_title.lower())

        if len(recommendations) == n_recommendations:
            break

    if not recommendations:
        return "Tidak ada film yang direkomendasikan berdasarkan kriteria.", None

    return None, recommendations

# ========================
# Streamlit App
# ========================
st.set_page_config(page_title="🎬 Sistem Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")

df = load_data()
if not df.empty:
    vectorizer, tfidf_matrix, knn = prepare_model(df)

    st.sidebar.header("🔐 Pengaturan")
    api_key = st.sidebar.text_input("Masukkan OMDb API Key", type="password")

    st.subheader("Masukkan Judul Film")
    title_input = st.text_input("Judul film", "Cobra Kai")
    n = st.slider("Jumlah rekomendasi", 1, 10, 5)
    min_rating = st.slider("Minimal rating", 0.0, 10.0, 7.0)
    min_votes = st.number_input("Minimal jumlah votes", min_value=0, value=1000)

    if st.button("Rekomendasikan"):
        if not api_key:
            st.warning("Masukkan OMDb API Key di sidebar untuk mendapatkan gambar poster!")
        error_msg, hasil = recommend(title_input, n, min_rating, min_votes, api_key)
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"Berikut {len(hasil)} rekomendasi film mirip '{title_input}'")
            for film in hasil:
                cols = st.columns([1, 3])
                with cols[0]:
                    if film['Poster']:
                        st.image(film['Poster'], width=120)
                    else:
                        st.markdown("❌ Poster tidak ditemukan")
                with cols[1]:
                    st.markdown(f"**🎞️ {film['Title']}**  \n"
                                f"📚 *{film['Genre']}*  \n"
                                f"⭐ Rating: {film['Rating']}  \n"
                                f"👥 Votes: {film['Votes']}  \n"
                                f"📊 Skor Rekomendasi: {film['Score']}  \n"
                                f"📝 _{film['Description']}_")
                    st.markdown("---")
else:
    st.stop()
