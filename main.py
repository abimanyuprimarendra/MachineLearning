import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import re

# --- Load data dan preprocessing ---
@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1snEw9T4JdILb2_QVSbnXfHC5uoki_tOm"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['director'] = data['director'].fillna('Unknown')
    data['country'] = data['country'].fillna('Unknown')
    data['release_year'] = pd.to_numeric(data['release_year'], errors='coerce')
    data['release_year'] = data['release_year'].fillna(data['release_year'].median())
    data['title'] = data['title'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data['genres'] = data['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])
    data['type'] = data['type'].str.strip().str.lower()
    return data

def extract_duration(row):
    dur = str(row['duration'])
    tipe = row['type'].strip().lower()
    match = re.search(r'(\d+)', dur)
    if not match:
        return 0
    jumlah = int(match.group(1))
    # TV Show diberikan bobot lebih besar agar bisa dibedakan
    if tipe == 'movie':
        return jumlah
    elif tipe == 'tv show':
        return jumlah * 400
    return 0

def combine_features(row):
    genres_str = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
    director_str = str(row['director']).replace(' ', '')
    country_str = str(row['country']).replace(' ', '')
    return f"{genres_str} {director_str} {country_str}"

@st.cache_data(show_spinner=False)
def prepare_data(df):
    df['duration_min'] = df.apply(extract_duration, axis=1)
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[['release_year', 'duration_min']])
    
    X = hstack([tfidf_matrix, num_features])
    X = csr_matrix(X)
    return df, X

@st.cache_data(show_spinner=False)
def build_model(X):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(X)
    return knn

def recommend(df, knn, X, movie_title, n_recommendations=5, filter_type=None):
    # Filter berdasarkan tipe jika ada filter_type
    if filter_type is not None:
        df_filtered = df[df['type'].str.lower() == filter_type.lower()]
        if df_filtered.empty:
            return f"Tidak ada film dengan tipe '{filter_type}' ditemukan."
    else:
        df_filtered = df

    # Cari movie_title dalam df_filtered, case-insensitive
    matches = df_filtered[df_filtered['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return f"Film '{movie_title}' tidak ditemukan dalam tipe '{filter_type or 'Movie/TV Show'}'!"
    
    idx = matches.index[0]
    distances, indices = knn.kneighbors(X[idx], n_neighbors=n_recommendations + 1)
    
    recs = []
    for i in range(1, len(indices[0])):
        rec_idx = indices[0][i]
        movie = df.iloc[rec_idx]
        similarity = 1 - distances[0][i]
        recs.append({
            'Title': movie['title'],
            'Director': movie['director'],
            'Year': int(movie['release_year']),
            'Genres': ', '.join(movie['genres']),
            'Type': movie['type'],
            'Similarity': similarity
        })
    return recs

# === MAIN APP ===
st.title("Rekomendasi Film Mirip Netflix")

# Load & siapkan data (cached)
df = load_data_from_drive()
df, X = prepare_data(df)
knn = build_model(X)

# Dropdown tipe film
type_options = ['movie', 'tv show']

# Session state setup default tipe film
if 'selected_type' not in st.session_state:
    st.session_state.selected_type = 'movie'

selected_type = st.selectbox(
    "Pilih Tipe Film",
    options=[t.title() for t in type_options],
    index=type_options.index(st.session_state.selected_type),
    key='selected_type'
)

def get_filtered_titles(selected_type_lower):
    titles = sorted(df[df['type'] == selected_type_lower]['title'].unique())
    return titles

filtered_titles = get_filtered_titles(selected_type.lower())

if not filtered_titles:
    st.warning("Tidak ada film ditemukan untuk tipe ini.")
else:
    # Update session state default judul film supaya tidak error
    if 'selected_title' not in st.session_state or st.session_state.selected_title not in filtered_titles:
        st.session_state.selected_title = filtered_titles[0]

    selected_title = st.selectbox(
        "Pilih Film",
        options=filtered_titles,
        index=filtered_titles.index(st.session_state.selected_title),
        key='selected_title'
    )

    if st.button("Rekomendasikan Film Mirip"):
        results = recommend(df, knn, X, selected_title, filter_type=selected_type)
        if isinstance(results, str):
            st.warning(results)
        else:
            st.markdown(f"### Rekomendasi film mirip dengan **{selected_title}** (Tipe: {selected_type.title()})")
            for i, rec in enumerate(results, 1):
                st.markdown(f"**{i}. {rec['Title']}** ({rec['Year']}) - {rec['Type'].title()}")
                st.markdown(f"- Director: {rec['Director']}")
                st.markdown(f"- Genres: {rec['Genres']}")
                st.markdown(f"- Similarity Score: {rec['Similarity']:.4f}")
                st.markdown("---")
