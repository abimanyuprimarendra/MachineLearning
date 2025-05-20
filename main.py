import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import re

# Load dataset dari Google Drive CSV
@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['director'] = data['director'].fillna('Unknown')
    data['country'] = data['country'].fillna('Unknown')
    data['release_year'] = pd.to_numeric(data['release_year'], errors='coerce')
    data['release_year'] = data['release_year'].fillna(data['release_year'].median())
    data['title'] = data['title'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x))
    data['genres'] = data['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')])
    return data

def extract_duration(row):
    dur = str(row['duration'])
    tipe = row['type'].strip().lower()
    match = re.search(r'(\d+)', dur)
    if not match:
        return 0
    jumlah = int(match.group(1))
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
    if filter_type is not None:
        df_filtered = df[df['type'].str.lower() == filter_type.lower()]
    else:
        df_filtered = df

    matches = df_filtered[df_filtered['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        return f"Film '{movie_title}' tidak ditemukan dalam tipe '{filter_type or 'Movie/TV Show'}'!"
    
    idx = matches.index[0]
    distances, indices = knn.kneighbors(X[idx], n_neighbors=n_recommendations+1)
    
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

# --- Streamlit UI ---

st.title("🎬 Sistem Rekomendasi Film Netflix")

# Load data
df = load_data_from_drive()
df, X = prepare_data(df)
knn = build_model(X)

# Pilih tipe film
type_options = ['Movie', 'TV Show']
filter_type = st.selectbox("Pilih Tipe Film", type_options)

# Filter film berdasarkan tipe
filtered_titles = sorted(df[df['type'].str.lower() == filter_type.lower()]['title'].unique())
selected_title = st.selectbox("Pilih Film", filtered_titles)

# Tombol rekomendasi
if st.button("Rekomendasikan Film Mirip"):
    results = recommend(df, knn, X, selected_title, filter_type=filter_type)
    if isinstance(results, str):
        st.warning(results)
    else:
        st.write(f"Rekomendasi film mirip dengan **{selected_title}** (Tipe: {filter_type}):")
        for i, rec in enumerate(results, 1):
            st.markdown(f"**{i}. {rec['Title']}** ({rec['Year']}) - {rec['Type']}")
            st.markdown(f"- Director: {rec['Director']}")
            st.markdown(f"- Genres: {rec['Genres']}")
            st.markdown(f"- Similarity Score: {rec['Similarity']:.4f}")
            st.markdown("---")
