# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import re
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st

import gdown
import tempfile
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_data_from_drive():
    file_id = '13iDxqKf2Jh9CpYSfXOQ76dEMfoUnRs89'  # ganti dengan ID file kamu
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Buat temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = os.path.join(tmp_dir, 'Netflix.xlsx')
        gdown.download(url, temp_path, quiet=True)
        df = pd.read_excel(temp_path)

    # Preprocessing
    df['listed_in'] = df['listed_in'].fillna('')
    df['director'] = df['director'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(df['release_year'].median())
    # Jangan lupa fitur lain seperti 'duration_min', 'combined_features' sesuai kode utama
    
    return df

    return df

# Contoh panggilan
df_full = load_data_from_drive()

# === TF-IDF dan KNN ===
@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])

    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(data[['release_year', 'duration_min']])

    X = hstack([tfidf_matrix, num_features])
    X = csr_matrix(X)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(X)
    return knn, X

knn, X = build_model(df_full)

# === Rekomendasi ===
def get_recommendations(title, tipe, n=5):
    df = df_full[df_full['type'].str.lower() == tipe.lower()]
    matches = df[df['title'].str.lower() == title.lower()]
    if matches.empty:
        return None

    idx = matches.index[0]
    distances, indices = knn.kneighbors(X[idx], n_neighbors=n+1)

    recs = []
    for i in range(1, len(indices[0])):
        rec_idx = indices[0][i]
        movie = df_full.iloc[rec_idx]
        similarity = 1 - distances[0][i]
        recs.append({
            'Title': movie['title'],
            'Release Year': int(movie['release_year']),
            'Type': movie['type'],
            'Director': movie['director'],
            'Genres': ', '.join(movie['genres']),
            'Similarity': similarity
        })
    return pd.DataFrame(recs)

# === STREAMLIT UI ===
st.title("🎬 Rekomendasi Film Netflix")

tipe_pilihan = st.selectbox("Pilih Tipe:", ['Movie', 'TV Show'])
film_list = sorted(df_full[df_full['type'] == tipe_pilihan]['title'].unique())
film_selected = st.selectbox("Pilih Judul Film:", film_list)

if st.button("Tampilkan Rekomendasi"):
    hasil = get_recommendations(film_selected, tipe_pilihan, n=5)
    
    if hasil is None:
        st.warning("Film tidak ditemukan.")
    else:
        st.subheader(f"Hasil Rekomendasi untuk '{film_selected}'")
        st.dataframe(hasil)

        # Visualisasi Similarity
        st.subheader("🔍 Skor Kemiripan")
        st.bar_chart(hasil.set_index('Title')['Similarity'])

        # Visualisasi Genre
        st.subheader("🎭 Distribusi Genre")
        all_genres = hasil['Genres'].str.split(', ').explode()
        genre_count = all_genres.value_counts().head(10)
        fig1, ax1 = plt.subplots()
        genre_count.plot(kind='barh', color='skyblue', ax=ax1)
        ax1.set_xlabel("Jumlah")
        ax1.set_ylabel("Genre")
        st.pyplot(fig1)

        # Pie chart tipe (walau akan selalu 1 tipe saat difilter)
        st.subheader("📊 Tipe Film")
        tipe_count = hasil['Type'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(tipe_count, labels=tipe_count.index, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

        st.success("Rekomendasi selesai ditampilkan ✅")

