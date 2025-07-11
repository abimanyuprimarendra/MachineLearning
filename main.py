import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors
import requests
import io

# ===============================
# Ambil dataset dari Google Drive via file ID
# ===============================
@st.cache_data
def load_data_from_gdrive():
    file_id = "1s9bogJ0B0rhGgYvTNUjaobEbxL4UfkhZ"
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Gagal mengambil data dari Google Drive.")
        return None
    return pd.read_csv(io.BytesIO(response.content))

# ===============================
# Preprocessing
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return ' '.join(tokens)

# ===============================
# Fungsi rekomendasi
# ===============================
def get_recommendations_verbose(title, df, tfidf_matrix, knn_model, n=10):
    try:
        idx = df[df['movie title'].str.lower() == title.lower()].index[0]
        distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)

        results = []
        for i in range(1, len(indices[0])):
            film_index = indices[0][i]
            distance = distances[0][i]
            similarity = 1 - distance

            film_title = df.iloc[film_index]['movie title']
            rating = df.iloc[film_index].get('Rating', '-')

            raw_genre = df.iloc[film_index]['Generes']
            try:
                genre_list = eval(raw_genre) if isinstance(raw_genre, str) else raw_genre
                genre = ', '.join(genre_list)
            except:
                genre = raw_genre

            overview = df.iloc[film_index].get('Overview', '')

            results.append({
                'Rank': i,
                'Judul': film_title,
                'Genre': genre,
                'Rating': rating,
                'Deskripsi': overview,
                'Jarak': round(distance, 4),
                'Similarity': round(similarity, 4)
            })

        return pd.DataFrame(results)

    except IndexError:
        return None

# ===============================
# Streamlit App
# ===============================
st.title("🎬 Sistem Rekomendasi Film - Content-Based + KNN")

df = load_data_from_gdrive()

if df is not None:
    st.success("✅ Dataset berhasil dimuat!")

    # Gabungkan fitur
    features = ['movie title', 'Generes', 'Director', 'Writer']
    for feature in features:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df.apply(lambda row: ' '.join(row[feature] for feature in features), axis=1)
    df['clean_text'] = df['combined_features'].apply(preprocess_text)

    # TF-IDF dan KNN
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(tfidf_matrix)

    # Input judul film
    judul_input = st.text_input("Masukkan judul film:", value="Spider-Man")

    if st.button("🔍 Rekomendasikan"):
        rekomendasi_df = get_recommendations_verbose(judul_input, df, tfidf_matrix, knn_model)

        if rekomendasi_df is not None:
            st.subheader("🎯 Rekomendasi Film Serupa:")
            st.dataframe(rekomendasi_df)

            # Barplot visualisasi
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Judul', y='Jarak', data=rekomendasi_df, palette='magma', ax=ax)
            ax.set_title(f'Cosine Distance Film Mirip "{judul_input}"', fontsize=14)
            ax.set_xlabel("Judul Film")
            ax.set_ylabel("Distance")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ Film '{judul_input}' tidak ditemukan dalam dataset.")
else:
    st.error("❌ Gagal memuat data.")
