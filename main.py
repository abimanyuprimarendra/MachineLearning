import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

@st.cache_data
def load_data_from_drive():
    csv_url = 'https://docs.google.com/spreadsheets/d/1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB/export?format=csv'
    df = pd.read_csv(csv_url)

    # Pastikan kolom tersedia dan bersih
    df['listed_in'] = df.get('listed_in', pd.Series()).fillna('')
    df['director'] = df.get('director', pd.Series()).fillna('Unknown')
    df['country'] = df.get('country', pd.Series()).fillna('Unknown')
    
    if 'release_year' in df.columns:
        release_year_num = pd.to_numeric(df['release_year'], errors='coerce')
        median_year = release_year_num.dropna().median()
        df['release_year'] = release_year_num.fillna(median_year if not np.isnan(median_year) else 2000)
    else:
        df['release_year'] = 2000

    if 'duration_min' in df.columns:
        df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce').fillna(90)
    else:
        df['duration_min'] = 90

    df['title'] = df['title'].astype(str).str.strip()
    df['type'] = df['type'].astype(str).str.strip().str.capitalize()

    if 'genres' not in df.columns:
        df['genres'] = df['listed_in'].apply(lambda x: [g.strip() for g in x.split(',')] if x else [])

    return df


def preprocess_features(df):
    df = df.copy()
    def join_genres(row):
        if isinstance(row['genres'], list):
            return ' '.join(row['genres'])
        return str(row['genres'])
    
    df['genres_str'] = df.apply(join_genres, axis=1)
    df['combined_features'] = (
        df['director'].astype(str) + ' ' +
        df['country'].astype(str) + ' ' +
        df['listed_in'].astype(str) + ' ' +
        df['genres_str']
    )
    return df

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

def get_recommendations(title, tipe, df_full, knn, X, n=5):
    df = df_full[df_full['type'].str.lower() == tipe.lower()]
    df = df[df['title'].notna()]
    
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
            'Genres': ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres'],
            'Similarity': similarity
        })
    return pd.DataFrame(recs)

# === MAIN APP ===
st.title("🎬 Rekomendasi Film Netflix")

df_full = load_data_from_drive()
st.write("🔎 Jumlah total data:", len(df_full))
st.write("📌 Tipe yang tersedia:", df_full['type'].unique())

df_full = preprocess_features(df_full)
knn, X = build_model(df_full)

tipe_pilihan = st.selectbox("Pilih Tipe:", sorted(df_full['type'].unique()))

df_filtered = df_full[df_full['type'].str.lower() == tipe_pilihan.lower()]
st.write(f"📺 Jumlah judul dengan tipe '{tipe_pilihan}': {len(df_filtered)}")

film_list = sorted(df_filtered['title'].dropna().unique())

if len(film_list) == 0:
    st.warning(f"Tidak ada judul untuk tipe '{tipe_pilihan}'.")
    film_selected = None
else:
    film_selected = st.selectbox("Pilih Judul Film:", film_list)

if film_selected:
    if st.button("Tampilkan Rekomendasi"):
        hasil = get_recommendations(film_selected, tipe_pilihan, df_full, knn, X, n=5)
        if hasil is None or hasil.empty:
            st.warning("Film tidak ditemukan atau tidak ada rekomendasi.")
        else:
            st.subheader(f"Hasil Rekomendasi untuk '{film_selected}'")
            st.dataframe(hasil)

            # Bar Chart Similarity
            st.subheader("🔍 Skor Kemiripan")
            st.bar_chart(hasil.set_index('Title')['Similarity'])

            # Genre Distribution
            st.subheader("🎭 Distribusi Genre")
            all_genres = hasil['Genres'].str.split(', ').explode()
            genre_count = all_genres.value_counts().head(10)
            fig1, ax1 = plt.subplots()
            genre_count.plot(kind='barh', color='skyblue', ax=ax1)
            ax1.set_xlabel("Jumlah")
            ax1.set_ylabel("Genre")
            st.pyplot(fig1)

            # Pie Chart Tipe
            st.subheader("📊 Tipe Film")
            tipe_count = hasil['Type'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(tipe_count, labels=tipe_count.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

            st.success("✅ Rekomendasi selesai ditampilkan.")
