import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

def load_data_from_drive():
    file_id = '13iDxqKf2Jh9CpYSfXOQ76dEMfoUnRs89'  # ganti sesuai file ID
    url = f'https://drive.google.com/uc?id={file_id}&export=download'
    df = pd.read_excel(url)

    df['listed_in'] = df.get('listed_in', pd.Series()).fillna('')
    df['director'] = df.get('director', pd.Series()).fillna('Unknown')
    df['country'] = df.get('country', pd.Series()).fillna('Unknown')

    if 'release_year' in df.columns:
        release_year_num = pd.to_numeric(df['release_year'], errors='coerce')
        median_year = release_year_num.dropna().median()
        if pd.isna(median_year):
            median_year = 2000
        df['release_year'] = release_year_num.fillna(median_year)
    else:
        df['release_year'] = 2000

    if 'duration_min' in df.columns:
        df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce').fillna(df['duration_min'].median())
    else:
        df['duration_min'] = 90

    return df

def preprocess_features(df):
    cols_to_combine = ['director', 'country', 'listed_in', 'genres']
    for col in cols_to_combine:
        if col not in df.columns:
            df[col] = ''
    df['combined_features'] = df[cols_to_combine].astype(str).agg(' '.join, axis=1)
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

def get_recommendations(title, tipe, n=5):
    if not isinstance(title, str) or title.strip() == '':
        return None

    df = df_full[df_full['type'].str.lower() == tipe.lower()]
    df = df[df['title'].notna()]
    df = df[df['title'].apply(lambda x: isinstance(x, str))]

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
            'Genres': movie['genres'] if isinstance(movie['genres'], str) else ', '.join(movie['genres']),
            'Similarity': similarity
        })
    return pd.DataFrame(recs)


# === MAIN ===
df_full = load_data_from_drive()
df_full = preprocess_features(df_full)

knn, X = build_model(df_full)

# === STREAMLIT UI ===
st.title("🎬 Rekomendasi Film Netflix")

tipe_pilihan = st.selectbox("Pilih Tipe:", ['Movie', 'TV Show'])

# Input pencarian film dengan teks bebas
search_title = st.text_input("Cari film (ketik judul lengkap atau sebagian):")

if search_title:
    filtered_titles = df_full[
        df_full['title'].str.contains(search_title, case=False, na=False) &
        (df_full['type'].str.lower() == tipe_pilihan.lower())
    ]['title'].unique()
    if filtered_titles.size > 0:
        film_selected = st.selectbox("Hasil pencarian:", sorted(filtered_titles))
    else:
        st.warning("Tidak ditemukan film dengan kata kunci tersebut.")
        film_selected = None
else:
    film_list = sorted(df_full[df_full['type'].str.lower() == tipe_pilihan.lower()]['title'].unique())
    film_selected = st.selectbox("Pilih Judul Film:", film_list)

if film_selected:
    film_info = df_full[df_full['title'] == film_selected].iloc[0]
    st.markdown(f"**Judul:** {film_info['title']}")
    st.markdown(f"**Tipe:** {film_info['type']}")
    st.markdown(f"**Tahun Rilis:** {film_info['release_year']}")
    st.markdown(f"**Durasi (menit):** {film_info['duration_min']}")
    st.markdown(f"**Direktur:** {film_info['director']}")
    st.markdown(f"**Negara:** {film_info['country']}")
    st.markdown(f"**Genre:** {film_info['genres']}")

    if st.button("Tampilkan Rekomendasi"):
        with st.spinner("Mencari rekomendasi..."):
            hasil = get_recommendations(film_selected, tipe_pilihan, n=5)

        if hasil is None or hasil.empty:
            st.warning("Film tidak ditemukan atau tidak ada rekomendasi.")
        else:
            st.subheader(f"Hasil Rekomendasi untuk '{film_selected}'")
            st.dataframe(hasil)

            st.subheader("🔍 Skor Kemiripan")
            st.bar_chart(hasil.set_index('Title')['Similarity'])

            st.subheader("🎭 Distribusi Genre")
            all_genres = hasil['Genres'].str.split(', ').explode()
            genre_count = all_genres.value_counts().head(10)
            fig1, ax1 = plt.subplots()
            genre_count.plot(kind='barh', color='skyblue', ax=ax1)
            ax1.set_xlabel("Jumlah")
            ax1.set_ylabel("Genre")
            st.pyplot(fig1)

            st.subheader("📊 Tipe Film")
            tipe_count = hasil['Type'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(tipe_count, labels=tipe_count.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)

            st.success("Rekomendasi selesai ditampilkan ✅")
