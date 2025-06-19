import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# 1. Load Data dari GitHub
# -----------------------
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/abimanyuprimarendra/MachineLearning/refs/heads/main/dataset_film.csv"
    df = pd.read_csv(csv_url)

    # Normalisasi kolom
    df.columns = df.columns.str.strip().str.lower()
    if 'release year' in df.columns:
        df.rename(columns={'release year': 'releaseyear'}, inplace=True)

    # Validasi kolom penting
    required_cols = ['title', 'genres', 'releaseyear']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Kolom '{col}' tidak ditemukan di dataset.")
            return pd.DataFrame()

    # Pra-pemrosesan
    df = df.dropna(subset=required_cols)
    df['title'] = df['title'].str.lower().str.strip()
    df['genres'] = df['genres'].str.lower().str.strip()
    df['releaseyear'] = df['releaseyear'].astype(int)

    # Gabungkan fitur konten
    df['combined_features'] = (
        df['title'] + ' ' + df['genres'] + ' ' + df['releaseyear'].astype(str)
    )

    return df

# Load data
df = load_data()

# -----------------------
# 2. TF-IDF dan Cosine Similarity
# -----------------------
@st.cache_data
def compute_similarity(df):
    if df.empty:
        return None, None
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = compute_similarity(df)

# -----------------------
# 3. Fungsi Rekomendasi
# -----------------------
def recommend(title, n=5):
    if cosine_sim is None or indices is None:
        return None

    title = title.lower().strip()
    if title not in indices.index:
        return None

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    seen = set()
    results = []
    for i, _ in sim_scores:
        film_title = df.loc[i, 'title']
        if film_title not in seen:
            seen.add(film_title)
            results.append(i)
        if len(results) == n:
            break
    return df[['title', 'genres', 'releaseyear']].iloc[results]

# -----------------------
# 4. Tampilan Streamlit
# -----------------------
st.set_page_config(page_title="🎬 Rekomendasi Film", layout="wide")
st.title("🎬 Sistem Rekomendasi Film")
st.caption("Content-Based Filtering | Dataset dari GitHub")

if df.empty:
    st.stop()

# -----------------------
# Statistik Dataset
# -----------------------
st.subheader("📊 Statistik Umum Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Film", len(df))
col2.metric("Genre Unik", df['genres'].nunique())
col3.metric("Tahun Rilis Unik", df['releaseyear'].nunique())

# -----------------------
# Visualisasi: Distribusi Tahun
# -----------------------
st.subheader("🕰️ Distribusi Film per Tahun Rilis")
fig1, ax1 = plt.subplots()
df['releaseyear'].value_counts().sort_index().plot(kind='bar', ax=ax1)
ax1.set_xlabel("Tahun Rilis")
ax1.set_ylabel("Jumlah Film")
st.pyplot(fig1)

# -----------------------
# Visualisasi: Genre Terpopuler
# -----------------------
st.subheader("🎭 Genre Terpopuler")
genre_series = df['genres'].str.split(',').explode().str.strip()
top_genres = genre_series.value_counts().head(10)

fig2, ax2 = plt.subplots()
top_genres.plot(kind='barh', ax=ax2, color='skyblue')
ax2.set_xlabel("Jumlah Film")
ax2.set_ylabel("Genre")
st.pyplot(fig2)

# -----------------------
# Rekomendasi Film
# -----------------------
st.subheader("🎯 Rekomendasi Film")

film_options = df['title'].drop_duplicates().sort_values().str.title().tolist()
selected_title = st.selectbox("Pilih judul film", film_options)

if selected_title:
    st.markdown(f"**Hasil rekomendasi untuk:** `{selected_title}`")
    result_df = recommend(selected_title)
    if result_df is not None:
        st.success(f"{len(result_df)} rekomendasi ditemukan!")
        st.dataframe(result_df.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("❌ Film tidak ditemukan atau tidak ada rekomendasi.")
