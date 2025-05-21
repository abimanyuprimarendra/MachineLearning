import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from numpy import log1p
import difflib
import plotly.express as px

# Load dataset dari Google Drive
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

# Siapkan model TF-IDF dan KNN
@st.cache_resource
def prepare_model(df):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, knn

# Fungsi rekomendasi
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
                df.iloc[rec_idx]['genre'],
                round(similarity, 3),
                rating,
                round(score, 4),
                df.iloc[rec_idx]['description'][:150] + '...',
                f"{votes:,}",
                df.iloc[rec_idx].get('poster_url', '')  # Kolom URL gambar
            ))
            added_titles.add(rec_title.lower())

        if len(recommendations) == n_recommendations:
            break

    if not recommendations:
        return "Tidak ada film yang direkomendasikan berdasarkan kriteria.", None

    recommendations = sorted(recommendations, key=lambda x: x[4], reverse=True)
    df_result = pd.DataFrame(recommendations, columns=[
        'Title', 'Genre', 'Similarity', 'Rating', 'Score', 'Description', 'Votes', 'Poster'
    ])
    return None, df_result

# =======================
# Streamlit App
# =======================
st.title("🎬 Sistem Rekomendasi Film")

df = load_data()
if not df.empty:
    vectorizer, tfidf_matrix, knn = prepare_model(df)

    title_input = st.text_input("Masukkan judul film", "Cobra Kai")
    n = st.slider("Jumlah rekomendasi", 1, 20, 10)
    min_rating = st.slider("Minimal rating", 0.0, 10.0, 7.0)
    min_votes = st.number_input("Minimal jumlah votes", min_value=0, value=1000)

    if st.button("Rekomendasikan"):
        error_msg, hasil = recommend(title_input, n, min_rating, min_votes)
        if error_msg:
            st.warning(error_msg)
        else:
            st.success(f"Berikut adalah {len(hasil)} film mirip '{title_input}' 🎉")

            # 🔳 VISUALISASI 5 TERATAS DENGAN GAMBAR POSTER
            st.markdown("## 🎥 Rekomendasi Visual")
            top5 = hasil.head(5)
            for _, row in top5.iterrows():
                st.markdown(f"### 🎬 {row['Title']}")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if row['Poster']:
                        st.image(row['Poster'], width=120)
                    else:
                        st.text("Poster tidak tersedia")
                with col2:
                    st.markdown(f"**Genre:** {row['Genre']}")
                    st.markdown(f"**Rating:** {row['Rating']} ⭐  |  **Votes:** {row['Votes']}")
                    st.markdown(f"**Score:** {row['Score']}")
                    st.markdown(f"**Deskripsi:** {row['Description']}")
                st.markdown("---")

            # 🔍 VISUALISASI GRAFIK
            st.subheader("📊 Visualisasi Data Rekomendasi")

            fig_score = px.bar(
                top5,
                x='Title',
                y='Score',
                color='Score',
                color_continuous_scale='viridis',
                title='🔢 Skor Rekomendasi Film',
                labels={'Score': 'Skor', 'Title': 'Judul'}
            )
            st.plotly_chart(fig_score, use_container_width=True)

            fig_scatter = px.scatter(
                top5,
                x='Similarity',
                y='Rating',
                text='Title',
                size='Score',
                color='Score',
                title='🎯 Similarity vs Rating',
                labels={'Similarity': 'Kemiripan', 'Rating': 'Rating'}
            )
            fig_scatter.update_traces(textposition='top center')
            st.plotly_chart(fig_scatter, use_container_width=True)

            if top5['Genre'].nunique() > 1:
                genre_counts = top5['Genre'].value_counts().reset_index()
                genre_counts.columns = ['Genre', 'Count']
                fig_pie = px.pie(
                    genre_counts,
                    names='Genre',
                    values='Count',
                    title='📊 Genre Film yang Direkomendasikan'
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Tabel Data
            st.markdown("## 📋 Tabel Data")
            st.dataframe(hasil.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
else:
    st.stop()
