import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    file_path = 'https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-'
    df = pd.read_csv(file_path)
    df.fillna('', inplace=True)
    # Gabungkan kolom yang ingin dijadikan fitur
    df['combined_features'] = df['director'] + ' ' + df['country'] + ' ' + df['listed_in']
    return df

@st.cache_data
def create_tfidf_cosine(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim, df, n=5):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'director', 'release_year', 'rating']]

def main():
    st.title("Rekomendasi Film Netflix (Simple Version)")

    df = load_data()
    cosine_sim = create_tfidf_cosine(df)

    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Pilih film:", movie_list)
    n_recs = st.slider("Jumlah rekomendasi:", 3, 10, 5)

    if st.button("Cari Rekomendasi"):
        recs = get_recommendations(selected_movie, cosine_sim, df, n_recs)
        if recs is not None:
            st.subheader(f"Film mirip dengan '{selected_movie}':")
            st.dataframe(recs.reset_index(drop=True))
        else:
            st.error("Judul film tidak ditemukan.")

if __name__ == '__main__':
    main()
