import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer

# Pastikan nltk resources tersedia
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

lemmatizer = WordNetLemmatizer()

def preprocess_text_lemmatize(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) > 2]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [t for t in tokens if t not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]
    return ' '.join(lemmatized_tokens)

@st.cache_data(show_spinner=True)
def load_data():
    file_path = 'https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-'
    df = pd.read_csv(file_path)
    for col in ['director', 'country', 'listed_in']:
        df[col] = df[col].fillna('')
        df[col] = df[col].apply(preprocess_text_lemmatize)
    df['combined_features'] = df.apply(lambda row: ' '.join([row['director'], row['country'], row['listed_in']]), axis=1)
    return df

@st.cache_data(show_spinner=True)
def create_tfidf_cosine_knn(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim, knn

def get_recommendations(title, cosine_sim, df, n_neighbors=5):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return None
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_neighbors+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'director', 'release_year', 'rating']]

def main():
    st.title("Sistem Rekomendasi Film Netflix (Content-Based Filtering)")
    df = load_data()
    tfidf, tfidf_matrix, cosine_sim, knn = create_tfidf_cosine_knn(df)

    movie_list = df['title'].tolist()
    selected_movie = st.selectbox("Pilih film untuk direkomendasikan:", movie_list)

    n_neighbors = st.slider("Jumlah rekomendasi yang diinginkan:", 3, 10, 5)

    if st.button("Cari Rekomendasi"):
        recommendations = get_recommendations(selected_movie, cosine_sim, df, n_neighbors)
        if recommendations is not None:
            st.subheader(f"Rekomendasi film mirip dengan '{selected_movie}':")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.error("Judul film tidak ditemukan di dataset.")

if __name__ == '__main__':
    main()
