import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Fungsi load data dari Google Drive dengan caching agar efisien
@st.cache_data
def load_data_from_drive():
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    data['description'] = data['description'].fillna('') if 'description' in data.columns else ''
    data['combined'] = data['title'] + " " + data['listed_in'] + " " + data['description']
    return data

# Membuat matriks TF-IDF dari gabungan kolom teks
@st.cache_data
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return tfidf_matrix, tfidf # Mengembalikan tfidf juga, jika dibutuhkan untuk metrik lain

# Membuat model KNN dengan metric 'cosine'
@st.cache_resource
def create_knn_cosine_model():
    # Metric 'cosine' artinya menggunakan jarak cosine (bukan similarity langsung)
    return NearestNeighbors(metric='cosine', algorithm='brute')

# Membuat model KNN dengan metric 'euclidean'
@st.cache_resource
def create_knn_euclidean_model():
    # Metric 'euclidean' artinya menggunakan jarak euclidean
    return NearestNeighbors(metric='euclidean', algorithm='brute')

# Fungsi rekomendasi berbasis cosine similarity full matrix
def get_content_based_recommendations(title, cosine_sim, df, top_n=5):
    """
    Cosign Similarity:
    Mengukur kemiripan dua vektor teks (film) dengan rumus:
    cosine_sim(A, B) = (A dot B) / (||A|| * ||B||)
    Menghasilkan nilai antara 0 (tidak mirip) sampai 1 (identik).
    """
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Ambil top N selain diri sendiri
    return [(df['title'].iloc[i], score) for i, score in sim_scores]

# Fungsi rekomendasi berbasis KNN dengan jarak cosine
def get_knn_cosine_recommendations(title, knn_model, df, tfidf_matrix, top_n=5):
    """
    KNN dengan metric 'cosine' menghitung jarak cosine:
    distance = 1 - cosine_similarity
    Model mencari tetangga terdekat berdasarkan jarak ini.
    Skor similarity dikembalikan dengan membalik jarak:
    similarity = 1 - distance
    """
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended_indices = indices_knn.flatten()[1:]  # Kecuali diri sendiri
    distances = distances.flatten()[1:]
    return [(df['title'].iloc[i], 1 - dist) for i, dist in zip(recommended_indices, distances)]

# Fungsi rekomendasi berbasis KNN dengan jarak Euclidean
def get_knn_euclidean_recommendations(title, knn_model, df, tfidf_matrix, top_n=5):
    """
    KNN dengan metric 'euclidean' menghitung jarak Euclidean.
    Skor similarity dihitung sebagai 1 / (1 + distance) agar nilai lebih besar berarti lebih mirip.
    """
    if title not in df['title'].values:
        return []
    idx = df[df['title'] == title].index[0]
    item_vector = tfidf_matrix[idx]
    distances, indices_knn = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    recommended_indices = indices_knn.flatten()[1:]
    distances = distances.flatten()[1:]
    
    # Konversi jarak Euclidean ke skor "kemiripan" (yang lebih besar berarti lebih mirip)
    # Salah satu cara umum: 1 / (1 + distance)
    return [(df['title'].iloc[i], 1 / (1 + dist)) for i, dist in zip(recommended_indices, distances)]


# Streamlit UI utama
st.title("Perbandingan Rekomendasi Film: KNN vs Cosine Similarity")

df = load_data_from_drive()
tfidf_matrix, _ = create_tfidf_matrix(df) # tfidf_vectorizer tidak digunakan di sini, jadi diabaikan
cosine_sim = cosine_similarity(tfidf_matrix)

# Inisialisasi kedua model KNN
knn_cosine_model = create_knn_cosine_model()
knn_cosine_model.fit(tfidf_matrix)

knn_euclidean_model = create_knn_euclidean_model()
knn_euclidean_model.fit(tfidf_matrix)

title = st.selectbox("Pilih judul film:", options=df['title'].sort_values().unique())
selected_knn_metric = st.selectbox("Pilih Metrik KNN:", options=["Cosine", "Euclidean"])

if title:
    cosine_recs = get_content_based_recommendations(title, cosine_sim, df)
    
    knn_recs = []
    knn_header_text = ""

    if selected_knn_metric == "Cosine":
        knn_recs = get_knn_cosine_recommendations(title, knn_cosine_model, df, tfidf_matrix)
        knn_header_text = "KNN Recommendation (Cosine Metric)"
    elif selected_knn_metric == "Euclidean":
        knn_recs = get_knn_euclidean_recommendations(title, knn_euclidean_model, df, tfidf_matrix)
        knn_header_text = "KNN Recommendation (Euclidean Metric)"

    # Tampilkan rekomendasi berdampingan
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cosine Similarity")
        if cosine_recs:
            for i, (rec_title, score) in enumerate(cosine_recs, 1):
                st.write(f"{i}. {rec_title} (similarity: {score:.4f})")
        else:
            st.write("Tidak ada rekomendasi yang ditemukan.")

    with col2:
        st.subheader(knn_header_text)
        if knn_recs:
            for i, (rec_title, score) in enumerate(knn_recs, 1):
                st.write(f"{i}. {rec_title} (similarity: {score:.4f})")
        else:
            st.write("Tidak ada rekomendasi yang ditemukan.")

    ---
    st.subheader("Visualisasi Perbandingan Skor Kemiripan")
    
    # Kumpulkan data untuk visualisasi
    combined_data = pd.DataFrame({
        'Film': [rec[0] for rec in cosine_recs] + [rec[0] for rec in knn_recs],
        'Similarity': [rec[1] for rec in cosine_recs] + [rec[1] for rec in knn_recs],
        'Metode': ['Cosine Similarity'] * len(cosine_recs) + [knn_header_text.replace(" Recommendation", "")] * len(knn_recs)
    })

    if not combined_data.empty:
        plt.figure(figsize=(12, 6)) # Ukuran plot lebih besar
        sns.barplot(data=combined_data, x='Similarity', y='Film', hue='Metode', palette='viridis')
        plt.title('Perbandingan Skor Kemiripan Rekomendasi')
        plt.xlabel('Skor Kemiripan')
        plt.ylabel('Judul Film')
        plt.tight_layout() # Mengatur layout agar label tidak tumpang tindih
        st.pyplot(plt.gcf())
    else:
        st.write("Tidak ada data untuk visualisasi.")

    ---
    st.subheader("Analisis Metrik Perbandingan")
    
    if cosine_recs and knn_recs:
        cosine_titles = set([title for title, _ in cosine_recs])
        knn_titles = set([title for title, _ in knn_recs])
        
        common_titles = list(cosine_titles & knn_titles)
        num_common = len(common_titles)
        
        # Ambil K yang sama dari kedua metode (biasanya top_n yang Anda set di fungsi)
        # Ambil minimum dari panjang list rekomendasi untuk menghindari indeks error
        k_value = min(len(cosine_recs), len(knn_recs)) 
        
        if k_value > 0:
            # Tingkat Kesamaan Rekomendasi (Overlap)
            overlap_percentage = (num_common / k_value) * 100
            st.write(f"**Tingkat Kesamaan Rekomendasi (Overlap):** **{overlap_percentage:.2f}%**")
            st.write(f"({num_common} dari {k_value} rekomendasi teratas yang sama)")
        else:
            st.write("Tidak ada rekomendasi untuk dianalisis (k_value = 0).")

        # Statistik Skor Kemiripan
        st.write("#### Statistik Skor Kemiripan:")
        cosine_scores_list = [score for _, score in cosine_recs]
        knn_scores_list = [score for _, score in knn_recs]
        
        if cosine_scores_list:
            cosine_scores_series = pd.Series(cosine_scores_list)
            st.write(f"**Cosine Similarity:**")
            st.write(f"- Min: {cosine_scores_series.min():.4f}, Max: {cosine_scores_series.max():.4f}, Rata-rata: {cosine_scores_series.mean():.4f}")
        else:
            st.write("- Cosine Similarity: Tidak ada skor yang tersedia.")

        if knn_scores_list:
            knn_scores_series = pd.Series(knn_scores_list)
            st.write(f"**{knn_header_text.replace(' Recommendation', '')}:**")
            st.write(f"- Min: {knn_scores_series.min():.4f}, Max: {knn_scores_series.max():.4f}, Rata-rata: {knn_scores_series.mean():.4f}")
        else:
            st.write(f"- {knn_header_text.replace(' Recommendation', '')}: Tidak ada skor yang tersedia.")

    else:
        st.write("Tidak cukup rekomendasi untuk analisis perbandingan.")
