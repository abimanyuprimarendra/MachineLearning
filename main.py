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
    # Menggunakan URL Google Drive yang sudah ada di versi Streamlit sebelumnya
    csv_url = "https://drive.google.com/uc?id=1cjFVBpIv9SOoyWvSmg1FgReqmdXxaxB-"
    data = pd.read_csv(csv_url)
    data['listed_in'] = data['listed_in'].fillna('')
    # Memastikan kolom 'description' ada sebelum mengisinya
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
    Mengukur kemiripan dua vektor teks (film) dengan rumus Cosine Similarity.
    Menghasilkan nilai antara 0 (tidak mirip) sampai 1 (identik).
    Mengembalikan list of tuples (judul film, skor kemiripan).
    """
    if title not in df['title'].values:
        return [] # Mengembalikan list kosong jika judul tidak ditemukan
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Ambil top N selain diri sendiri
    return [(df['title'].iloc[i], score) for i, score in sim_scores]

# Fungsi rekomendasi berbasis KNN dengan jarak cosine
def get_knn_cosine_recommendations(title, knn_model, df, tfidf_matrix, top_n=5):
    """
    KNN dengan metric 'cosine' menghitung jarak cosine.
    Skor similarity dikembalikan dengan membalik jarak (1 - distance).
    Mengembalikan list of tuples (judul film, skor kemiripan).
    """
    if title not in df['title'].values:
        return [] # Mengembalikan list kosong jika judul tidak ditemukan
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
    Mengembalikan list of tuples (judul film, skor kemiripan).
    """
    if title not in df['title'].values:
        return [] # Mengembalikan list kosong jika judul tidak ditemukan
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

# Memuat data dan membuat TF-IDF matrix
with st.spinner('Memuat data dan menyiapkan model...'):
    df = load_data_from_drive()
    tfidf_matrix, _ = create_tfidf_matrix(df)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Inisialisasi kedua model KNN
    knn_cosine_model = create_knn_cosine_model()
    knn_cosine_model.fit(tfidf_matrix)

    knn_euclidean_model = create_knn_euclidean_model()
    knn_euclidean_model.fit(tfidf_matrix)
st.success('Data dan model siap!')

# Pilihan judul film dan metrik KNN
title = st.selectbox("Pilih judul film:", options=df['title'].sort_values().unique())
selected_knn_metric = st.selectbox("Pilih Metrik KNN:", options=["Cosine", "Euclidean"])

if title:
    # Mendapatkan rekomendasi dari kedua metode
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
    st.markdown("---") # Garis pemisah
    st.header("Hasil Rekomendasi")
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

    st.markdown("---") # Garis pemisah untuk visualisasi
    st.header("Visualisasi Perbandingan Skor Kemiripan")

    # Kumpulkan data untuk visualisasi Bar Plot
    # Pastikan hanya film yang benar-benar direkomendasikan yang masuk ke combined_data
    combined_data = pd.DataFrame({
        'Film': [rec[0] for rec in cosine_recs] + [rec[0] for rec in knn_recs],
        'Similarity': [rec[1] for rec in cosine_recs] + [rec[1] for rec in knn_recs],
        'Metode': ['Cosine Similarity'] * len(cosine_recs) + [knn_header_text.replace(" Recommendation", "")] * len(knn_recs)
    })

    if not combined_data.empty:
        fig_barplot, ax_barplot = plt.subplots(figsize=(12, 6)) # Ukuran plot lebih besar
        sns.barplot(data=combined_data, x='Similarity', y='Film', hue='Metode', palette='viridis', ax=ax_barplot)
        ax_barplot.set_title('Perbandingan Skor Kemiripan Rekomendasi')
        ax_barplot.set_xlabel('Skor Kemiripan')
        ax_barplot.set_ylabel('Judul Film')
        plt.tight_layout() # Mengatur layout agar label tidak tumpang tindih
        st.pyplot(fig_barplot)
    else:
        st.write("Tidak ada data untuk visualisasi bar plot.")

    st.markdown("---") # Garis pemisah untuk analisis metrik
    st.header("Analisis Metrik Perbandingan")

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

            # --- Visualisasi Overlap ---
            st.write("#### Visualisasi Tingkat Kesamaan Rekomendasi")
            fig_overlap, ax_overlap = plt.subplots(figsize=(6, 3))
            methods_labels = ['Overlap']
            overlap_values = [overlap_percentage]

            ax_overlap.bar(methods_labels, overlap_values, color='skyblue')
            ax_overlap.set_ylim(0, 100)
            ax_overlap.set_ylabel('Persentase (%)')
            ax_overlap.set_title('Persentase Overlap Rekomendasi')
            for i, v in enumerate(overlap_values):
                ax_overlap.text(i, v + 2, f"{v:.2f}%", ha='center', va='bottom', fontsize=10)
            st.pyplot(fig_overlap)
            # --- Akhir Visualisasi Overlap ---

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

        # --- Visualisasi Distribusi Skor (Box Plot) ---
        # Pastikan combined_data tidak kosong dan ada skor yang valid
        if not combined_data.empty and 'Similarity' in combined_data.columns and 'Metode' in combined_data.columns:
            st.write("#### Visualisasi Distribusi Skor Kemiripan")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 5))

            # Pastikan 'Metode' kolom di combined_data sudah benar untuk hue
            sns.boxplot(data=combined_data, x='Similarity', y='Metode', ax=ax_dist, palette='viridis')
            ax_dist.set_title('Distribusi Skor Kemiripan per Metode')
            ax_dist.set_xlabel('Skor Kemiripan')
            ax_dist.set_ylabel('Metode')
            st.pyplot(fig_dist)
        # --- Akhir Visualisasi Distribusi Skor ---

    else:
        st.write("Tidak cukup rekomendasi untuk analisis perbandingan.")

