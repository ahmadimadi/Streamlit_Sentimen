# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
from IPython.display import display, HTML

# === Konfigurasi halaman ===
st.set_page_config(page_title="Dashboard Sentimen Tokopedia", layout="wide")
data = pd.read_csv ('dataset/hasil_stemming.csv')
# === Gaya CSS Custom ===
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(to right, #0aa66e, #1abc9c);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header-title {
        color: white;
        font-size: 28px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        padding: 10px;
        margin-top: 40px;
        color: white;
        background: linear-gradient(to right, #0aa66e, #1abc9c);
        border-radius: 8px;
        font-size: 14px;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# === Header ===
col_logo, col_title = st.columns([1, 5])
with col_logo:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=110)
    except FileNotFoundError:
        st.warning("Logo tidak ditemukan di folder 'assets/logo.png'")

with col_title:
    st.markdown("""
        <div class="header-container">
            <div class="header-title">📊 Streamlit (Dashboard Interaktif) </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align:right;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)
st.markdown("---")

# === Data Evaluasi Model ===
df_perbandingan = pd.DataFrame({
    'Model': [
        'Random Forest (80:20)', 'KNN (80:20)',
        'Random Forest (70:30)', 'KNN (70:30)',
        'XGBoost (Boosted-RF)', 'Gradient Boosting (Boosted-RF)', 'LightGBM (Boosted-RF)',
        'Hybrid Boosted-KNN (LGBM+KNN)'
    ],
    'Akurasi': [0.6644, 0.4247, 0.7123, 0.4658, 0.6644, 0.8493, 0.7397, 0.6096],
    'F1 Score': [0.3143, 0.2532, 0.2966, 0.2846, 0.6458, 0.8513, 0.7440, 0.6004]
})

# === Layout 3 Kolom ===
col1, col2, col3 = st.columns([1.3, 2, 1.7])

# === Kolom 1: Tabel Evaluasi & Distribusi Sentimen ===
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Tabel Evaluasi Model")
    st.dataframe(df_perbandingan.style.format({'Akurasi': '{:.2f}', 'F1 Score': '{:.2f}'}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # === Grafik Distribusi Sentimen ===
    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("📊 Distribusi Ulasan")

    # daftar kata positif dan negatif setelah stemming
    kata_positif = [
        'bagus', 'cepat', 'murah', 'puas', 'baik', 'mantap', 'top', 'oke',
        'rekomendasi', 'senang', 'rapi', 'tepat', 'suka', 'keren', 'recommended',
        'worth', 'bagus', 'mantapp','terpecaya','nyaman','memuaskan','murah'
    ]
    
    kata_negatif = [
        'lama', 'buruk', 'jelek', 'mengecewa', 'parah', 'tidak', 'kecewa',
        'lelet', 'rusak', 'telat', 'mahal', 'salah', 'kurang', 'jelekkk', 'cacat',
        'hilang', 'lambat', 'gagal', 'batal','ribet','bohong','zonk','macet'
    ]
    
    # --- Fungsi Analisis Sentimen ---
    def sentiment_analysis_indonesian(text):
        text = text.lower()
        score = 0
        for word in kata_positif:
            if word in text:
                score += 1
        for word in kata_negatif:
            if word in text:
                score -= 1
    
        if score > 0:
            return (score, 'positif')
        elif score < 0:
            return (score, 'negatif')
        else:
            return (score, 'netral')
    
    # --- Terapkan Fungsi ke Kolom 'content_stemming' ---
    results = data['hasil_stemming'].astype(str).apply(sentiment_analysis_indonesian)
    results = list(zip(*results))
    
    # --- Tambahkan Kolom 'score' dan 'Sentimen' ke DataFrame ---
    data['score'] = results[0]
    data['Sentimen'] = results[1]

    # --- Buat Tabel Rekapitulasi Sentimen ---
    sentimen_counts = data['Sentimen'].value_counts().reset_index()
    sentimen_counts.columns = ['Sentimen', 'Jumlah']

    fig_sent, ax_sent = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    bars = ax_sent.bar(
        sentimen_counts['Sentimen'],
        sentimen_counts['Jumlah'],
        color=colors,
        edgecolor='none',
        width=0.6,
        zorder=3
    )

    for bar in bars:
        height = bar.get_height()
        ax_sent.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height),
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

    ax_sent.set_title('Distribusi Sentimen Ulasan', fontsize=14, fontweight='bold', color='#34495e', pad=20)
    ax_sent.set_ylabel('Jumlah', fontsize=11, color='#34495e')
    ax_sent.set_xticks(range(len(sentimen_counts)))
    ax_sent.set_xticklabels(sentimen_counts['Sentimen'], fontsize=11, color='#34495e')
    ax_sent.tick_params(axis='y', labelsize=10, colors='#7f8c8d')
    ax_sent.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax_sent.spines['top'].set_visible(False)
    ax_sent.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_sent)
    st.markdown("</div>", unsafe_allow_html=True)

# === Kolom 2: Grafik Komparasi + SMOTE ===
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Grafik Komparasi Model")
    df_plot = df_perbandingan.melt(id_vars='Model', var_name='Metode Evaluasi', value_name='Skor')
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, x='Model', y='Skor', hue='Metode Evaluasi', palette='Set2', ax=ax)
    ax.set_title("Akurasi & F1 Score - RF & KNN (Boosted & Non-Boosted)", fontsize=13, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # === Grafik Perbandingan Sebelum & Sesudah SMOTE ===
    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("📉 Perbandingan Sebelum & Sesudah SMOTE")

    sebelum_smote = pd.Series({'Negatif': 243, 'Netral': 219, 'Positif': 97})
    sesudah_smote = pd.Series({'Negatif': 243, 'Netral': 243, 'Positif': 243})

    df_smote = pd.DataFrame({
        'Sentimen': sebelum_smote.index,
        'Sebelum SMOTE': sebelum_smote.values,
        'Sesudah SMOTE': sesudah_smote.values
    })

    sns.set(style="whitegrid")
    fig_smote, ax_smote = plt.subplots(figsize=(6.5, 4))
    bar_width = 0.35
    x = range(len(df_smote))
    warna = ['#5DADE2', '#58D68D']

    bars1 = ax_smote.bar(x, df_smote['Sebelum SMOTE'], width=bar_width, label='Sebelum SMOTE', color=warna[0])
    bars2 = ax_smote.bar([i + bar_width for i in x], df_smote['Sesudah SMOTE'], width=bar_width, label='Sesudah SMOTE', color=warna[1])

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax_smote.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 4), textcoords="offset points", ha='center', fontsize=9)
    ax_smote.set_xlabel('Kategori Sentimen', fontsize=11)
    ax_smote.set_ylabel('Jumlah Data', fontsize=11)
    ax_smote.set_xticks([i + bar_width / 2 for i in x])
    ax_smote.set_xticklabels(df_smote['Sentimen'], fontsize=10)
    ax_smote.legend(title='Keterangan', fontsize=9)
    sns.despine()

    plt.tight_layout()
    st.pyplot(fig_smote)
    st.markdown("</div>", unsafe_allow_html=True)

# === Kolom 3: Word Cloud + Stemming ===
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("☁️ Word Cloud Ulasan Pelanggan")
    teks_ulasan = """
    pengiriman cepat produk sesuai kualitas bagus sangat puas terima kasih mantap harga murah oke respon cepat
    kemasan aman barang original cocok akan beli lagi sangat rekomendasi terpercaya packing rapi top banget
    """
    wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='tab10').generate(teks_ulasan)
    fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)
    st.markdown("</div>", unsafe_allow_html=True)

    # === Word Cloud dari hasil Stemming ===
    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("🧾 Word Cloud dari Proses Stemming")

    # Contoh data stemming
    data = pd.DataFrame({
        'content_stemming': [
            'produk bagus', 'pengiriman cepat', 'harga murah',
            'produk asli', 'packing rapi', 'terima kasih produk oke',
            'produk bagus dan cepat sampai', 'harga terjangkau kualitas oke',
        ]
    })

    wordcloud2 = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', max_words=200)\
                .generate(" ".join(data['content_stemming'].dropna().astype(str)))

    fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3))
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc2)

    # Tabel keterangan
    st.markdown("#### 📌 Keterangan")
    tabel = pd.DataFrame({
        'Keterangan': ['Preprocessing terakhir', 'Tujuan visualisasi'],
        'Isi': ['Stemming dengan Sastrawi', 'Menampilkan kata paling sering muncul']
    })
    st.table(tabel)
    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
