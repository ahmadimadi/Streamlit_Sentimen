import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Dashboard Sentimen Tokopedia", layout="wide")

# === CSS Custom untuk Tampilan Smooth ===
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .header-container {
        background: linear-gradient(90deg, #0aa66e 0%, #1abc9c 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-title {
        color: white;
        font-size: 30px;
        font-weight: bold;
        margin-left: 10px;
    }
    .footer {
        text-align: center;
        padding: 15px 0;
        color: white;
        background: linear-gradient(to right, #0aa66e, #1abc9c);
        border-radius: 12px;
        font-size: 14px;
        margin-top: 50px;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Menu ===
menu = st.sidebar.radio("📂 Menu Dashboard", [
    "Beranda",
    "Evaluasi Model",
    "Distribusi Sentimen",
    "SMOTE & Komparasi",
    "Word Cloud"
])

# === Header Logo dan Judul ===
col1, col2 = st.columns([1, 5])
with col1:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=100)
    except:
        st.warning("Logo tidak ditemukan.")

with col2:
    st.markdown("""
        <div class="header-container">
            <div class="header-title">📊 Dashboard Analisis Sentimen Tokopedia</div>
        </div>
    """, unsafe_allow_html=True)

# === Load Data ===
data = pd.read_csv("dataset/hasil_stemming.csv")

# === Data Evaluasi ===
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

# === Analisis Sentimen Manual ===
kata_positif = ['bagus', 'cepat', 'murah', 'puas', 'baik', 'mantap', 'top', 'oke',
                'rekomendasi', 'senang', 'rapi', 'tepat', 'suka', 'keren', 'recommended',
                'worth', 'mantapp', 'terpecaya', 'nyaman', 'memuaskan']
kata_negatif = ['lama', 'buruk', 'jelek', 'mengecewa', 'parah', 'tidak', 'kecewa',
                'lelet', 'rusak', 'telat', 'mahal', 'salah', 'kurang', 'cacat',
                'hilang', 'lambat', 'gagal', 'batal', 'ribet', 'bohong', 'zonk', 'macet']
def sentiment_analysis(text):
    text = str(text).lower()
    score = sum(word in text for word in kata_positif) - sum(word in text for word in kata_negatif)
    if score > 0: return 'positif'
    elif score < 0: return 'negatif'
    else: return 'netral'
data['Sentimen'] = data['content_stemming'].apply(sentiment_analysis)

# === Konten Berdasarkan Menu ===

# ---------------------- Beranda ----------------------
if menu == "Beranda":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📖 Judul Penelitian")
    st.markdown("<p style='text-align:justify;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Evaluasi Model ----------------------
elif menu == "Evaluasi Model":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Tabel Evaluasi Model")
    st.dataframe(df_perbandingan.style.format({'Akurasi': '{:.2f}', 'F1 Score': '{:.2f}'}), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Distribusi Sentimen ----------------------
elif menu == "Distribusi Sentimen":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Distribusi Sentimen Ulasan")
    sentimen_counts = data['Sentimen'].value_counts().reset_index()
    sentimen_counts.columns = ['Sentimen', 'Jumlah']
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sentimen_counts, x='Sentimen', y='Jumlah',
                palette=['#2ecc71', '#e74c3c', '#95a5a6'], ax=ax)
    ax.set_title('Distribusi Sentimen', fontsize=14)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- SMOTE & Komparasi ----------------------
elif menu == "SMOTE & Komparasi":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Komparasi Model")
    df_plot = df_perbandingan.melt(id_vars='Model', var_name='Metode', value_name='Skor')
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_plot, x='Model', y='Skor', hue='Metode', palette='Set2', ax=ax)
    plt.xticks(rotation=25)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📉 Perbandingan Sebelum & Sesudah SMOTE")
    sebelum = pd.Series({'Negatif': 243, 'Netral': 219, 'Positif': 97})
    sesudah = pd.Series({'Negatif': 243, 'Netral': 243, 'Positif': 243})
    df_smote = pd.DataFrame({
        'Sentimen': sebelum.index,
        'Sebelum': sebelum.values,
        'Sesudah': sesudah.values
    })
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(df_smote.index, df_smote['Sebelum'], width=0.4, label='Sebelum SMOTE', color='#5DADE2')
    ax2.bar(df_smote.index + 0.4, df_smote['Sesudah'], width=0.4, label='Sesudah SMOTE', color='#58D68D')
    ax2.set_xticks(df_smote.index + 0.2)
    ax2.set_xticklabels(df_smote['Sentimen'])
    ax2.legend()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Word Cloud ----------------------
elif menu == "Word Cloud":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("☁️ Word Cloud Ulasan Pelanggan")
    text = " ".join(data['content_stemming'].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI - Teknik Informatika
    </div>
""", unsafe_allow_html=True)
