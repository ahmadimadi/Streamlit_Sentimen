import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Dashboard Sentimen Tokopedia", layout="wide")

# === CSS Styling ===
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(to right, #0aa66e, #1abc9c);
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 10px;
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
            <div class="header-title">📊 Streamlit (Dashboard Interaktif)</div>
        </div>
    """, unsafe_allow_html=True)
    
# === Sub Judul Penelitian ===
st.markdown("<p style='text-align:right;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)

# === MENU PILIHAN ===
menu_option = st.radio("📁 Pilih Menu", ["Dashboard", "Klasifikasi", "Training"], horizontal=True)
st.markdown("---")

# === DASHBOARD (Isi Lengkap Tetap) ===
if menu_option == "Dashboard":
    # Data Dummy Evaluasi
    df_perbandingan = pd.DataFrame({
        'Model': [
            'Random Forest (80:20)', 'KNN (80:20)',
            'Random Forest (70:30)', 'KNN (70:30)',
            'XGBoost', 'Gradient Boosting', 'LightGBM',
            'Hybrid Boosted-KNN'
        ],
        'Akurasi': [0.6644, 0.4247, 0.7123, 0.4658, 0.6644, 0.8493, 0.7397, 0.6096],
        'F1 Score': [0.3143, 0.2532, 0.2966, 0.2846, 0.6458, 0.8513, 0.7440, 0.6004]
    })

    # Layout 3 Kolom
    col1, col2, col3 = st.columns([1.3, 2, 1.7])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📋 Tabel Evaluasi Model")
        st.dataframe(df_perbandingan.style.format({'Akurasi': '{:.2f}', 'F1 Score': '{:.2f}'}), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
        st.subheader("📊 Distribusi Ulasan")
        data = pd.DataFrame({'Sentimen': ['Positif']*97 + ['Netral']*219 + ['Negatif']*243})
        sentimen_counts = data['Sentimen'].value_counts().reset_index()
        sentimen_counts.columns = ['Sentimen', 'Jumlah']

        fig_sent, ax = plt.subplots(figsize=(6, 4))
        warna = ['#2ecc71', '#95a5a6', '#e74c3c']
        ax.bar(sentimen_counts['Sentimen'], sentimen_counts['Jumlah'], color=warna)
        st.pyplot(fig_sent)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Grafik Komparasi Model")
        df_plot = df_perbandingan.melt(id_vars='Model', var_name='Metode', value_name='Skor')
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=df_plot, x='Model', y='Skor', hue='Metode', palette='Set2', ax=ax)
        ax.set_title("Akurasi & F1 Score", fontsize=13)
        plt.xticks(rotation=25)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
        st.subheader("📉 Perbandingan SMOTE")
        df_smote = pd.DataFrame({
            'Sentimen': ['Positif', 'Netral', 'Negatif'],
            'Sebelum': [97, 219, 243],
            'Sesudah': [243, 243, 243]
        })
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        bar_width = 0.35
        x = range(len(df_smote))
        ax2.bar(x, df_smote['Sebelum'], width=bar_width, label='Sebelum SMOTE', color='#5DADE2')
        ax2.bar([i + bar_width for i in x], df_smote['Sesudah'], width=bar_width, label='Sesudah SMOTE', color='#58D68D')
        ax2.set_xticks([i + bar_width/2 for i in x])
        ax2.set_xticklabels(df_smote['Sentimen'])
        ax2.legend()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("☁️ Word Cloud Ulasan Pelanggan")
        teks = "bagus murah cepat mantap oke terpercaya puas puas harga oke murah oke bagus aman"
        wc = WordCloud(width=600, height=300, background_color='white').generate(teks)
        fig_wc, ax_wc = plt.subplots()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig_wc)
        st.markdown("</div>", unsafe_allow_html=True)

# === MENU KLASIFIKASI ===
elif menu_option == "Klasifikasi":
    st.subheader("📌 Halaman Klasifikasi")
    st.markdown("Form klasifikasi ulasan pelanggan akan ditambahkan di sini.")

# === MENU TRAINING ===
elif menu_option == "Training":
    st.subheader("⚙️ Halaman Training Model")
    st.markdown("Fitur training model atau upload dataset akan ditambahkan di sini.")

# === FOOTER ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
