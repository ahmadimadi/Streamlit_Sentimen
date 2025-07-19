import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Dashboard Sentimen Tokopedia", layout="wide")

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
        margin-bottom: 10px;
    }
    .header-title {
        color: white;
        font-size: 28px;
        font-weight: bold;
    }
    .menu-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .menu-button {
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        color: #333;
        padding: 10px 25px;
        margin: 0 10px;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        text-align: center;
    }
    .menu-button:hover {
        background-color: #1abc9c;
        color: white;
        border: 1px solid #1abc9c;
    }
    .active {
        background-color: #1abc9c !important;
        color: white !important;
        border: 1px solid #1abc9c !important;
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

# === Subjudul Penelitian ===
st.markdown("<p style='text-align:right;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)

# === MENU TETAP ===
menu_option = st.radio(
    "Pilih Halaman:",
    ["Dashboard", "Klasifikasi", "Training"],
    horizontal=True,
    index=0,
)

st.markdown("---")

# === KONTEN BERDASARKAN MENU ===
if menu_option == "Dashboard":
    st.subheader("📊 Halaman Dashboard")
    st.markdown("Silakan tambahkan grafik, tabel distribusi, atau metrik evaluasi di sini.")

elif menu_option == "Klasifikasi":
    st.subheader("📌 Halaman Klasifikasi")
    st.markdown("Di sini Anda bisa menambahkan form input teks ulasan dan prediksi sentimen.")

elif menu_option == "Training":
    st.subheader("⚙️ Halaman Training Model")
    st.markdown("Di sini Anda bisa tambahkan fitur pelatihan model, upload data, atau evaluasi ulang.")

# === FOOTER ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
