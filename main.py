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

# === Menu Navigasi di Bawah Judul ===
menu = st.selectbox(
    "📁 Pilih Menu",
    options=[
        "Beranda",
        "Evaluasi Model",
        "Distribusi Sentimen",
        "Wordcloud Ulasan",
        "Wordcloud Stemming",
        "Tentang"
    ]
)

st.markdown("---")

# === Konten berdasarkan menu ===
if menu == "Beranda":
    st.subheader("📌 Beranda")
    st.markdown("Halaman pembuka dashboard. Tambahkan konten sesuai kebutuhan Anda.")

elif menu == "Evaluasi Model":
    st.subheader("📋 Tabel Evaluasi Model")
    st.markdown("Tabel dan grafik evaluasi model akan ditambahkan di sini.")

elif menu == "Distribusi Sentimen":
    st.subheader("📊 Distribusi Sentimen")
    st.markdown("Grafik distribusi sentimen akan ditambahkan di sini.")

elif menu == "Wordcloud Ulasan":
    st.subheader("☁️ Wordcloud Ulasan Pelanggan")
    st.markdown("Visualisasi wordcloud dari ulasan pelanggan akan ditambahkan di sini.")

elif menu == "Wordcloud Stemming":
    st.subheader("🧾 Wordcloud dari Proses Stemming")
    st.markdown("Visualisasi wordcloud dari hasil stemming akan ditambahkan di sini.")

elif menu == "Tentang":
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown("Informasi tentang peneliti, data, dan tujuan proyek dapat ditambahkan di sini.")

# === Footer ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
