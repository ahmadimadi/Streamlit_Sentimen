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
            <div class="header-title">📊 Streamlit (Dashboard Interaktif)</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<p style='text-align:right;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)
st.markdown("---")

# === Layout 3 Kolom ===
col1, col2, col3 = st.columns([1.3, 2, 1.7])

# === Kolom 1 ===
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Tabel Evaluasi Model")
    st.markdown("<!-- Tabel akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("📊 Distribusi Ulasan")
    st.markdown("<!-- Grafik distribusi akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Kolom 2 ===
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Grafik Komparasi Model")
    st.markdown("<!-- Grafik komparasi akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("📉 Perbandingan Sebelum & Sesudah SMOTE")
    st.markdown("<!-- Grafik SMOTE akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Kolom 3 ===
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("☁️ Word Cloud Ulasan Pelanggan")
    st.markdown("<!-- Wordcloud akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='margin-top: 20px;'>", unsafe_allow_html=True)
    st.subheader("🧾 Word Cloud dari Proses Stemming")
    st.markdown("<!-- Wordcloud stemming akan ditambahkan di sini -->", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
    <div class="footer">
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
