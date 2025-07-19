import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Dashboard Sentimen Tokopedia", layout="wide")

# === Header dengan Logo ===
col1, col2 = st.columns([1, 5])
with col1:
    try:
        logo = Image.open("assets/logo.png")
        st.image(logo, width=110)
    except:
        st.markdown("🖼️ Logo")

with col2:
    st.markdown("""
        <div style='background: linear-gradient(to right, #0aa66e, #1abc9c); padding: 20px 30px; border-radius: 10px;'>
            <h2 style='color: white;'>📊 Streamlit (Dashboard Interaktif)</h2>
        </div>
    """, unsafe_allow_html=True)

# === Sub Judul Penelitian ===
st.markdown("<p style='text-align:right;'>KOMPARASI KINERJA ALGORITMA RANDOM FOREST DAN KNN DALAM ANALISIS SENTIMEN ULASAN PELANGGAN DI PLATFORM E-COMMERCE TOKOPEDIA DENGAN PENERAPAN TEKNIK BOOSTING</p>", unsafe_allow_html=True)

# === MENU HORIZONTAL (seperti Bootstrap) ===
selected = option_menu(
    menu_title=None,  # Hide the title
    options=["Dashboard", "Klasifikasi", "Training"],
    icons=["bar-chart-line", "search", "cpu"],  # Bootstrap icons
    menu_icon="cast",  # Icon kiri
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#1abc9c", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#1abc9c", "color": "white"},
    }
)

st.markdown("---")

# === DASHBOARD ===
if selected == "Dashboard":
    st.subheader("📋 Evaluasi Model")
    df_perbandingan = pd.DataFrame({
        'Model': [
            'Random Forest', 'KNN', 'XGBoost', 'LightGBM'
        ],
        'Akurasi': [0.72, 0.61, 0.84, 0.74],
        'F1 Score': [0.69, 0.60, 0.85, 0.74]
    })
    st.dataframe(df_perbandingan.style.format({'Akurasi': '{:.2f}', 'F1 Score': '{:.2f}'}), use_container_width=True)

    st.subheader("📊 Grafik Distribusi Sentimen")
    df_sent = pd.DataFrame({'Sentimen': ['Positif', 'Netral', 'Negatif'], 'Jumlah': [97, 219, 243]})
    fig, ax = plt.subplots()
    ax.bar(df_sent['Sentimen'], df_sent['Jumlah'], color=['green', 'gray', 'red'])
    st.pyplot(fig)

    st.subheader("☁️ Word Cloud")
    text = "mantap cepat murah bagus rekomendasi terpercaya puas mantap oke murah bagus aman"
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig_wc, ax_wc = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig_wc)

# === KLASIFIKASI ===
elif selected == "Klasifikasi":
    st.subheader("🔍 Klasifikasi Ulasan")
    ulasan = st.text_area("Masukkan teks ulasan pelanggan:")
    if st.button("Klasifikasikan"):
        st.success("Sentimen: Positif")  # Placeholder hasil

# === TRAINING ===
elif selected == "Training":
    st.subheader("⚙️ Training Model")
    st.markdown("Upload data, latih model, dan evaluasi di sini.")

# === FOOTER ===
st.markdown("""
    <div style='text-align:center; padding:10px; margin-top:40px; color:white;
                background: linear-gradient(to right, #0aa66e, #1abc9c);
                border-radius:8px; font-size:14px;'>
        &copy; 2025 | Dibuat oleh <b>Ahmadi</b> | USTI Teknik Informatika
    </div>
""", unsafe_allow_html=True)
