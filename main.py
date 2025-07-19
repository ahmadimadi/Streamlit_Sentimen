import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from PIL import Image

# === Page Config ===
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# === Custom CSS for Modern Look ===
st.markdown("""
    <style>
    body {
        background-color: #f9fafc;
    }
    .main > div {
        padding-top: 1rem;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Menu ===
with st.sidebar:
    selected = option_menu("Menu", ["Beranda", "Distribusi Sentimen", "Evaluasi Model", "Wordcloud", "Tentang"],
                           icons=["house", "bar-chart", "cpu", "cloud", "info-circle"],
                           menu_icon="cast", default_index=0, styles={
                               "container": {"padding": "5px", "background-color": "#ffffff"},
                               "icon": {"color": "#1f77b4", "font-size": "20px"},
                               "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
                               "nav-link-selected": {"background-color": "#1f77b4", "color": "white"},
                           })

# === Content Based on Menu ===
if selected == "Beranda":
    st.title("📊 Dashboard Analisis Sentimen")
    st.subheader("Platform: Tokopedia • Algoritma: Random Forest & K-NN dengan Boosting")
    st.markdown("Visualisasi dan evaluasi dari hasil analisis sentimen menggunakan algoritma klasifikasi dengan peningkatan performa melalui teknik boosting.")
    
    st.image("https://img.freepik.com/free-vector/data-analysis-illustration_23-2148809086.jpg", width=700)

elif selected == "Distribusi Sentimen":
    st.header("🔍 Distribusi Sentimen")
    
    # Contoh data dummy (ganti dengan hasil model Anda)
    data = pd.DataFrame({
        'Sentimen': ['Positif', 'Negatif', 'Netral'],
        'Jumlah': [420, 210, 170]
    })
    
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='Sentimen', y='Jumlah', palette='pastel', ax=ax)
    ax.set_title("Distribusi Kelas Sentimen")
    st.pyplot(fig)

elif selected == "Evaluasi Model":
    st.header("📈 Evaluasi Model")
    st.markdown("Perbandingan performa algoritma sebelum dan sesudah diterapkan teknik boosting.")

    # Contoh data metrik evaluasi (sesuaikan dengan hasil Anda)
    eval_df = pd.DataFrame({
        'Model': ['RF', 'KNN', 'RF + Boosting', 'KNN + Boosting'],
        'Accuracy': [0.85, 0.81, 0.89, 0.86],
        'F1-Score': [0.84, 0.80, 0.88, 0.85]
    })
    st.dataframe(eval_df)

    st.bar_chart(eval_df.set_index("Model"))

elif selected == "Wordcloud":
    st.header("☁️ Wordcloud dari Ulasan")
    col1, col2 = st.columns(2)
    with col1:
        st.image("assets/wordcloud_positif.png", caption="Wordcloud Sentimen Positif")
    with col2:
        st.image("assets/wordcloud_negatif.png", caption="Wordcloud Sentimen Negatif")

elif selected == "Tentang":
    st.header("📌 Tentang Penelitian")
    st.markdown("""
    **Judul:** Komparasi Kinerja Algoritma Random Forest dan K-Nearest Neighbor  
    **Topik:** Analisis Sentimen Ulasan Pelanggan Tokopedia  
    **Fokus:** Evaluasi performa dan pengaruh teknik Boosting seperti XGBoost, LightGBM, dan Gradient Boosting.  
    """)

# === Footer ===
st.markdown("""
    <hr style='border:1px solid #ccc'>
    <center>© 2025 Ahmadi • Dashboard Analisis Sentimen</center>
""", unsafe_allow_html=True)
