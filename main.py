# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- Konfigurasi ---
st.set_page_config(page_title="Analisis Sentimen Tokopedia", layout="wide")

# --- Header ---
st.title("📊 Dashboard Analisis Sentimen Tokopedia")
st.markdown("""
**Rumusan Masalah**  
Bagaimana kinerja algoritma **Random Forest** dan **K-Nearest Neighbor (KNN)** dalam menganalisis sentimen ulasan Tokopedia?  
Apakah teknik **boosting** dapat meningkatkan performa kedua algoritma tersebut?  
Dan, algoritma mana yang memberikan hasil terbaik setelah diterapkan teknik boosting?

**Tujuan Penelitian**  
Membandingkan kinerja **Random Forest dan KNN**, mengevaluasi efektivitas **teknik boosting (XGBoost, Gradient Boosting, LightGBM, dan Hybrid KNN)**,  
serta mengukur hasil menggunakan **akurasi, presisi, recall, dan F1-score**.
""")

# === Sidebar ===
st.sidebar.header("Navigasi")
opsi = st.sidebar.radio("Pilih Tampilan:", ["📂 Dataset", "📈 Grafik Distribusi", "📉 Komparasi Model", "📊 SMOTE", "📋 WordCloud", "📜 Kesimpulan"])

# === Load data (contoh dummy jika tidak ada file asli) ===
@st.cache_data
def load_data():
    return pd.read_csv("data_tokopedia_sentimen.csv")  # ganti dengan datasetmu

# === Visualisasi berdasarkan pilihan ===
if opsi == "📂 Dataset":
    st.subheader("Data Ulasan Pelanggan Tokopedia")
    df = load_data()
    st.dataframe(df)

elif opsi == "📈 Grafik Distribusi":
    df = load_data()
    st.subheader("Distribusi Sentimen Sebelum & Sesudah SMOTE")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sebelum SMOTE**")
        sns.countplot(data=df, x="Sentimen", palette="viridis")
        st.pyplot(plt.gcf())
        plt.clf()
    with col2:
        st.markdown("**Sesudah SMOTE**")
        smote_df = pd.read_csv("data_smote.csv")  # ganti dengan hasil balancing SMOTE
        sns.countplot(data=smote_df, x="Sentimen", palette="rocket")
        st.pyplot(plt.gcf())
        plt.clf()

elif opsi == "📉 Komparasi Model":
    st.subheader("Perbandingan Kinerja Model")
    model_data = pd.read_csv("evaluasi_model.csv")  # kolom: Model, Akurasi, Presisi, Recall, F1
    st.dataframe(model_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    model_data.plot(x="Model", y=["Akurasi", "Presisi", "Recall", "F1"], kind="bar", ax=ax)
    plt.title("Komparasi Performa Model")
    plt.ylabel("Nilai (%)")
    plt.ylim(0, 1)
    st.pyplot(fig)

elif opsi == "📊 SMOTE":
    st.subheader("Distribusi Sentimen Pasca SMOTE")
    df = pd.read_csv("data_smote.csv")
    st.dataframe(df['Sentimen'].value_counts())

elif opsi == "📋 WordCloud":
    from wordcloud import WordCloud
    df = load_data()
    text = " ".join(df["content_stemming"].dropna().astype(str))
    wc = WordCloud(width=1000, height=400, background_color='white').generate(text)
    st.image(wc.to_array(), use_column_width=True)

elif opsi == "📜 Kesimpulan":
    st.subheader("Kesimpulan")
    st.markdown("""
    ✅ **Random Forest** menunjukkan performa stabil sebelum dan sesudah boosting  
    ✅ **KNN** meningkat signifikan setelah diterapkan **Hybrid Boosting**  
    ✅ Teknik **XGBoost** dan **LGBM** mampu meningkatkan performa hingga 10% pada beberapa metrik  
    ✅ Model terbaik berdasarkan F1-score adalah **XGBoost Random Forest**  
    """)

