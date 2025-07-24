import streamlit as st
import pandas as pd
import joblib
from utils.worker import _barplot, preprocessing
import numpy as np

# Load model
model_lgb = joblib.load('models/lgb.pkl')
model_knn = joblib.load('models/knn_lgbm.pkl')

# Label map konsisten
label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}

# Konfigurasi halaman
st.set_page_config(
    page_title="Analysis Sentiment Tokopedia",
    page_icon="img/icon.png",
    layout="wide"
)

def main():
    st.title("Analysis Sentiment Tokopedia")
    st.write("Aplikasi analisis sentimen untuk ulasan pelanggan di Tokopedia.")
    st.write("Selamat datang di dashboard analisis sentimen!")

    tab1, tab2 = st.tabs(["📊 Dashboard", "🧠 Classification"])

    # === TAB 1 - DASHBOARD ===
    with tab1:
        col1, col2 = st.columns(2)

        # Model Base
        with col1:
            model_base_result = pd.DataFrame({
                'model': ['K-NearestNeighbors', 'K-NearestNeighbors', 'Random Forest', 'Random Forest'],
                'splitting_data': ["70% - 30%", "80% - 20%", "70% - 30%", "80% - 20%"],
                'accuracy': [0.70, 0.68, 0.94, 0.94],
                'precision': [0.79, 0.78, 0.94, 0.94],
                'recall': [0.70, 0.68, 0.94, 0.94],
                'f1-score': [0.67, 0.66, 0.94, 0.94]
            })
            st.subheader("🎯 Model Base Results")
            metrics = st.selectbox("Pilih metrik:", ["Accuracy", "Precision", "Recall", "F1-Score"], key="base_metrics")
            _barplot(model_base_result, metrics.lower())

        # Model Boosting
        with col2:
            model_boosting_result = pd.DataFrame({
                'model': ['K-NearestNeighbors', 'K-NearestNeighbors', 'Random Forest', 'Random Forest'],
                'splitting_data': ["70% - 30%", "80% - 20%", "70% - 30%", "80% - 20%"],
                'accuracy': [0.97, 0.97, 0.98, 0.98],
                'precision': [0.78, 0.78, 0.94, 0.94],
                'recall': [0.71, 0.70, 0.94, 0.94],
                'f1-score': [0.69, 0.68, 0.94, 0.94]
            })
            st.subheader("⚡ Model After Boosting")
            metrics2 = st.selectbox("Pilih metrik:", ["Accuracy", "Precision", "Recall", "F1-Score"], key="boosting_metrics")
            _barplot(model_boosting_result, metrics2.lower())

    # === TAB 2 - KLASIFIKASI ===
    with tab2:
        st.subheader("🔍 Klasifikasi Kalimat Tunggal")

        teks = st.text_input("Masukkan satu kalimat ulasan:", placeholder="Contoh: Barang cepat sampai dan sesuai deskripsi.")

        if st.button("🔎 Klasifikasi Kalimat"):
            if not teks:
                st.warning("Masukkan kalimat terlebih dahulu.")
            else:
                try:
                    teks_prepro = preprocessing(teks)
                    probs = model_lgb.predict_proba([teks_prepro])
                    pred = model_knn.predict(probs)
                    label = label_map.get(pred[0], "Tidak diketahui")
                    st.success(f"Hasil klasifikasi: **{label}**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

        st.markdown("---")
        st.subheader("📂 Klasifikasi Dataset CSV")

        uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'score':", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if 'score' not in df.columns:
                    st.error("Kolom 'score' tidak ditemukan di file CSV.")
                else:
                    with st.spinner("Melakukan klasifikasi..."):
                        df['processed'] = df['score'].astype(str).apply(preprocessing)
                        lgb_probs = model_lgb.predict_proba(df['processed'].tolist())
                        predictions = model_knn.predict(lgb_probs)
                        df['prediksi_sentimen'] = [label_map[p] for p in predictions]

                        st.success("Klasifikasi selesai.")
                        st.write(df[['score', 'prediksi_sentimen']])

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil", data=csv, file_name="hasil_klasifikasi.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Kesalahan saat memproses file: {e}")

if __name__ == "__main__":
    main()
