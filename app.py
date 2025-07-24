import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils.worker import _barplot, preprocessing
import joblib

# Load model
model_lgb = joblib.load('model/lgbm.pkl')
model_knn = joblib.load('model/knn_lgbm.pkl')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analysis Sentiment Tokopedia", 
    page_icon="img/icon.png", 
    layout="wide"
)

# CSS styling agar tampilan lembut dan modern
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 15px;
            padding: 10px;
            background-color: #e3f2fd;
            color: #0d47a1;
            border-radius: 10px 10px 0 0;
            margin-right: 2px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #bbdefb;
            color: black;
        }
        .stButton>button {
            background-color: #64b5f6;
            color: white;
            border-radius: 8px;
            height: 2.5em;
            width: auto;
            font-size: 14px;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("📦 Analysis Sentiment Tokopedia")
    st.write("This is a simple Streamlit app for sentiment analysis of Tokopedia reviews.")
    st.write("Welcome to the sentiment analysis dashboard!")

    tab1, tab2 = st.tabs(["Dashboard", "Classification"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            model_base_result = pd.DataFrame({
                'model': ['K-NearestNieghbors', 'K-NearestNieghbors', 'Random Forest', 'Random Forest'],
                'splitting_data': ["70% - 30%", "80% - 20%", "70% - 30%", "80% - 20%"],
                'accuracy': [0.70, 0.68, 0.94, 0.94],
                'precision': [0.79, 0.78, 0.94, 0.94],
                'recall': [0.70, 0.68, 0.94, 0.94],
                'f1-score': [0.67, 0.66, 0.94, 0.94]
            })

            st.subheader("🔹 Model Base Results")
            metrics = st.selectbox(
                "Select Metric to Display",
                ["Accuracy", "Precision", "Recall", "F1-Score"],
                key="base_metrics"
            )
            _barplot(model_base_result, metrics.lower())

        with col2:
            model_boosting_result = pd.DataFrame({
                'model': ['XGBoost', 'XGBoost', 'GradBoost', 'GradBoost', 'LightGBM', 'LightGBM'],
                'splitting_data': ["70% - 30%", "80% - 20%", "70% - 30%", "80% - 20%", "70% - 30%", "80% - 20%"],
                'accuracy': [0.96, 0.97, 0.96, 0.97, 0.96, 0.97],
                'precision': [0.96, 0.97, 0.96, 0.97, 0.96, 0.97],
                'recall': [0.96, 0.97, 0.96, 0.97, 0.96, 0.97],
                'f1-score': [0.96, 0.97, 0.96, 0.97, 0.96, 0.97]
            })

            st.subheader("🚀 Model After Boosting")
            metrics2 = st.selectbox(
                "Select Metric to Display",
                ["Accuracy", "Precision", "Recall", "F1-Score"],
                key="boosting_metrics_1"
            )
            _barplot(model_boosting_result, metrics2.lower())

        col_1, col_2 = st.columns(2)
        with col_1:
            model_hybrid_knn = pd.DataFrame({
                'model': ['LGBM + KNN', 'LGBM + KNN'],
                'splitting_data': ["70% - 30%", "80% - 20%"],
                'accuracy': [0.96, 0.97],
                'precision': [0.96, 0.97],
                'recall': [0.96, 0.97],
                'f1-score': [0.96, 0.97]
            })

            st.subheader("🔧 Model After Boosted (LGBM + KNN)")
            metrics3 = st.selectbox(
                "Select Metric to Display",
                ["Accuracy", "Precision", "Recall", "F1-Score"],
                key="boosting_metrics_2"
            )
            _barplot(model_hybrid_knn, metrics3.lower())

        with col_2:
            st.subheader("📉 Confusion Matrix")

            split_option = st.selectbox(
                "Select Splitting Data",
                ("Splitting 70:30", "Splitting 80:20")
            )

            if split_option == "Splitting 70:30":
                cm_manual = np.array([
                    [247, 0, 1],
                    [6, 238, 4],
                    [4, 7, 237]
                ])
                accuracy = 0.9704
            else:
                cm_manual = np.array([
                    [198, 1, 0],
                    [4, 193, 3],
                    [2, 6, 192]
                ])
                accuracy = 0.9680

            labels = ['negatif', 'netral', 'positif']

            fig, ax = plt.subplots(figsize=(2.4, 1.7))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_manual, display_labels=labels)
            disp_plot = disp.plot(cmap='Blues', ax=ax, values_format='d', xticks_rotation=45)

            ax.set_xlabel('Predicted label', fontsize=4)
            ax.set_ylabel('True label', fontsize=4)
            ax.tick_params(axis='both', labelsize=4)

            for t in ax.texts:
                t.set_fontsize(4)

            cbar = disp_plot.figure_.axes[-1]
            cbar.tick_params(labelsize=4)

            ax.text(0.5, 1.08, f"Akurasi: {accuracy:.4f}", transform=ax.transAxes,
                    fontsize=4, ha='center')

            plt.tight_layout()
            st.pyplot(fig)

    with tab2:
        label_map = {1: 'netral', 0: 'negatif', 2: 'positif'}

        st.subheader("🧠 This is Tab Classification")
        texts = st.text_input("Enter text for classification:", placeholder="e.g Tokopedia is a great platform for online shopping.")
        clf = st.button("🚀 Classify Text")

        if clf:
            if not texts:
                st.error("⚠️ Please enter some text to classify.")
            else:
                preprocessed_text = preprocessing(texts)
                result_classification = model_lgb.predict_proba(preprocessed_text)
                result_hybrid = model_knn.predict(result_classification)
                st.success(f"✅ Classification completed successfully! Result is: **{result_hybrid[0]}**")

if __name__ == "__main__":
    main()
