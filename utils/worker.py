from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import re
import pandas as pd
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import numpy as np
import joblib 

factory = StemmerFactory()
stemmer = factory.create_stemmer()
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Buat fungsi untuk langkah stemming bahasa Indonesia
def Stemming(text):
    text = stemmer.stem(text)
    return text

stop_words = set(stopwords.words('english'))

def Filtering(text):
    clean_words = []
    for word in text:
        if word not in stop_words:
            clean_words.append(word)
    return " ".join(clean_words)

def Tokenizing(text):
    
    return text.split()
# 1. Baca file key_norm.csv
key_norm = pd.read_csv('dataset/key_norm.csv')

# 2. Buat dictionary untuk normalisasi
kamus_norm = dict(zip(key_norm['singkat'], key_norm['hasil']))

# 3. Fungsi normalisasi kata
def WordNormalization(text):
    text = ' '.join([kamus_norm.get(word, word) for word in text.split()])
    return text.lower()

def _barplot(data, metrics) -> None:
    """
    Create a bar plot for the given data and metrics.

    Parameters:
    - data: DataFrame containing the model results.
    - metrics: The metric to be plotted.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax = sns.barplot(x=data['splitting_data'], y=data[metrics], hue=data['model'], palette='viridis')
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10, padding=3)

    plt.title(f'Model Performance by {metrics}')
    plt.xlabel('Splitting Data')
    plt.ylim(0, 1.4)
    plt.ylabel(metrics)
    plt.legend(title='Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
def datacleaning(text):
    text = re.sub(r'(\[.*?\]|\(.*?\))','',text) #menghilangkan kata-kata dalam kurung
    text = re.sub(r'\+\d{2} \d{3}-\d{4}-\d{4}','',text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.','',text) #menghilangkan format tanggal dan waktu
    text = re.sub(r'\d{2}\s\w{3,}\s\d{4}.','', text) #Menghilangkan format tanggal "XX NAMA BULAN TAHUN"
    text = re.sub(r'(menit|mnt|thn|tahun|minggu|mg|hari|hr|jam|jm|detik|dtk|sekon)*','', text) #Menghilangkan satuan waktu
    text = re.sub(r'(\d{1,}\s*gb|\d{1,}\s*kb|\d{1,}\s*mb|\d{1,}\s*tb|lte)',"", text) #Menghilangkan satuan byte dan kata lte
    text = re.sub(r'(ribu|rb|jt|juta|milyar|miliar|triliun|trilyun)',"", text) # Menghilangkan satuan uang
    text = re.sub(r'\w*\.*\w{1,}\.*\/\w{1,}','',text) #Menghilangkan pecahan
    text = re.sub(r'rp\s*\d{1,}\s','',text) # Menghilangkan jumlah tarif
    text = re.sub(r"\*\d{3,}\*\d{3,}\#","", text) # Menghilangkan kode aktivasi-1
    text = re.sub(r"\*\d{3,}\#","", text) #Menghilangkan kode aktivasi-2
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # menghapus mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # menghapus hashtag
    text = re.sub(r'RT[\s]', '', text) # menghapus retweet
    text = re.sub(r'[?|$|.|@#%^/&*=!_:")(-+,]', '', text) # menghapus simbol
    text = re.sub(r"http\S+", '', text) # menghapus link
    text = re.sub(r'[0-9]+', '', text) # menghapus angka
    emoticon_pattern = re.compile("["
                                  u"\U0001F600-\U0001F64F"  # emotikon wajah umum
                                  u"\U0001F300-\U0001F5FF"  # emotikon kategori alat
                                  u"\U0001F680-\U0001F6FF"  # emotikon transportasi dan simbol
                                  u"\U0001F1E0-\U0001F1FF"  # emotikon bendera negara
                                  u"\U00002700-\U000027BF"  # emotikon simbol matematika
                                  u"\U000024C2-\U0001F251"
                                  u"\U0001F932"
                                  u"\U0001F92D"
                                  "]+", flags=re.UNICODE)
    # Menghapus emotikon dari teks
    text = emoticon_pattern.sub(r'', text)
    text = text.replace('\n', ' ') # mengganti baris baru menjadi spasi
    text = text.strip(' ') # hapus spasi dari kiri dan kanan teks
    return text

# Ambil vektor tiap kata, lalu hitung rata-ratanya
def sentence_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def preprocessing(texts: str):
    # bersihin & normalisasi
    text = datacleaning(texts)
    text = WordNormalization(text)

    # tokenisasi & filter stopword
    tokens = Tokenizing(text)
    filtered = Filtering(tokens)

    # stemming pada seluruh kalimat
    stemmed = Stemming(filtered)

    # transform ke TF-IDF
    tfidf_vector = vectorizer.transform([stemmed])
    return tfidf_vector
