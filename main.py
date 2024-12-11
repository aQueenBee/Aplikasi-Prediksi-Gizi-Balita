import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import io

# Judul Aplikasi
st.title("Aplikasi Prediksi Status Gizi Balita")
st.info("Streamlit adalah framework berbasis Python untuk membuat aplikasi web interaktif dengan fokus pada data science dan machine learning.")

# Deskripsi Aplikasi
st.markdown("""
Aplikasi ini dirancang untuk membantu tenaga kesehatan atau peneliti di bidang gizi anak dalam:
""")

# Tujuan Aplikasi
st.subheader("Tujuan Aplikasi")
st.markdown("""
1. **Memprediksi status gizi balita**  
   Berdasarkan data antropometri seperti berat badan, tinggi badan, lingkar lengan atas (LiLA), dan usia.
2. **Melakukan prediksi status gizi balita**  
   Kategori status gizi meliputi:
   - Gizi buruk (severely wasted)
   - Gizi kurang (wasted)
   - Gizi baik (normal)
   - Berisiko gizi lebih (possible risk of overweight)
   - Gizi lebih (overweight)
   - Obesitas (obese)
3. **Mengatasi ketidakseimbangan kelas gizi menggunakan teknik:**  
   - Under-sampling
   - SMOTE 
""")

# Sidebar untuk Pemilihan Algoritma
st.sidebar.header('Algoritma Klasifikasi')
algorithm = st.sidebar.selectbox(
    'Pilih Algoritma',
    ['K-Nearest Neighbor', 'Regresi Linear', 'Naive Bayes']
)

# Penjelasan algoritma
st.subheader('Penjelasan Algoritma')
if algorithm == 'K-Nearest Neighbor':
    st.write('K-Nearest Neighbors (KNN) adalah algoritma yang digunakan untuk klasifikasi atau regresi. KNN bekerja dengan mencari data terdekat dan memberikan label berdasarkan mayoritas label dari tetangga terdekat.')
elif algorithm == 'Regresi Linear':
    st.write('Regresi linear adalah metode statistik yang digunakan untuk memodelkan hubungan antara variabel independen dan dependen.')
else:
    st.write('Naive Bayes adalah algoritma klasifikasi berbasis probabilitas yang berdasarkan pada Teorema Bayes dengan asumsi independensi antara fitur-fitur.')

# Fungsi untuk memuat model KNN
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Fungsi untuk memuat scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Upload Data
st.subheader("Unggah Dataset")
uploaded_file = st.file_uploader("Pilih file CSV untuk dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset yang diunggah:", data.head())

    # Menampilkan jumlah baris dan kolom
    st.write(f"Jumlah Baris: {data.shape[0]}")
    st.write(f"Jumlah Kolom: {data.shape[1]}")

    # Memuat model KNN dan scaler
    try:
        model = load_model("knn_model.pkl")
        scaler = load_scaler("scaler.pkl")
        st.success("Model KNN dan Scaler berhasil dimuat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")

    # Form input data balita
    st.subheader("Input Data Balita")
    usia = st.number_input("Usia (dalam bulan)", min_value=0, step=1)
    berat_badan = st.number_input("Berat Badan (dalam kg)", min_value=0.1, step=0.1)
    tinggi_badan = st.number_input("Tinggi Badan (dalam cm)", min_value=1, step=1)

    if st.button("Prediksi Status Gizi"):
        # Normalisasi data input menggunakan scaler yang dimuat
        new_data = [[usia, berat_badan, tinggi_badan]]
        new_data_scaled = scaler.transform(new_data)

        # Prediksi status gizi menggunakan model KNN
        predicted_class = model.predict(new_data_scaled)

        # Mapping kembali hasil prediksi ke kategori asli
        bb_tb_reverse_mapping = {
            1: 'Gizi Buruk',
            2: 'Gizi Kurang',
            3: 'Normal',
            4: 'Beresiko Gizi Lebih',
            5: 'Gizi Lebih',
            6: 'Obesitas'
        }

        predicted_status = bb_tb_reverse_mapping[predicted_class[0]]

        # Menampilkan hasil prediksi
        st.success(f"Status Gizi untuk data balita: {predicted_status}")
