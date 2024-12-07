import streamlit as st

st.title("🎈 Aplikasi Prediksi Status Gizi balita")
st.write(
    "Aplikasi berbasis web untuk memprediksi status Gizi Balita"
)

# Menambahkan sidebar
st.sidebar.header('Pilih Algoritma')

# Menambahkan dropdown menu di sidebar
option = st.sidebar.selectbox(
    'Pilih Algoritma',
    ['Algoritma KNN', 'Algoritma Decision Tree (DT)', 'Algoritma Naive Bayes']
)
