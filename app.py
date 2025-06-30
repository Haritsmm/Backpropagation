# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from predictor import predict_potensi, FEATURE_COLUMNS

# Baca data lama
DATA_PATH = "data_siswa_smp.csv"
data = pd.read_csv(DATA_PATH)

st.title("Prediksi Potensi Akademik Siswa SMP")

st.markdown("""
Silakan isi data Anda di bawah ini. Sistem akan memprediksi potensi akademik Anda (Sains, Bahasa, Sosial, Teknologi) 
dan membandingkan dengan data siswa lainnya.
""")

# Buat form input
with st.form("input_form"):
    usia = st.number_input("Usia", min_value=10, max_value=20, value=12)
    nilai_mtk = st.number_input("Nilai Matematika", min_value=0, max_value=100, value=80)
    nilai_ipa = st.number_input("Nilai IPA", min_value=0, max_value=100, value=80)
    nilai_ips = st.number_input("Nilai IPS", min_value=0, max_value=100, value=80)
    nilai_bindo = st.number_input("Nilai Bahasa Indonesia", min_value=0, max_value=100, value=80)
    nilai_bing = st.number_input("Nilai Bahasa Inggris", min_value=0, max_value=100, value=80)
    nilai_tik = st.number_input("Nilai TIK", min_value=0, max_value=100, value=80)
    minat_sains = st.slider("Minat Sains (1-5)", 1, 5, 3)
    minat_bahasa = st.slider("Minat Bahasa (1-5)", 1, 5, 3)
    minat_sosial = st.slider("Minat Sosial (1-5)", 1, 5, 3)
    minat_teknologi = st.slider("Minat Teknologi (1-5)", 1, 5, 3)
    
    submitted = st.form_submit_button("Simulasi")

if submitted:
    # Buat dict data input
    input_data = {
        'Usia': usia,
        'Nilai Matematika': nilai_mtk,
        'Nilai IPA': nilai_ipa,
        'Nilai IPS': nilai_ips,
        'Nilai Bahasa Indonesia': nilai_bindo,
        'Nilai Bahasa Inggris': nilai_bing,
        'Nilai TIK': nilai_tik,
        'Minat Sains': minat_sains,
        'Minat Bahasa': minat_bahasa,
        'Minat Sosial': minat_sosial,
        'Minat Teknologi': minat_teknologi
    }
    
    # Prediksi
    potensi_pred, prob = predict_potensi(input_data)
    
    st.success(f"Prediksi Potensi Akademik Anda adalah: **{potensi_pred}**")
    
    # Tampilkan probabilitas
    prob_dict = {
        "Bahasa": prob[0],
        "Sains": prob[1],
        "Sosial": prob[2],
        "Teknologi": prob[3],
    }
    st.write("Probabilitas:")
    st.write(prob_dict)
    
    # Simpan data input + hasil prediksi ke CSV
    new_row = input_data.copy()
    new_row["Potensi"] = potensi_pred
    
    # Tambah ke data lama
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
    data.to_csv(DATA_PATH, index=False)
    
    st.success("Data berhasil disimpan.")
    
    # Visualisasi perbandingan
    fig = px.histogram(
        data,
        x="Potensi",
        title="Distribusi Potensi Akademik Siswa",
        color="Potensi",
        text_auto=True
    )
    st.plotly_chart(fig)
    
    st.subheader("Data Siswa Terbaru:")
    st.dataframe(data.tail(10))
