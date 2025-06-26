import streamlit as st
import joblib
import numpy as np

# === Load model dan encoder ===
model = joblib.load('model_gender.pkl')
name_encoder = joblib.load('label_encoder.pkl')
gender_encoder = joblib.load('label_encoder.pkl')

# === UI Streamlit ===
st.title("Prediksi Gender Berdasarkan Nama Depan")
st.write("Masukkan nama depan untuk memprediksi apakah itu laki-laki atau perempuan.")

# === Input nama dari user ===
input_name = st.text_input("Masukkan Nama Depan")

# === Tombol Prediksi ===
if st.button("Prediksi"):
    if input_name.strip() == "":
        st.warning("Nama tidak boleh kosong.")
    else:
        # Encode nama input (pastikan sudah dikenal encoder)
        try:
            encoded_name = name_encoder.transform([input_name])
            prediction = model.predict(encoded_name.reshape(-1, 1))
            predicted_gender = gender_encoder.inverse_transform(prediction)
            st.success(f"Prediksi Gender: **{predicted_gender[0]}**")
        except ValueError:
            st.error("Nama yang dimasukkan belum pernah dilatih pada model. Harap coba nama lain.")
