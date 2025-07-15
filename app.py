import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime

# ======================
# Load model
# ======================
model = joblib.load("stunting_model.pkl")

# WHO Reference (interpolated)
who_reference = {
    (0, 'laki-laki'): (49.9, 1.9), (0, 'perempuan'): (49.1, 1.8),
    (6, 'laki-laki'): (67.6, 2.5), (6, 'perempuan'): (65.7, 2.4),
    (12, 'laki-laki'): (76.1, 2.9), (12, 'perempuan'): (74.0, 2.8),
    (24, 'laki-laki'): (87.8, 3.1), (24, 'perempuan'): (86.4, 3.1),
    (36, 'laki-laki'): (96.1, 3.4), (36, 'perempuan'): (95.1, 3.3),
    (48, 'laki-laki'): (103.3, 3.7), (48, 'perempuan'): (101.9, 3.5),
    (60, 'laki-laki'): (110.0, 4.0), (60, 'perempuan'): (108.7, 3.7),
}

def interpolate_who(age, gender):
    gender = gender.lower()
    keys = sorted(k for k in who_reference if k[1] == gender)
    ages = [k[0] for k in keys]
    medians = [who_reference[k][0] for k in keys]
    sds = [who_reference[k][1] for k in keys]
    median = np.interp(age, ages, medians)
    sd = np.interp(age, ages, sds)
    return median, sd

def calculate_z_score(age, gender, height):
    median, sd = interpolate_who(age, gender)
    return (height - median) / sd

def classify_z(z):
    if z < -3:
        return "Severely Stunted", "😢 Tinggi badan sangat kurang. Ayo jaga gizi dan cek ke posyandu ya!"
    elif z < -2:
        return "Stunted", "🙂 Tinggi badan kurang. Tetap semangat makan sehat dan minum susu!"
    else:
        return "Not Stunted", "🎉 Tinggi badan baik sesuai usianya. Pertahankan pola makan sehat!"

def hitung_umur_dalam_bulan(tgl_lahir):
    today = date.today()
    selisih_hari = (today - tgl_lahir).days
    umur_bulan = selisih_hari / 30.4375
    return round(umur_bulan, 1)

def calculate_bmi_zscore(bmi):
    if bmi < 18.49:
        return round(bmi, 2), "Underweight"
    elif 18.5 < bmi < 24.9:
        return round(bmi, 2), "Ideal"
    elif 25 < bmi > 27:
        return round(bmi, 2), "Overweight"
    else:
        return round(bmi, 2), "Obesitas"

# ======================
# UI
# ======================

st.set_page_config(page_title="Deteksi Stunting Anak", page_icon="🍼", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🍼 Deteksi Stunting & Status Gizi Anak</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Masukkan data anak, dan lihat apakah pertumbuhannya sesuai standar WHO</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    tgl_lahir = st.date_input("📅 Tanggal Lahir Anak", value=date(2021, 1, 1), max_value=date.today())
    tinggi_input = st.text_input("📏 Tinggi Badan (cm)", max_chars=5)
with col2:
    jk = st.selectbox("🚻 Jenis Kelamin", ["laki-laki", "perempuan"])
    berat_input = st.text_input("⚖️ Berat Badan (kg)", max_chars=5)

if st.button("🚀 Cek Pertumbuhan"):
    if not (tinggi_input.replace('.', '', 1).isdigit() and berat_input.replace('.', '', 1).isdigit()):
        st.error("❗ Tinggi dan berat harus berupa angka valid.")
    else:
        tinggi = float(tinggi_input)
        berat = float(berat_input)
        umur = hitung_umur_dalam_bulan(tgl_lahir)

        if not (30 <= tinggi <= 160):
            st.warning("⚠️ Tinggi harus antara 30-160 cm.")
        elif not (2 <= berat <= 99.9):
            st.warning("⚠️ Berat harus antara 2-99.9 kg.")
        else:
            # STUNTING
            z = calculate_z_score(umur, jk, tinggi)
            status, penjelasan = classify_z(z)
            jk_enc = 0 if jk == "laki-laki" else 1
            data = pd.DataFrame([[umur, jk_enc, tinggi]], columns=["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"])
            pred = model.predict(data)[0]

            # BMI-FOR-AGE
            tinggi_m = tinggi / 100
            bmi = berat / (tinggi_m ** 2)
            bmi_z, bmi_status = calculate_bmi_zscore(bmi)

            # OUTPUT
            st.success(f"📊 **Z-Score (Stunting)**: {z:.2f}")
            st.info(f"📌 **Status WHO**: {status}")
            st.markdown(f"🧒 **Penjelasan**: {penjelasan}")
            st.markdown(f"🤖 **Prediksi Model**: {pred}")

            st.markdown("---")
            st.markdown("### 🧮 Status Gizi Berdasarkan BMI")
            st.info(f"**Status WHO BMI-for-Age:** {bmi_status}\n\n**BMI:** {bmi_z}")

            st.markdown("---")
            st.markdown("💡 *Tips:* Pastikan anak mendapat makanan bergizi, ASI eksklusif, dan rutin ke posyandu.")