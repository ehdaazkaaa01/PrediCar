import streamlit as st
import joblib
import pandas as pd
import easyocr
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# === CONFIG HALAMAN ===
st.set_page_config(page_title="Prediksi Harga Mobil Toyota Bekas", layout="centered")

# === CSS UI ELEGAN NAVY & KUNING ===
def set_custom_background():
    st.markdown("""
        <style>
        body {
            background-color: #0c1a36;
            color: #f4f4f4;
        }
        .stApp {
            background-color: #0c1a36;
        }
        h1, h2, h3, h4, h5 {
            color: #f9cb40;
        }
        label, .css-1cpxqw2, .stTextInput label, .stNumberInput label {
            color: #f9cb40 !important;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #f9cb40;
            color: #000000;
            font-weight: bold;
        }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input {
            background-color: #1f2e4d;
            color: #ffffff;
        }
        .stFileUploader>div>div>button {
            background-color: #f9cb40;
            color: #000000;
            font-weight: bold;
        }
        .stMarkdown {
            color: #f4f4f4;
        }
        .custom-container {
            background-color: #122244;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(249, 203, 64, 0.2);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_background()

# === HEADER ===
st.markdown("""
    <h1 style='text-align: center;'>🚗 Prediksi Harga Mobil Toyota Bekas</h1>
    <hr style='border: 1px solid #f9cb40;'>
""", unsafe_allow_html=True)

# === CONTAINER UNTUK ISI UTAMA ===
with st.container():
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)

    # === GAMBAR MOBIL ===
    st.subheader("📷 Ambil Gambar Mobil dari Kamera")
    car_image = st.camera_input("Ambil Foto Mobil")
    if car_image:
        st.image(car_image, caption="Gambar Mobil", use_column_width=True)

    # === PLAT NOMOR ===
    st.subheader("🏷️ Ambil Gambar Plat Nomor dari Kamera")
    plate_image = st.camera_input("Ambil Foto Plat Nomor")
    plate_text = "Belum terbaca"
    if plate_image:
        img_path = "uploaded_plate.jpg"
        with open(img_path, "wb") as f:
            f.write(plate_image.read())

        model = YOLO("yolov8n.pt")
        reader = easyocr.Reader(['en'])

        results = model(img_path)
        for r in results:
            boxes = r.boxes
            if boxes:
                box = boxes[0].xyxy[0].cpu().numpy().astype(int)
                image = cv2.imread(img_path)
                plate_crop = image[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite("plate_crop.jpg", plate_crop)
                result = reader.readtext("plate_crop.jpg", detail=0)
                plate_text = ' '.join(result)
                st.success(f"📛 Nomor Plat: {plate_text}")
                break
        else:
            st.warning("Plat nomor tidak terdeteksi.")

    # === INPUT FITUR ===
    st.subheader("📋 Masukkan Spesifikasi Mobil")
    model_input = st.text_input("Model (contoh: Yaris)")
    year = st.number_input("Tahun", min_value=1990, max_value=2025, step=1)
    mileage = st.number_input("Jarak Tempuh (dalam mile)")
    tax = st.number_input("Pajak (£)", step=1)
    mpg = st.number_input("MPG")
    engineSize = st.number_input("Ukuran Mesin (Liter)")

    # === PREDIKSI ===
    def predict_price(input_data):
        model_knn = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[columns]
        input_scaled = scaler.transform(input_df)

        pred = model_knn.predict(input_scaled)
        return pred[0]

    # === TOMBOL PREDIKSI ===
    if st.button("🔍 Prediksi Harga"):
        if model_input and year and mileage and tax and mpg and engineSize:
            data_input = {
                'model': model_input,
                'year': year,
                'mileage': mileage,
                'tax': tax,
                'mpg': mpg,
                'engineSize': engineSize
            }
            harga = predict_price(data_input)
            st.success(f"💸 Estimasi Harga Mobil: £{harga:,.2f}")
            st.info(f"📛 Plat Nomor Terdeteksi: {plate_text}")
        else:
            st.error("Lengkapi semua input terlebih dahulu.")

    st.markdown("</div>", unsafe_allow_html=True)  # END custom container
