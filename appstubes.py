
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pydub import AudioSegment



# Tangkap file audio yang diunggah
uploaded_file = st.file_uploader("Upload File Audio Here", type=["mp3"])

if uploaded_file is not None:
    # Ambil nama file
    audio = uploaded_file.name

    # Mengubah file input menjadi AudioSegment
    def convert_to_wav(input_file, output_file):
        try:
            # Membaca file input
            audio = AudioSegment.from_file(input_file)

            # Mengubah audio ke mono dan mengatur frame rate ke 16000
            audiovgk = audio.set_channels(1).set_frame_rate(16000)

            # Menyimpan hasilnya ke format .wav
            audiovgk.export(output_file, format="wav")
            print(f"File berhasil disimpan sebagai: {output_file}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat konversi audio: {e}")

    # Contoh penggunaan
    input_file = uploaded_file
    durasi = "output_audio.wav"
    convert_to_wav(input_file, durasi)
else:
    st.warning("Silakan unggah file audio terlebih dahulu.")
