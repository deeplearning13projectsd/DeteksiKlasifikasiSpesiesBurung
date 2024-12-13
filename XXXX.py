import streamlit as st
import librosa
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.models import load_model

import os
import json
import tensorflow as tf
from kaggle.api.kaggle_api_extended import KaggleApi
import streamlit as st

# Fungsi untuk mengunduh model cnn_melspec.h5 dari Kaggle API
def download_melspec_model_from_kaggle(kernel_name, output_file, dest_folder):
    try:
        model_path = os.path.join(dest_folder, output_file)

        # Cek apakah model sudah ada
        if os.path.exists(model_path):
            return False  # Model sudah ada, tidak perlu mengunduh ulang

        # Ambil kredensial dari secrets
        kaggle_username = st.secrets["kaggle"]["KAGGLE_USERNAME"]
        kaggle_key = st.secrets["kaggle"]["KAGGLE_KEY"]

        # Simpan kredensial ke file kaggle.json
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        os.makedirs(os.path.dirname(kaggle_json_path), exist_ok=True)

        with open(kaggle_json_path, 'w') as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)

        # Autentikasi dengan Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Unduh file model
        os.makedirs(dest_folder, exist_ok=True)
        api.kernels_output(kernel_name, path=dest_folder, force=True)

        return True
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh model: {str(e)}")
        return None

# Parameter untuk mengunduh model
kernel_name = "alberanalafean/transferlearningconvnexttypebase"
output_file = "convnextaugmentasiepochs50.keras"
dest_folder = "./models/"

# Unduh model jika diperlukan
download_status = download_melspec_model_from_kaggle(kernel_name, output_file, dest_folder)

# Path model yang akan dimuat
melspec_model_save_path = os.path.join(dest_folder, output_file)

# Memuat model cnn_melspec.h5
if os.path.exists(melspec_model_save_path):
    try:
        model = tf.keras.models.load_model(melspec_model_save_path)
        st.success("Model Melspec berhasil dimuat.")
    except Exception as e:
        st.error(f"Gagal memuat model Melspec: {str(e)}")
else:
    st.warning("Model Melspec tidak ditemukan.")









# Kelas untuk klasifikasi
class_indices = {
    'Cacomantis Merulinus': 0,
    'Culicicapa Ceylonensis': 1,
    'Geopelia Striata': 2,
    'Halcyon Smyrnensis': 3,
    'Ninox Scutulata': 4,
    'Orthotomus Ruficeps': 5,
    'Pitta Sordida': 6,
    'Prinia Flaviventris': 7,
    'Spilopelia Chinensis': 8,
    'Surniculus Lugubris': 9
}
class_labels = {v: k for k, v in class_indices.items()}

# Fungsi untuk memproses file audio menggunakan librosa
def preprocess_audio(file):
    # Membaca file audio menggunakan librosa
    audio, sr = librosa.load(file, sr=16000)  # Mengatur sample rate menjadi 16000 Hz
    
    # Menghitung energi audio (energi rata-rata per frame)
    energy = np.square(audio).mean(axis=0)
    
    # Tentukan ambang batas energi untuk mendeteksi segmen dengan energi tinggi
    threshold = np.percentile(energy, 90)  # Ambang batas di persentil 90 untuk energi tinggi
    
    # Ambil indeks frame dengan energi tinggi
    high_energy_frames = np.where(energy > threshold)[0]
    
    # Tentukan durasi 5 detik (16000 frame untuk 16000 Hz)
    segment_duration = 16000 * 5  # 5 detik
    
    # Ambil 5 detik dari segmen dengan energi tinggi
    if len(high_energy_frames) > segment_duration:
        high_energy_audio = audio[high_energy_frames[0]:high_energy_frames[segment_duration]]
    else:
        high_energy_audio = audio[:segment_duration]  # Jika kurang dari 5 detik, ambil sisa audio
    
    return high_energy_audio, sr


# Fungsi untuk mengonversi audio menjadi Mel-spectrogram
def audio_to_melspectrogram(high_energy_audio, sr):
    # Mengubah audio menjadi Mel-spectrogram menggunakan librosa
    melspec = librosa.feature.melspectrogram(y=high_energy_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Membuat visualisasi dari Mel-spectrogram
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(melspec_db, aspect='auto', origin='lower', cmap='inferno')
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    return buf

# Fungsi untuk menyiapkan gambar untuk input model
def prepare_image(image_buf):
    # Menyiapkan gambar untuk input model
    img = image.load_img(image_buf, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Fungsi untuk menambahkan background ke halaman Streamlit
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1444464666168-49d633b86797?q=80&w=1469&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: top center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Menambahkan background
add_bg_from_url()

# Streamlit app header
st.title("Implementasi Model Transfer Learning untuk Klasifikasi Suara Burung")
st.markdown("### Kelompok 13 Deep Learning")

# File uploader untuk audio
uploaded_file = st.file_uploader("Upload File Audio Here", type=["mp3"])

if uploaded_file:
    # Proses file audio
    high_energy_audio, sr = preprocess_audio(uploaded_file)
    
    # Konversi audio menjadi Mel-spectrogram
    spectrogram_buf = audio_to_melspectrogram(high_energy_audio, sr)

    # Menyiapkan gambar untuk input model
    img_array = prepare_image(spectrogram_buf)

    # Prediksi menggunakan model
    predicted_probabilities = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predicted_probabilities, axis=1)[0]

    # Ambil kelas dan akurasi
    kelas = class_labels.get(predicted_class, "Unknown")
    akurasi = predicted_probabilities[0][predicted_class] * 100

    # Menampilkan hasil prediksi
    st.metric(label="Kelas Terprediksi", value=kelas)
    st.metric(label="Akurasi Prediksi", value=f"{akurasi:.2f}%")

    # Menampilkan gambar sesuai kelas
    if kelas == 'Cacomantis Merulinus':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/a.png"
    elif kelas  == 'Culicicapa Ceylonensis':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/b.png"
    elif kelas  == 'Geopelia Striata':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/c.png"
    elif kelas  == 'Halcyon Smyrnensis':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/d.png"
    elif kelas  == 'Ninox Scutulata':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/e.png"
    elif kelas  == 'Orthotomus Ruficeps':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/f.png"
    elif kelas  == 'Pitta Sordida':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/g.png"
    elif kelas  == 'Prinia Flaviventris':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/h.png"
    elif kelas  == 'Spilopelia Chinensis':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/i.png"
    elif kelas  == 'Spilopelia Chinensis':
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/j.png"

    st.image(image_url, caption=f"Kelas: {kelas}", use_column_width=True)

# Footer
st.markdown("### Version Beta 1.0.4")
st.markdown("### On Progress Deployment")
