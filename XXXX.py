import streamlit as st
import librosa
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tensorflow.keras.models import load_model

# Mengatur Kaggle API
os.makedirs('~/.kaggle', exist_ok=True)
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Mengunduh model dari Kaggle
!kaggle kernels pull alberanalafean/transferlearningconvnexttypebase -p ./  # Mengunduh kernel
# Pastikan untuk mengekstrak file .keras dari kernel yang diunduh
model = load_model('convnextaugmentasiepochs50')  # Ganti dengan nama file model yang sesuai

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
    audio, sr = librosa.load(file, sr=16000)
    energy = np.square(audio).mean(axis=0)
    threshold = np.percentile(energy, 90)
    high_energy_frames = np.where(energy > threshold)[0]
    segment_duration = 16000 * 5

    if len(high_energy_frames) > segment_duration:
        high_energy_audio = audio[high_energy_frames[0]:high_energy_frames[0] + segment_duration]
    else:
        high_energy_audio = audio[:segment_duration]

    return high_energy_audio, sr

# Fungsi untuk mengonversi audio menjadi Mel-spectrogram
def audio_to_melspectrogram(high_energy_audio, sr):
    melspec = librosa.feature.melspectrogram(y=high_energy_audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(melspec_db, aspect='auto', origin='lower', cmap='inferno')
    canvas = FigureCanvas(fig)
    buf = io.BytesIO()
    canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    return buf

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
    img_array = np.array(plt.imread(spectrogram_buf)).reshape(1, 224, 224, 3)  # Pastikan ukuran sesuai dengan input model
