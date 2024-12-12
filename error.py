import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from pydub import AudioSegment
from pydub.utils import which
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gdown




# Unduh model yang telah dilatih
url = 'https://drive.google.com/uc?id=1-6TpLc73-nLMn1z6vQEVjbr5uZHZLnsq' 
output_path = 'convnextaugmentasiepochs50.keras'
gdown.download(url, output_path, quiet=False)
model = load_model(output_path)

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

import librosa
import numpy as np

def preprocess_audio(file):
    # Membaca file audio menggunakan librosa (menggunakan librosa.load untuk membaca file)
    audio, sr = librosa.load(file, sr=16000, mono=True)  # Setel sample rate ke 16000 dan mono
    
    # Menghitung energi audio (energi rata-rata per frame)
    energy = np.square(audio).mean(axis=0)
    
    # Tentukan ambang batas energi untuk mendeteksi segmen dengan energi tinggi
    threshold = np.percentile(energy, 90)  # Ambang batas di persentil 90 untuk energi tinggi
    
    # Ambil indeks frame dengan energi tinggi
    high_energy_frames = np.where(energy > threshold)[0]
    
    # Tentukan durasi 5 detik (80.000 frame untuk 16000 Hz)
    segment_duration = 16000 * 5  # 5 detik
    
    # Ambil 5 detik dari segmen dengan energi tinggi
    if len(high_energy_frames) > segment_duration:
        high_energy_audio = audio[high_energy_frames[0]:high_energy_frames[segment_duration]]
    else:
        high_energy_audio = audio[:segment_duration]  # Jika kurang dari 5 detik, ambil sisa audio
    
    return high_energy_audio, sr


def audio_to_melspectrogram(audio):
    # Mengubah audio menjadi Mel-spectrogram menggunakan librosa
    samples = np.array(high_energy_audio.get_array_of_samples())
    sr = high_energy_audio.frame_rate
    melspec = librosa.feature.melspectrogram(
        y=samples.astype(float), sr=sr, n_fft=1024, hop_length=512, n_mels=128
    )
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

def prepare_image(image_buf):
    # Menyiapkan gambar untuk input model
    img = image.load_img(image_buf, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Background HTML untuk Streamlit
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

add_bg_from_url()

# Streamlit app header
st.title("Implementasi Model Transfer Learning untuk Klasifikasi Suara Burung")
st.markdown("### Kelompok 13 Deep Learning")

# File uploader
uploaded_file = st.file_uploader("Upload File Audio Here", type=["mp3"])

if uploaded_file:
    # Proses file audio
    audio = preprocess_audio(uploaded_file)
    spectrogram_buf = audio_to_melspectrogram(audio)

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
    if kelas == 0:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif kelas == 1:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    elif kelas == 2:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"
    elif kelas == 3:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/4.png"
    else:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/5.png"

    st.image(image_url, caption=f"Kelas: {kelas}", use_column_width=True)

# Footer
st.markdown("### Version Beta 1.0.4")
st.markdown("### On Progress Deployment")
