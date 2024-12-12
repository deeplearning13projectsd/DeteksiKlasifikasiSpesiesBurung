import streamlit as st
import numpy as np
import librosa
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input  # For example, you may use another model

# Fungsi untuk preprocessing audio dan ekstraksi fitur
def preprocess_audio(file):
    # Baca file audio menggunakan pydub
    audio = AudioSegment.from_file(file, format="mp3")
    # Konversi ke mono dan atur sample rate
    audio = audio.set_channels(1).set_frame_rate(16000)

    # Simpan sementara sebagai WAV untuk digunakan oleh librosa
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio.export(temp_audio.name, format="wav")
        temp_audio_path = temp_audio.name

    # Load audio menggunakan librosa
    y, sr = librosa.load(temp_audio_path, sr=16000)
    os.remove(temp_audio_path)  # Hapus file sementara

    # Hitung energi pada setiap frame
    frame_length = 1024
    hop_length = 512
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])

    # Ambil segmen dengan energi tertinggi (5 detik)
    top_frame = np.argmax(energy)
    start_sample = top_frame * hop_length
    end_sample = start_sample + 5 * sr
    y_segment = y[start_sample:end_sample]

    return y_segment, sr

# Fungsi untuk mendapatkan Mel Spectrogram
def extract_melspectrogram(y, sr):
    # Hitung Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels for better visual representation
    return mel_spec_db

# Fungsi untuk menyiapkan input gambar untuk model
def prepare_image(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Fungsi untuk mendapatkan URL gambar berdasarkan nama file
def get_image_url(hasil):    
    if hasil == "1.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif hasil == "2.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    elif hasil == "3.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"
    elif hasil == "4.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/4.png"
    elif hasil == "5.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/5.png"
    elif hasil == "6.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/6.png"
    elif hasil == "7.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/7.png"
    elif hasil == "8.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/8.png"
    elif hasil == "9.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/9.png"
    else:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/10.png"

# Fungsi untuk memuat model dan melakukan prediksi
def predict_class(img_array):
    # Memuat model
    model = load_model('hgjjgg.h5')  # Pastikan untuk memberikan path yang benar ke model Anda
    # Lakukan prediksi
    prediction = model.predict(img_array)
    prediction= np.argmax(prediction)
    return prediction  # Mengembalikan indeks kelas yang diprediksi

# Fungsi untuk mendapatkan akurasi model
def get_accuracy():
    accuracy = 0.470  # Angka akurasi misalnya diambil dari hasil model
    return accuracy

# Tambahkan latar belakang ke Streamlit
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

# Tambahkan latar belakang
add_bg_from_url()

# Kode HTML utama untuk form upload
html_code = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>deeplearning13</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-size: cover;
            color: white;
            font-family: 'Roboto', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }
        .upload-form {
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 6px 30px rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        input[type="file"] {
            margin: 15px 0;
            padding: 12px;
            border: none;
            border-radius: 10px;
            background-color: white;
            color: black;
            font-size: 18px;
            transition: transform 0.3s;
        }
        input[type="file"]:hover {
            transform: scale(1.05);
        }
        input[type="submit"] {
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            background-color: #28a745;
            color: white;
            font-size: 18px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #218838;
            transform: scale(1.05);
        }
        img {
            margin-top: 20px;
            max-width: 80%;
            border-radius: 15px;
            box-shadow: 0 6px 30px rgba(0, 0, 0, 0.7);
        }
    </style>
</head>
<body>
    <div> 
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo1.png" alt="Logo 1" width="85" height="85">    
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo2.png" alt="Logo 2" width="85" height="85">
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo0.png" alt="Logo 4" width="85" height="85">
    </div>
    <h1>Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas</h1>
    <h2>© Developer: Kelompok 13 Deep Learning</h2>
</body>
</html>
"""

# Tampilkan form HTML
st.components.v1.html(html_code, height=400, scrolling=False)

# Tangkap file audio yang diunggah
uploaded_file = st.file_uploader("Upload File Audio Here", type=["mp3"])

# Proses file audio
if uploaded_file:
    # Menampilkan akurasi
    accuracy = get_accuracy()
    
    # Tampilkan akurasi menggunakan st.metric
    st.metric(label="Akurasi Model ConvNeXt type base", value=f"{accuracy:.4f}", delta=None)
    
    # Ambil nama file
    file_name = uploaded_file.name
    
    # Proses audio
    y_segment, sr = preprocess_audio(uploaded_file)

    # Ekstraksi Mel Spectrogram
    mel_spec_db = extract_melspectrogram(y_segment, sr)

    # Konversi Mel Spectrogram ke Gambar
    img_array = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr)
    fig = plt.gcf()
    plt.close(fig)  # Close the figure to prevent it from displaying in Streamlit

    # Simpan mel spectrogram sebagai gambar
    image_path = 'temp_melspec.png'
    fig.savefig(image_path)
    
    # Persiapkan gambar untuk prediksi model
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = prepare_image(img_array)
    
    # Prediksi kelas menggunakan model
    predicted_class = predict_class(img_array)

    # Tampilkan hasil prediksi
    st.write(f'Prediksi Kelas: {predicted_class}')

    # Dapatkan URL gambar berdasarkan prediksi kelas
    image_url = get_image_url(f"{predicted_class}.mp3")

    # Tampilkan gambar hasil klasifikasi
    st.image(image_url, use_column_width=False)

# Menggunakan st.components.v1.html() untuk menampilkan teks dengan kontrol lebih banyak
html_code = """<div class="footer" style="text-align: center; color: white;">
    <h2>Version Beta 1.0.4</h2>
    <h3>On Progress Deployment</h3>
</div>
"""
st.components.v1.html(html_code, height=50)

