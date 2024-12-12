
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pydub import AudioSegment

import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pydub import AudioSegment
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from tempfile import NamedTemporaryFile

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

def generate_mel_spectrogram(y, sr):
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot dan simpan mel spectrogram sebagai gambar sementara
    fig, ax = plt.subplots()
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
    plt.axis('off')
    with NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
        fig.savefig(temp_img.name, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return temp_img.name

def prepare_image_for_model(image_path):
    # Load dan preprocess gambar untuk model
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_class(model, img_array, class_labels):
    # Lakukan prediksi dengan model
    predicted_probabilities = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predicted_probabilities, axis=1)[0]
    predicted_label = class_labels.get(predicted_class, "Unknown")
    accuracy = predicted_probabilities[0][predicted_class] * 100
    return predicted_label, accuracy

# Streamlit UI
st.title("Klasifikasi Suara Burung")
st.markdown("Implementasi Model Transfer Learning untuk Klasifikasi Suara Burung.")

uploaded_file = st.file_uploader("Unggah File Audio (MP3)", type=["mp3"])
if uploaded_file:
    try:
        # Proses audio
        y_segment, sr = preprocess_audio(uploaded_file)
        mel_image_path = generate_mel_spectrogram(y_segment, sr)

        # Siapkan gambar untuk model
        img_array = prepare_image_for_model(mel_image_path)

        # Load model
        model_path = 'https://drive.google.com/file/d/1-6TpLc73-nLMn1z6vQEVjbr5uZHZLnsq/view?usp=sharing' #5'  # Ganti dengan path model Anda
        model = load_model(model_path)

        # Klasifikasi
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
        predicted_label, accuracy = predict_class(model, img_array, class_labels)

        # Tampilkan hasil
        st.success(f"Prediksi: {predicted_label}")
        st.metric(label="Akurasi", value=f"{accuracy:.2f}%")
        st.image(mel_image_path, caption="Mel Spectrogram", use_column_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
