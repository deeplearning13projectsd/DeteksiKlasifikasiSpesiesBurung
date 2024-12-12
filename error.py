import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load pre-trained model
MODEL_PATH = "https://drive.google.com/file/d/1-6TpLc73-nLMn1z6vQEVjbr5uZHZLnsq/view?usp=sharing"
model = load_model(MODEL_PATH)

# Class mapping
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

# Helper functions
def preprocess_audio(file):
    audio = AudioSegment.from_file(file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = audio[:5000]  # Keep first 5 seconds
    return audio

def audio_to_melspectrogram(audio):
    samples = np.array(audio.get_array_of_samples())
    sr = audio.frame_rate
    melspec = librosa.feature.melspectrogram(
        y=samples.astype(float), sr=sr, n_fft=1024, hop_length=512, n_mels=128
    )
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

def prepare_image(image_buf):
    img = image.load_img(image_buf, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Background HTML
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
    audio = preprocess_audio(uploaded_file)
    spectrogram_buf = audio_to_melspectrogram(audio)

    img_array = prepare_image(spectrogram_buf)

    # Predict using model
    predicted_probabilities = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predicted_probabilities, axis=1)[0]

    # Retrieve class and accuracy
    kelas = class_labels.get(predicted_class, "Unknown")
    akurasi = predicted_probabilities[0][predicted_class] * 100

    # Display results
    st.metric(label="Kelas Terprediksi", value=kelas)
    st.metric(label="Akurasi Prediksi", value=f"{akurasi:.2f}%")

    # Image URL display logic
    if predicted_class == 0:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif predicted_class == 1:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    elif predicted_class == 2:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"
    elif predicted_class == 3:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/4.png"
    else:
        image_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/5.png"

    st.image(image_url, caption=f"Kelas: {kelas}", use_column_width=True)

# Footer
st.markdown("### Version Beta 1.0.4")
st.markdown("### On Progress Deployment")
