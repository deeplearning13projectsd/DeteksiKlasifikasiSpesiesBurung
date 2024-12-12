import streamlit as st
import numpy as np
from pydub import AudioSegment
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from io import BytesIO
import requests
from PIL import Image

# Load model
model = tf.keras.models.load_model('jfjerigi.h5')

# Class mapping
class_indices = {'Cacomantis Merulinus': 0,
                 'Culicicapa Ceylonensis': 1,
                 'Geopelia Striata': 2,
                 'Halcyon Smyrnensis': 3,
                 'Ninox Scutulata': 4,
                 'Orthotomus Ruficeps': 5,
                 'Pitta Sordida': 6,
                 'Prinia Flaviventris': 7,
                 'Spilopelia Chinensis': 8,
                 'Surniculus Lugubris': 9}
class_labels = {v: k for k, v in class_indices.items()}

# Streamlit Interface
st.title("Prediksi Spesies Burung Berdasarkan Audio")
st.write("Upload file audio dan ikuti proses deteksi!")

uploaded_file = st.file_uploader("Upload file audio", type=["mp3"])

if uploaded_file is not None:
    # Proses audio
    st.audio(uploaded_file, format='audio/mp3')
    
    # Convert audio file to AudioSegment
    audio = AudioSegment.from_file(uploaded_file, format="mp3")
    
    # Set channels to 1 and frame rate to 16000
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    # Convert audio to numpy array and get audio samples
    audio_samples = np.array(audio.get_array_of_samples())
    
    # Compute energy and find the segment with the highest energy (5 seconds)
    hop_length = 512
    frame_length = 16000  # 1 second at 16 kHz
    energy = np.square(audio_samples)
    window_energy = np.array([np.sum(energy[i:i+frame_length]) for i in range(0, len(energy), hop_length)])
    
    # Get the segment with the highest energy
    start_sample = np.argmax(window_energy) * hop_length
    end_sample = start_sample + frame_length * 5  # 5 seconds segment
    
    audio_segment = audio_samples[start_sample:end_sample]
    
    # Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=16000, n_fft=1024, hop_length=512, n_mels=128)
    
    # Convert spectrogram to log scale
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Plot the spectrogram
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(log_spectrogram, cmap='jet', aspect='auto', origin='lower')
    ax.axis('off')
    
    # Save the figure as an image
    img_path = 'spectrogram.png'
    fig.savefig(img_path)
    
    # Convert saved image to img array for prediction
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predicted_probabilities = model.predict(img_array, verbose=1)
    predicted_class = np.argmax(predicted_probabilities, axis=1)[0]
    
    # Mapping index to label
    predicted_label = class_labels.get(predicted_class, "Unknown")
    predicted_accuracy = predicted_probabilities[0][predicted_class] * 100  # Accuracy in percentage
    
    # Show result based on predicted label
    if predicted_label == "Culicicapa Ceylonensis":
        img_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif predicted_label == "Geopelia Striata":
        img_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    else:
        img_url = "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"
    
    # Display results
    st.image(img_url, caption=f"Predicted Class: {predicted_label} with {predicted_accuracy:.2f}% accuracy", use_column_width=True)
    st.write(f"Predicted Class: {predicted_label}")
    st.write(f"Prediction Accuracy: {predicted_accuracy:.2f}%")
