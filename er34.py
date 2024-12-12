import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io
import soundfile as sf

def process_audio(file):
    # Load audio file
    audio, sr = librosa.load(file, sr=None)

    # Convert to mono and resample to 16000 Hz
    audio = librosa.to_mono(audio)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Extract 5 seconds with the highest energy
    hop_length = 512
    frame_energy = librosa.feature.rms(y=audio, frame_length=hop_length, hop_length=hop_length).flatten()
    max_energy_frame = np.argmax(frame_energy)
    start_sample = max_energy_frame * hop_length
    end_sample = start_sample + 5 * 16000  # 5 seconds at 16000 Hz
    audio = audio[start_sample:end_sample]

    # Ensure audio length is exactly 5 seconds
    if len(audio) < 5 * 16000:
        audio = np.pad(audio, (0, 5 * 16000 - len(audio)))

    return audio

def generate_melspectrogram(audio, sr=16000):
    # Generate Mel spectrogram
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Save spectrogram as an image
    fig, ax = plt.subplots()
    librosa.display.specshow(melspec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return buf

def preprocess_image(image_path):
    # Preprocess image for prediction
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_image_url(predict):
    # Return image URL based on prediction
    if predict == 1:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif predict == 2:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    else:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"

# Load pre-trained model
model = load_model('https://drive.google.com/file/d/1-6TpLc73-nLMn1z6vQEVjbr5uZHZLnsq/view?usp=drive_link')

# Streamlit UI
st.title("Klasifikasi Suara Burung dengan Mel Spectrogram")

uploaded_file = st.file_uploader("Upload file audio (MP3)", type=["mp3"])

if uploaded_file is not None:
    with st.spinner("Memproses audio..."):
        # Process audio
        audio = process_audio(uploaded_file)

        # Generate Mel spectrogram
        spectrogram_buffer = generate_melspectrogram(audio)

        # Convert spectrogram to image
        spectrogram_image = Image.open(spectrogram_buffer)
        spectrogram_image.save("temp_spectrogram.png")

        # Preprocess image for prediction
        img_array = preprocess_image("temp_spectrogram.png")

        # Predict with model
        predicted_probabilities = model.predict(img_array, verbose=1)
        predicted_class = np.argmax(predicted_probabilities, axis=1)[0]

        # Map class index to label
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
        predicted_label = class_labels.get(predicted_class, "Unknown")
        predicted_accuracy = predicted_probabilities[0][predicted_class] * 100

        # Display prediction results
        st.subheader(f"Prediksi: {predicted_label}")
        st.write(f"Akurasi: {predicted_accuracy:.2f}%")

        # Display corresponding image
        image_url = get_image_url(predicted_class)
        st.image(image_url, caption=f"Kelas: {predicted_label}", use_column_width=True)

        # Display spectrogram image
        st.image("temp_spectrogram.png", caption="Mel Spectrogram", use_column_width=True)
