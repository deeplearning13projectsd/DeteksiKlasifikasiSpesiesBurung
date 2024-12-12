
import streamlit as st
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

#PROSES GAMBAR MELS

#MODEL
model_path = 'https://drive.google.com/file/d/1-6TpLc73-nLMn1z6vQEVjbr5uZHZLnsq/view?usp=sharing'
model = load_model(model_path)
#PREDIKSI
predicted_probabilities = model.predict(img_array, verbose=1)
predict = np.argmax(predicted_probabilities, axis=1)[0]
class_indices = {'Cacomantis Merulinus': 0,
                 'Culicicapa Ceylonensis': 1,
                 'Geopelia Striata': 2,
                 'Halcyon Smyrnensis':3,
                 'Ninox Scutulata':4,
                 'Orthotomus Ruficeps':5,
                 'Pitta Sordida':6,
                 'Prinia Flaviventris':7,
                 'Spilopelia Chinensis':8,
                 'Surniculus Lugubris':9
                 }
class_labels = {v: k for k, v in class_indices.items()}
predicted_label = class_labels.get(predicted_class, "Unknown")
accuracy = predicted_probabilities[0][predicted_class] * 100

# Fungsi untuk mendapatkan URL gambar berdasarkan nama file
def get_image_url(predict):    
    if predict == 0:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/1.png"
    elif predict == 1:
        predict "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/2.png"
    elif predict == 2:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/3.png"
    elif predict == 3:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/4.png"
    elif predict == 4:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/5.png"
    elif predict == 5:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/6.png"
    elif predict == 6:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/7.png"
    elif predict == 7:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/8.png"
    elif predict == 8:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/9.png"
    else:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/kelas/10.png"


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
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo3.png" alt="Logo 3" width="85" height="85">
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
    
    # Dapatkan URL gambar berdasarkan nama file
    image_url = get_image_url(file_name)
    
    # Tampilkan gambar hasil klasifikasi
    st.image(image_url, use_column_width=False)


# Menggunakan st.components.v1.html() untuk menampilkan teks dengan kontrol lebih banyak
html_code = """<div class="footer" style="text-align: center; color: white;">
    <h2>Version Beta 1.0.4</h2>
    <h3>On Progress Deployment</h3>
</div>
"""
st.components.v1.html(html_code, height=50)
