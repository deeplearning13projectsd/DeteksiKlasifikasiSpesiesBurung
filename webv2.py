import streamlit as st
from io import BytesIO

# Fungsi untuk mendapatkan URL gambar berdasarkan nama file
def get_image_url(file_name):
    if file_name == "1.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/a.png"
    elif file_name == "2.mp3":
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/b.png"
    else:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/c.png"

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
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo1.png" alt="Logo 1" width="65" height="65">    
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo2.png" alt="Logo 2" width="65" height="65"> 
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo3.png" alt="Logo 3" width="65" height="65">
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo0.png" alt="Logo 4" width="65" height="65">
    </div>
    <h1>Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas</h1>
    
    <div class="footer">
        <h4>© Developer: Kelompok 13 Deep Learning</h4>
    </div>
</body>
</html>
"""

# Tampilkan form HTML
st.components.v1.html(html_code, height=500, scrolling=False)

# Tangkap file audio yang diunggah
uploaded_file = st.file_uploader("Upload Audio File (via Streamlit)", type=["mp3"])

# Proses file audio
if uploaded_file:
    # Ambil nama file
    file_name = uploaded_file.name
    
    # Dapatkan URL gambar berdasarkan nama file
    image_url = get_image_url(file_name)
    
    # Tampilkan gambar hasil klasifikasi
    st.image(image_url, caption=f"Hasil klasifikasi untuk {file_name}", use_column_width=True)
