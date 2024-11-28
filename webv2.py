import streamlit as st

# Fungsi untuk menambahkan latar belakang
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
        unsafe_allow_html=True,
    )

# Tambahkan latar belakang
add_bg_from_url()

# Fungsi untuk menambahkan CSS khusus
def add_custom_styles():
    st.markdown(
        """
        <style>
        .custom-uploader {
            background-color: rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 6px 30px rgba(0, 0, 0, 0.7);
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }
        .custom-uploader input[type="file"] {
            margin: 15px 0;
            padding: 12px;
            border: none;
            border-radius: 10px;
            background-color: white;
            color: black;
            font-size: 18px;
            transition: transform 0.3s;
        }
        .custom-uploader input[type="file"]:hover {
            transform: scale(1.05);
        }
        .custom-uploader .stButton>button {
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            background-color: #28a745;
            color: white;
            font-size: 18px;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.3s;
        }
        .custom-uploader .stButton>button:hover {
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
        """,
        unsafe_allow_html=True,
    )

# Tambahkan CSS khusus
add_custom_styles()

# HTML Kustom untuk Elemen Statis
html_code = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>deeplearning13</title>
</head>
<body>
    <div> 
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo1.png" alt="Logo 1" width="65" height="65">    
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo2.png" alt="Logo 2" width="65" height="65"> 
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo3.png" alt="Logo 3" width="65" height="65">
        <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo0.png" alt="Logo 4" width="65" height="65">
    </div>
    <h1>Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas</h1>
    <h3>Yuk, Upload disini audionya</h3>
</body>
</html>
"""

# Tampilkan HTML di atas dalam Streamlit
st.components.v1.html(html_code, height=300, scrolling=False)

# Bagian File Uploader
st.markdown(
    """
    <div class="custom-uploader">
        <h3>Upload Audio</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# File uploader Streamlit
uploaded_file = st.file_uploader("", type=["mp3", "wav"])

# Proses jika file diunggah
if uploaded_file is not None:
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type}
    st.write("File Details:", file_details)
    st.success(f"File {uploaded_file.name} berhasil diunggah!")

