import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Function to set background from a URL (optional for design)
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

# Call function to set background
add_bg_from_url()

# Title of the application
st.title("Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas")

# Function to map output class to corresponding image URL
def hasilklasifikasi(output):
    if output == 1:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/a.png"
    elif output == 2:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/b.png"
    else:
        return "https://raw.githubusercontent.com/deeplearning13projectsd/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/c.png"

# Numeric input from the user
#output = st.number_input("Masukkan angka (1 atau 2 untuk kelas klasifikasi):", min_value=1, max_value=10, value=1)
output = st.number_input("Masukkan angka (1 atau 2 untuk kelas klasifikasi):", "")

# Get the image URL based on the classification output
image_url = hasilklasifikasi(output)

# Display the image
st.image(image_url, caption=f"Gambar untuk kelas {output}", use_column_width=True)
