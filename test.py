import streamlit as st

def add_bg_from_url():
    """Mengatur gambar latar belakang untuk aplikasi Streamlit."""
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

# Fungsi untuk membuat konten HTML berdasarkan bahasa
def create_html_content(language):
    if language == "id":
        # Konten untuk bahasa Indonesia
        title = "Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas"
        upload_text = "Yuk, Upload disini audionya"
        footer_text = "© Developer: Kelompok 13 Deep Learning"
    else:
        # Konten untuk bahasa Inggris (Anda bisa menyesuaikan dengan teks yang diinginkan)
        title = "Implementation of ConvNeXt Transfer Learning Architecture for Bird Sound Classification in Way Kambas National Park"
        upload_text = "Upload your audio here"
        footer_text = "© Developer: Deep Learning Group 13"

    html_code = f"""
    <!DOCTYPE html>
    <html lang="{language}">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{title}</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
      <style>
        {/* Gaya CSS yang sama seperti sebelumnya */}
      </style>
    </head>
    <body>
      <div>
        {/* Logo-logo */}
      </div>
      <h1>{title}</h1>
      <h3>{upload_text}</h3>
      <div class="upload-form">
        <form action="/upload" method="post" enctype="multipart/form-data">
          <input type="file" name="audio" accept="audio/*" required>
          <input type="submit" value="Upload Audio">
        </form>
      </div>
      <div class="footer">
        <h4>{footer_text}</h4>
      </div>
    </body>
    </html>
    """
    return html_code

# Pilih bahasa (Anda bisa menambahkan opsi lain jika diperlukan)
language = st.selectbox("Pilih Bahasa", ["Indonesia", "English"])

# Tampilkan konten HTML berdasarkan bahasa yang dipilih
st.components.v1.html(create_html_content(language.lower()), height=500, scrolling=False)
