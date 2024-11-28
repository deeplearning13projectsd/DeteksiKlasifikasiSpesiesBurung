import streamlit as st

def add_bg_from_url():
    """Sets the background image for the Streamlit app."""
    st.markdown(
        """
        <style>
            .stApp {
                background-image: url("https://images.unsplash.com/photo-1444464666168-49d633b86797?q=80&w=1469&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
                background-size: cover;
                background-position: top center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

def create_html_content(language):
    if language == "id":
        title = "Implementasi Model Transfer Learning Arsitektur ConvNeXt untuk Klasifikasi Suara Burung di Taman Nasional Way Kambas"
        upload_text = "Yuk, Upload disini audionya"
        footer_text = "© Developer: Kelompok 13 Deep Learning"
    else:
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
            body {{
                font-family: 'Roboto', sans-serif;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                color: #fff;
                background-color: rgba(0, 0, 0, 0.7);
            }}
            h1, h3 {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .upload-form {{
                background-color: rgba(255, 255, 255, 0.8);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            }}
            input[type="file"], input[type="submit"] {{
                padding: 10px;
                border-radius: 5px;
                border: none;
                margin: 5px 0;
            }}
            input[type="submit"] {{
                background-color: #007bff;
                color: #fff;
                cursor: pointer;
            }}
            input[type="submit"]:hover {{
                background-color: #0056b3;
            }}
            .logo-container {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 20px;
            }}
            .logo {{
                width: 65px;
                height: 65px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            }}
        </style>
    </head>
    <body>
        <div class="logo-container">
            <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo1.png" alt="Logo 1" class="logo">
            <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo2.png" alt="Logo 2" class="logo">
            <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo3.png" alt="Logo 3" class="logo">
            <img src="https://raw.githubusercontent.com/alberanalafean22/DeteksiKlasifikasiSpesiesBurung/main/Deployment/asset/Logo0.png" alt="Logo 0" class="logo">
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

# Select language
language = st.selectbox("Pilih Bahasa", ["Indonesia", "English"])

# Display HTML content based on language
st.components.v1.html(create_html_content(language.lower()), height=500, scrolling=False)
