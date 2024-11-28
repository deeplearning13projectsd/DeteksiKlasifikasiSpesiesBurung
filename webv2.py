import streamlit as st

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
        <h3>Yuk, Upload disini audionya</h3>
        <div class="upload-form">
            <form action="http://127.0.0.1:5000/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="audio" accept="audio/*" required>
                <input type="submit" value="Upload Audio">
            </form>
        </div>
    </div>
</body>
</html>
"""

st.components.v1.html(html_code, height=500, scrolling=False)
