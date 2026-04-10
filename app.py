import streamlit as st
import numpy as np
from PIL import Image
from utils import compress_image_auto_k

st.set_page_config(page_title="SVD Compression", layout="wide")

st.title("🚀 AI Image Compression using SVD")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
threshold = st.slider("Quality", 0.85, 0.99, 0.95)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original")

    compressed, k, score = compress_image_auto_k(img, threshold)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    with col2:
        st.image(compressed, caption="Compressed")

    st.write(f"k value: {k}")
    st.write(f"SSIM: {score:.4f}")
