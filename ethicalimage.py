#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import numpy as np
import pytesseract
from transformers import pipeline
import re

# Set Tesseract Path (Modify for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Load NLP Model for Ethical Analysis
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# List of Ethical Concerns
ETHICAL_CATEGORIES = [
    "privacy violation",
    "violence or abuse",
    "discrimination or bias",
    "misinformation",
    "graphic content",
    "environmental impact"
]

# Streamlit UI
st.title("🧐 Ethical Image Analysis Tool")

st.write("""
Upload an image, and this tool will analyze potential ethical concerns using **Computer Vision (OpenCV) and NLP**.
""")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Convert image to grayscale and apply thresholding for better OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    """Extract text from image using Tesseract OCR."""
    processed_img = preprocess_image(image)
    return pytesseract.image_to_string(processed_img, lang="eng")

def analyze_ethical_issues(text):
    """Classify ethical concerns in image description."""
    result = classifier(text, ETHICAL_CATEGORIES)
    issues = {label: score for label, score in zip(result["labels"], result["scores"]) if score > 0.5}
    return issues

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract Text from Image
    extracted_text = extract_text(image)

    if extracted_text.strip():
        st.subheader("📖 Extracted Text:")
        st.write(extracted_text)

        # Ethical Analysis
        st.subheader("⚖️ Ethical Concerns:")
        issues = analyze_ethical_issues(extracted_text)
        
        if issues:
            for issue, score in issues.items():
                st.write(f"🔴 **{issue}** - Confidence: {score:.2f}")
        else:
            st.write("✅ No significant ethical concerns detected.")
    else:
        st.warning("⚠️ No text found. Try uploading a clearer image.")

st.write("Developed by **Ammar Jamshed** | Powered by OpenCV & NLP 🚀")


# In[ ]:




