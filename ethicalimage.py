#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
import numpy as np
import pytesseract
import torch
from transformers import pipeline
from PIL import Image

# Load NLP model for ethical issue detection
ethics_analyzer = pipeline("text-classification", model="facebook/bart-large-mnli")

# Function to analyze an image using OpenCV
def analyze_image(image):
    # Convert to OpenCV format
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Perform OCR with Tesseract
    text = pytesseract.image_to_string(gray)

    return edges, text

# Function to detect ethical concerns
def detect_ethics(text):
    labels = ["violence", "hate speech", "offensive", "safe content"]
    result = ethics_analyzer(text, candidate_labels=labels)
    return result

# Streamlit App UI
st.title("🖼️ Ethical Image Analyzer")
st.write("Upload an image, and the system will analyze it for potential ethical issues.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Analyze Image
    edges, extracted_text = analyze_image(image)

    # Show Original & Processed Images
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")

    # Display Extracted Text
    st.subheader("Extracted Text from Image:")
    st.write(extracted_text if extracted_text.strip() else "No readable text found.")

    # Ethical Issue Detection
    if extracted_text.strip():
        st.subheader("Ethical Analysis of Extracted Text:")
        ethics_result = detect_ethics(extracted_text)
        for item in ethics_result:
            st.write(f"**{item['label'].capitalize()}**: {round(item['score'] * 100, 2)}% confidence")
    else:
        st.write("No text detected for ethical analysis.")


# In[ ]:




