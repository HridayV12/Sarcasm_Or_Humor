# app.py
import streamlit as st
from fusion_model import late_fusion_predict  # Import from your newly created module

st.title("Sarcasm & Humor Classification")

# File upload for image and text
image_input = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
text_input = st.text_area("Or Enter Text", "")

if st.button("Predict"):
    if image_input or text_input:
        result, probs = late_fusion_predict(image_input=image_input, text_input=text_input)

        if result is not None:
            if result == 0:
                st.write("Prediction: Not Funny or Not Sarcastic")
            else:
                st.write("Prediction: Funny or Sarcastic")

            st.write(f"Class probabilities: {probs}")
        else:
            st.write("Please upload an image or enter text for prediction.")
