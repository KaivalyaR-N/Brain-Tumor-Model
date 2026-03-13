import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("brain_tumor_model.h5")

st.title("Brain Tumor Detection AI")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")