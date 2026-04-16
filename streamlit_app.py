import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Model/cat_dog_model.keras", compile=False)
    return model

model = load_model()

st.markdown("<h1 style='text-align: center;'>🐶 Cat vs Dog Image Classifier</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center;'>Upload an image and the AI will predict whether it is a cat or a dog.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((150,150))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Dog 🐶"
        confidence = prediction
    else:
        label = "Cat 🐱"
        confidence = 1 - prediction

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence*100:.2f}%")
    compile=False

st.write("Developed by Krishna Parekh")
