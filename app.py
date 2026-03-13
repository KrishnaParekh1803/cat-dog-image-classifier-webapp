import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/cat_dog_model.keras")

def predict(image):

    image = image.resize((150,150))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    return {
        "Dog 🐶": float(prediction),
        "Cat 🐱": float(1 - prediction)
    }



with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🐶 Cat vs Dog Image Classifier")
    gr.Markdown("Upload an image and the AI model will classify it as **Cat or Dog**.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        output = gr.Label(num_top_classes=2, label="Prediction")

    predict_btn = gr.Button("🔍 Predict")

    predict_btn.click(predict, inputs=image_input, outputs=output)

    gr.Markdown("##### Developed by **Krishna Parekh** | AI/ML Project")

demo.launch(share=True)