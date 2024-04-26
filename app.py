""""
We are going to deploy our model using Gradio.
"""
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('melanoma_cancer_model.h5')

# Define the function to make predictions
def classify_image(img):
    img = np.expand_dims(img, axis=0)
    # Resize image
    resized_img = tf.image.resize(img, [160, 160])
    # Predict the image
    prediction = model.predict(resized_img)[0][0]
    # Convert to float value
    prediction = float(prediction)
    # return dictionary for Gradio
    return {"melanoma": prediction, "not melanoma": 1 - prediction}
    

# Launch the Gradio interface
gr.Interface(fn=classify_image, inputs='image', outputs="label").launch()
# Launch shareble Gradio interface 
# gr.Interface(fn=classify_image, inputs='image', outputs="label").launch(share=True)

