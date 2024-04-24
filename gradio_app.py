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
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    # Resize data for the model (128, 128,3)
    resized_img = tf.image.resize(img, [128, 128])

    prediction = model.predict(resized_img)
    # return melanuma or not melanoma
    if prediction < 0.5:
        return {"Not Melanoma": float(1-prediction)}
    else:
        return {"Melanoma": float(prediction)}
    

# add camera input
image = gr.inputs.Image(shape=(128, 128), type='pil', image_mode='RGB')
label = gr.outputs.Label(num_top_classes=2)

# Run on localhost
gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch(share=True,server_name="0.0.0.0", server_port=7860)
#gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch( server_port=7860)

# Run the app