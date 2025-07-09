# import streamlit as st
# import numpy as np

# import tensorflow as tf
# model = tf.keras.models.load_model("alzheimer_detection_model.h5")

# from PIL import Image
# import time 



# # Define class names
# class_names = ['Non Demented', 'Moderate Dementia', 'Mild Dementia', 'Very Mild Dementia']

# # Streamlit UI
# st.title("QuadPhase Alzheimer's Detection")
# st.write("Upload a brain MRI image to classify the Alzheimer's stage.")

# uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert('RGB')
#     st.image(img, caption='Uploaded MRI.', use_container_width=True)
    
#     # Preprocess the image
#     img = img.resize((128, 128))
#     img_array = np.array(img) 
#     img_array = img_array.reshape(1,128,128,3)
    

#     with st.spinner("Analyzing MRI... Please wait"):
#         time.sleep(3)
#         prediction = model.predict(img_array)
#         predicted_class = class_names[np.argmax(prediction)]
#         confidence = np.max(prediction) * 100

#     st.success(f"Prediction: **{predicted_class}**")
#     st.info(f"Confidence: {confidence:.2f}%")



#     # Prediction
#     # prediction = model.predict(img_array)
#     # predicted_class = class_names[np.argmax(prediction)]

import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import time

# Load the trained model
model = keras.models.load_model("alzheimer_detection_model.h5")

# Define class names
class_names = ['Non Demented', 'Moderate Dementia', 'Mild Dementia', 'Very Mild Dementia']

# Streamlit UI
st.title("QuadPhase Alzheimer's Detection")
st.write("Upload a brain MRI image to classify the Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI.', use_container_width=True)
    
    # ✅ Preprocess the image
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize pixel values (optional but good practice)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, 128, 128, 3)
    
    with st.spinner("Analyzing MRI... Please wait"):
        time.sleep(3)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

   
