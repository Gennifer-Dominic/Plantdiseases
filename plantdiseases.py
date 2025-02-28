import streamlit as st
import cv2
from keras.models import model_from_json
from PIL import Image
import numpy as np
from keras_preprocessing.image import load_img
import tempfile
import  os
from numpy import asarray
import json


json_file=open("plantdiseases.json","r")
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights("plantdiseases.h5")

with open('class_indices.json', 'r') as file:
    data = json.load(file)

res = {}
for key, value in data.items():
    res[int(key)] = value

def load_and_preprocess_image(image_path, target_size=(224,224)):
  if image_path is not None:
      # Open the image
      image = Image.open(image_path)
      image = image.convert("RGB")
      new_size = (224, 224)
      image_resized = image.resize(new_size)
      img_array = np.array(image_resized, dtype=np.float32)
      img_array = np.expand_dims(img_array, axis=0)
      image_resized = img_array / 255.0  # Normalization to [0,1]
      return image_resized

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name


st.title("Plant Diseases Detection from Uploaded Images")

def is_image_valid(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img.verify()  # Verifies if it is a valid image
        return True
    except:
        return False

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file and is_image_valid(uploaded_file):
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
else:
    st.error("Please upload a valid image file (jpg, png, jpeg).")

if st.button("Predict"):
    predicted_class_name = predict_image_class(model, uploaded_file, res)
    # Output the result
    st.write(f"Predicted Class Name: {predicted_class_name}")