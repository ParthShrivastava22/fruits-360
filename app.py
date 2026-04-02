import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. THE SYNCED CLASS LIST (The 9 classes from your project)
class_names = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Orange']

@st.cache_resource
def load_model():
    # Use the 100% accurate model
    return tf.keras.models.load_model('effnet_lstm_best.keras')

model = load_model()

st.title("Fruit Classification Lab 🍎")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 2. PREPROCESSING (Match your 64x64 training)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)
    
    img = image.resize((64, 64))
    img_array = np.array(img)
    # No /255 here, the model handles it!
    img_array = np.expand_dims(img_array, axis=0)
    
    # 3. PREDICTION
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Convert raw output to percentages
    
    class_idx = np.argmax(score)
    result = class_names[class_idx]
    confidence = np.max(score) * 100
    
    # 4. OUTPUT
    st.subheader(f"Prediction: {result}")
    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")