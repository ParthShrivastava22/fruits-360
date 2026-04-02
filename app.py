import streamlit as st
import tensorflow as tf
import keras  # Import base keras explicitly
import numpy as np
from PIL import Image

# 1. THE CLASS LIST (5 classes)
class_names = ['Apple', 'Avocado', 'Banana', 'Cherry', 'Orange']

# 2. CACHED MODEL LOADING
import streamlit as st
import keras
import numpy as np
from PIL import Image

@st.cache_resource
def load_my_model():
    return keras.models.load_model('effnet_lstm_best.keras')

# ... rest of your code ...

model = load_my_model()

# ... (Rest of your UI and prediction code from the previous block)

# 3. UI SETUP
st.set_page_config(page_title="Fruit Classifier", page_icon="🍓")
st.title("Fruit Classification AI 🍎🍌")
st.write("Upload a fruit photo for a real-time prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 4. IMAGE PREPROCESSING
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Resize to 64x64 to match training
    img = image.resize((64, 64))
    img_array = np.array(img)
    
    # Expand dims to create the 'batch' (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: No division by 255 here because your model has a Rescaling layer!

    if st.button('Classify Fruit'):
        with st.spinner('Analyzing...'):
            # 5. PREDICTION LOGIC
            predictions = model.predict(img_array)
            
            # Apply Softmax to get percentages
            probabilities = tf.nn.softmax(predictions[0]).numpy()
            score_index = np.argmax(probabilities)
            result = class_names[score_index]
            confidence = probabilities[score_index] * 100

        # 6. FINAL OUTPUT
        if confidence > 80:
            st.success(f"Prediction: **{result}** ({confidence:.2f}%)")
        else:
            st.warning(f"Likely: **{result}** ({confidence:.2f}%) - Lower confidence.")

        # Show a bar chart for your report
        with st.expander("View Probabilities Graph"):
            chart_data = dict(zip(class_names, probabilities))
            st.bar_chart(chart_data)