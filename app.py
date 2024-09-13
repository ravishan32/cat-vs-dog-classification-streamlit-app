import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time  # For simulating progress

# Load the pre-trained model
model = tf.keras.models.load_model('dog_vs_cat_classifier.keras')

# Function to preprocess the uploaded image
def preprocess_image(image, img_size=(150, 150)):
    image = image.resize(img_size)  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    if image.shape[-1] == 4:  # Convert RGBA to RGB if necessary
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the uploaded image
def predict_image(image, model):
    image = preprocess_image(image)  # Preprocess the uploaded image
    prediction = model.predict(image)  # Make prediction
    return prediction

# Sidebar menu with dropdown
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select an option", ("Home", "About App"))

st.sidebar.write("""
    This is a streamlit web app that classifies images of cats and dogs. 
    You can upload an image and the model will predict whether it is a dog or a cat, and display the result with a confidence score in the form of a pie chart.
    """)

# Home option: Upload image and get prediction
if menu == "Home":
    st.title("Cat vs Dog Classifier")
    st.image("banner1.jpg")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image of a dog or a cat", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            # Show a progress bar
            with st.spinner("Loading..."):
                # Simulate a delay (remove this line in actual usage)
                time.sleep(2)
                
                # Make prediction
                prediction = predict_image(image, model)
                
                # Interpretation of the result
                if prediction[0] > 0.5:
                    label = "Dog"
                    confidence = prediction[0][0] * 100
                    cat_confidence = 100 - confidence
                else:
                    label = "Cat"
                    confidence = (1 - prediction[0][0]) * 100
                    cat_confidence = confidence
                    confidence = 100 - confidence
                
                st.write(f"Prediction Result: {label}")
                #st.write(f"Confidence: {confidence:.2f}%")
                
                # Pie chart to visualize the prediction
                st.write("Graphical Representation of Predicted Result:")
                labels = ['Dog', 'Cat']
                sizes = [confidence, cat_confidence]
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
                st.pyplot(fig)

# About App option
elif menu == "About App":
    st.title("About This App")
    st.write("""
    This is a simple app that classifies images of cats and dogs. 
    You can upload an image and the model will predict whether it is a dog or a cat, and display the result with a confidence score in the form of a pie chart.
    The model is a Convolutional Neural Network (CNN) trained on a kaggle dataset of dog and cat images.
    """)
