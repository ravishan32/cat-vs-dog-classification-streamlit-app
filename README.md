# Cat vs. Dog Classifier Web App

## Overview

This is a Streamlit web application that classifies images of cats and dogs. You can upload an image, and the model will predict whether it is a cat or a dog. The result is displayed along with a confidence score in the form of a pie chart.

## Features

- Upload an image of a cat or dog.
- Get a prediction along with a confidence score.
- Visualize the prediction with a pie chart.
- Learn more about the app from the 'About App' section in the sidebar.

## Technologies Used

- **Streamlit**: For creating the interactive web interface.
- **TensorFlow/Keras**: For the trained Convolutional Neural Network (CNN) model.
- **Pillow**: For image processing.
- **Matplotlib**: For creating visualizations.

## Installation

To run the app locally, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>

2. **Install the Required Packages**

    ```bash
    pip install streamlit tensorflow pillow matplotlib

3. **Run the App**

    ```bash
    streamlit run app.py

## Usage

- Home: Upload an image of a cat or dog, and click the "Predict" button to get the classification result and confidence score.
- About App: View information about the app and its functionality.

