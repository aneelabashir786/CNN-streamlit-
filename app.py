
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import joblib  

# CONFIG
IMAGE_HEIGHT, IMAGE_WIDTH = 28, 28
MODEL_PATH = "cnn_model.keras"   
TEST_CSV_PATH = "test.csv"       
SCALER_PATH = "scaler.pkl"       

# LOAD MODEL & SCALER

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()


# LOAD TEST DATA

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_CSV_PATH)

test_df = load_test_data()


# STREAMLIT UI

st.title("CNN Prediction from Test CSV (Using Saved Scaler)")
st.write("This app picks a random row from test.csv, applies the exact scaler from training, and predicts its label.")

if st.button("Pick a Random Test Sample"):
    # Pick a random row
    random_idx = random.randint(0, len(test_df) - 1)
    sample = test_df.iloc[random_idx].values  

    # Show image
    st.write(f"**Random Test Index:** {random_idx}")
    fig, ax = plt.subplots()
    ax.imshow(sample.reshape(IMAGE_HEIGHT, IMAGE_WIDTH), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)

    # Scale using the saved scaler from Colab
    sample_scaled = scaler.transform(sample.reshape(1, -1))
    sample_scaled = sample_scaled.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)

    # Predict
    prediction = model.predict(sample_scaled)
    predicted_label = np.argmax(prediction)

    st.write(f"### Prediction: {predicted_label}")
    st.bar_chart(prediction[0])

