import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("🚗 Tesla Stock Price Prediction")

# Load data
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar
st.sidebar.header("Options")
days_to_plot = st.sidebar.slider("Days to show in chart", 10, 500, 100)

# Show Chart
st.subheader("Historical Stock Price")
st.line_chart(df.set_index('Date')['Adj Close'].tail(days_to_plot))

# Load the model we trained in the notebook
try:
    model = load_model('tesla_model.h5')
    st.success("Model loaded successfully!")
except:
    st.error("Please train the model in the notebook first to generate 'tesla_model.h5'")

st.info("This app uses a Deep Learning LSTM model to predict trends.")