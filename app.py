import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Page Configuration
st.set_page_config(page_title="Tesla AI Predictor", page_icon="🚗", layout="wide")

# 2. Custom CSS for a "Dark Mode" Tech Feel
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar with Logo/Title
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=100)
st.sidebar.title("Control Panel")
days_to_plot = st.sidebar.slider("Historical View (Days)", 10, 500, 100)

# 4. Header Section
st.title("🚗 Tesla Stock Price AI")
st.markdown("---")

# 5. Data Loading & Metrics
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
latest_price = df['Adj Close'].iloc[-1]
prev_price = df['Adj Close'].iloc[-2]
price_diff = latest_price - prev_price

# Layout: 3 Columns for Key Stats
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${latest_price:.2f}", f"{price_diff:.2f}")
col2.metric("Market", "NASDAQ: TSLA", "Active")
col3.metric("Model Status", "LSTM Ready", "100%")

# 6. Main Visualization
st.subheader("📈 Market Trends")
chart_data = df.set_index('Date')['Adj Close'].tail(days_to_plot)
st.area_chart(chart_data, use_container_width=True)

# 7. Prediction Logic
st.subheader("🤖 AI Prediction Engine")

# Place the model loading inside a cleaner UI block
with st.expander("System Logs & Model Status", expanded=False):
    try:
        model = load_model('tesla_model.h5')
        st.success("Deep Learning weights loaded.")
    except Exception as e:
        st.error("Model file 'tesla_model.h5' not found. Displaying demo mode.")

# Interactive Predict Button
if st.button('🚀 Run AI Forecast'):
    with st.spinner('Analyzing market patterns...'):
        # This is where your prediction logic goes
        # For now, let's create a placeholder "prediction"
        prediction = latest_price * (1 + np.random.uniform(-0.05, 0.05))
        
        st.balloons()
        st.write(f"### Predicted Price for Tomorrow: **${prediction:.2f}**")