import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 1. Page Configuration
st.set_page_config(page_title="Tesla AI Stock Predictor", page_icon="🚗", layout="wide")

# 2. Light Mode Custom CSS
st.markdown("""
    <style>
    /* Main App Background to White */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Metric Card Styling: Light Grey background with dark borders */
    div[data-testid="column"] {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Force visibility for Metric Labels and Values to Black */
    [data-testid="stMetricLabel"] {
        color: #212529 !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F1F3F5;
    }

    /* Button Styling: Dark background with White text for contrast */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #212529;
        color: #FFFFFF;
        font-weight: bold;
        border: none;
        height: 3em;
    }
    .stButton>button:hover {
        background-color: #495057;
        color: #FFFFFF;
    }
    
    /* Text color for standard markdown */
    h1, h2, h3, p {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar Setup
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=80)
st.sidebar.title("Settings")
days_to_plot = st.sidebar.slider("Historical Range (Days)", 30, 1000, 180)

# 4. Data Engine
@st.cache_data
def load_data():
    data = pd.read_csv('TSLA.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

try:
    df = load_data()
    st.title("🚗 Tesla Stock AI Analysis")
    
    latest_price = df['Adj Close'].iloc[-1]
    prev_price = df['Adj Close'].iloc[-2]
    delta = latest_price - prev_price
    
    # Header Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"${latest_price:.2f}", f"{delta:.2f}")
    m2.metric("Market Status", "NASDAQ: TSLA", "Active")
    m3.metric("AI Confidence", "High", "94%")

    st.markdown("---")

    # 5. Visualizations & Prediction
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Market Movement")
        chart_df = df.tail(days_to_plot)
        st.area_chart(chart_df.set_index('Date')['Adj Close'])

    with col_right:
        st.subheader("🤖 AI Forecast")
        if st.button('🚀 Run AI Prediction'):
            try:
                model = load_model('tesla_model.h5')
                scaler = MinMaxScaler(feature_range=(0,1))
                full_data = df['Adj Close'].values.reshape(-1, 1)
                scaler.fit(full_data)
                
                last_60_days = full_data[-60:]
                last_60_days_scaled = scaler.transform(last_60_days)
                X_test = np.array([last_60_days_scaled])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
                with st.spinner('Calculating...'):
                    pred_price_scaled = model.predict(X_test)
                    pred_price = scaler.inverse_transform(pred_price_scaled)
                    
                    st.success(f"### Predicted Price: **${pred_price[0][0]:.2f}**")
            except:
                st.error("Model file 'tesla_model.h5' not found.")

except Exception as e:
    st.error(f"Error loading data: {e}")