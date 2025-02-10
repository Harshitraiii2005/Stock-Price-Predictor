import streamlit as st
import yfinance as yf
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load trained model
model_path = "best_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# App Title & UI Enhancements
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #3498db;'>Stock Price Prediction App 📈</h1>", unsafe_allow_html=True)
st.write("Get real-time stock insights and AI-driven predictions!")

# Stock selection dropdown
stock_options = [
    "AAPL", "ADANIPORTS.NS", "AMAT", "AMD", "AMZN", "ASIANPAINT.NS", "BABA", "BAJAJ-AUTO.NS", 
    "BAJAJFINSV.NS", "BA", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS", "CRM", "CSCO", "DIS", 
    "DIVISLAB.NS", "GE", "GM", "GOOG", "GRASIM.NS", "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", 
    "IBM", "ICICIBANK.NS", "INFY.NS", "INTC", "INTU", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", 
    "KO", "LT.NS", "M&M.NS", "MARUTI.NS", "MSFT", "MS", "NFLX", "NTPC.NS", "NVDA", "PEP", "PG", 
    "POWERGRID.NS", "PYPL", "RELIANCE.NS", "SBIN.NS", "SHOP", "SNAP", "SPY", "SUNPHARMA.NS", 
    "TATAMOTORS.NS", "TCS.NS", "TECHM.NS", "TSLA", "UPL.NS", "V", "WIPRO.NS", "ZM"
]

# Dropdown for stock selection
ticker = st.selectbox("Select Stock Ticker:", stock_options, index=0)

# Fetch stock data from Yahoo Finance
if st.button("Get Stock Data"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="6mo")  # Fetch last 6 months of data

        if data.empty:
            st.error("Invalid stock ticker or no data available!")
        else:
            # Display Latest Data
            st.subheader(f"Latest Stock Data for {ticker}")
            st.write(data.tail())
            
            # Line Chart for Stock Prices
            fig = px.line(data, x=data.index, y=['Open', 'Close'], title=f"Stock Price Trend of {ticker}")
            st.plotly_chart(fig)
            
            # Candlestick Chart
            fig_candle = go.Figure(data=[go.Candlestick(x=data.index,
                                                        open=data['Open'],
                                                        high=data['High'],
                                                        low=data['Low'],
                                                        close=data['Close'])])
            fig_candle.update_layout(title=f"Candlestick Chart for {ticker}", xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_candle)
            
            # Stock Volume Bar Chart
            fig_bar = px.bar(data, x=data.index, y='Volume', title=f"Trading Volume for {ticker}", color_discrete_sequence=['#f39c12'])
            st.plotly_chart(fig_bar)
            
            # Predict Next Day Close Price
            latest_data = data.iloc[-1][['Open', 'High', 'Low', 'Volume']].values.reshape(1, -1)
            prediction = model.predict(latest_data)[0]
            st.success(f"📈 **Predicted Next Close Price: ${prediction:.2f}**")
            
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

st.write("---")
st.write("🚀 **Developed by Harshit Rai**")
st.write("📡 Powered by Yahoo Finance & Advanced Machine Learning 🤖")
st.write("🔍 Stay ahead in the market with AI-driven predictions and in-depth analysis!")
