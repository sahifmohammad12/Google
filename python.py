import requests
import json
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st

# Initialize FastAPI
app = FastAPI()
scaler = MinMaxScaler(feature_range=(0,1))

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")["Close"]
    return df.values.reshape(-1, 1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Train LSTM model
def train_lstm_model(ticker):
    data = fetch_stock_data(ticker)
    data = scaler.fit_transform(data)
    
    X_train, y_train = [], []
    for i in range(60, len(data)):
        X_train.append(data[i-60:i, 0])
        y_train.append(data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = torch.tensor(X_train).float().view(-1, 60, 1)
    y_train = torch.tensor(y_train).float().view(-1, 1)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "lstm_model.pth")
    return model

# API Endpoint for Stock Prediction
class StockRequest(BaseModel):
    ticker: str

@app.post("/predict")
def predict_stock(data: StockRequest):
    try:
        model = LSTMModel()
        model.load_state_dict(torch.load("lstm_model.pth"))
        model.eval()
        
        df = fetch_stock_data(data.ticker)
        last_60_days = df[-60:].reshape(-1, 1)
        last_60_days = scaler.transform(last_60_days)
        X_test = torch.tensor(last_60_days).float().view(1, 60, 1)
        
        prediction = model(X_test).item()
        prediction = scaler.inverse_transform([[prediction]])[0][0]
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to communicate with OpenAI API
def get_ai_response(query):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer YOUR_OPENAI_API_KEY", "Content-Type": "application/json"}
    request_body = {"model": "gpt-4", "messages": [{"role": "user", "content": query}]}
    response = requests.post(url, headers=headers, data=json.dumps(request_body))
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

# API Endpoint for AI Chatbot
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chatbot_response(data: ChatRequest):
    try:
        response = get_ai_response(data.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit Frontend
st.title("GenAI Financial Assistant")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)")
if st.button("Predict Stock Price"):
    response = requests.post("http://localhost:8000/predict", json={"ticker": ticker})
    if response.status_code == 200:
        st.success(f"Predicted Price: ${response.json()['prediction']:.2f}")
        df = yf.download(ticker, period="1y", interval="1d")
        st.line_chart(df["Close"])
    else:
        st.error("Error fetching prediction")

query = st.text_area("Ask the Financial AI Chatbot")
if st.button("Get AI Response"):
    response = requests.post("http://localhost:8000/chat", json={"query": query})
    if response.status_code == 200:
        st.write(response.json()['response'])
    else:
        st.error("Error fetching chatbot response")
