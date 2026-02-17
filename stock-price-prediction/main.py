import streamlit as st 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.optim as optim

from lstm_model import get_data, yf_figure, PredictionModel, evaluation_figure


st.title("📈 Predicting Stock Prices with a Long-Short Term Memory (LSTM) Predictive Model")

st.header("Long Short-Term Memory (LSTM) Architecture")


device_choice = st.selectbox(
    "What is your available GPU?",
    ("MPS (Apple)", "CUDA (Nvdia)", "CPU"),
)

gc.collect()
if device_choice == "MPS (Apple)":
    device = torch.device('mps')
    torch.mps.empty_cache()
elif device_choice == "CUDA (Nvdia)":
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")


# User input for ticker
ticker = st.text_input("Company Ticker (e.g., AAPL for Apple)")

# Stop the script if ticker is empty
if not ticker:
    st.info("Please enter a stock ticker to begin.")
    st.stop()  # This prevents any code below this line from running

# User input for date
years = st.number_input("Years of data")
if years < 1:
    st.info("Please enter a number of years of at least one.")
    st.stop()
else:
    latest_date = datetime.now() - timedelta(days=(365 * years))
    date = latest_date.date()

# Data preview
st.subheader("Data Preview")
data = get_data(ticker, str(date))
st.dataframe(data.head())

# Figure of closing prices
st.subheader("Closing Prices")
close_figure = yf_figure(data, 'Close', date)
st.pyplot(close_figure)

## Load training and testing data 
# Looking at first 29 days to predict the 30th day
days = 29
X_close = []
y_close = []

for i in range(data.shape[0] - days):
    X_close.append(data.loc[i:(i+days-1), 'Close'])
    y_close.append([data.loc[(i+days), 'Close'].item()])

train_size = int(data.shape[0] * 0.75)

X_train_np = np.array(X_close[ :train_size])
y_train_np = np.array(y_close[ :train_size])

X_test_np = np.array(X_close[train_size: ])
y_test_np = np.array(y_close[train_size: ])

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = X_scaler.fit_transform(X_train_np)
X_test_scaled = X_scaler.transform(X_test_np)

y_train_scaled = y_scaler.fit_transform(y_train_np)
y_test_scaled = y_scaler.transform(y_test_np)

X_train_tensor = torch.from_numpy(X_train_scaled).type(torch.Tensor).to(device)
y_train_tensor = torch.from_numpy(y_train_scaled).type(torch.Tensor).to(device)

X_test_tensor = torch.from_numpy(X_test_scaled).type(torch.Tensor).to(device)
y_test_tensor = torch.from_numpy(y_test_scaled).type(torch.Tensor).to(device)

# X data needs to be a 3D tensor: (Batch Size: How many samples?, Sequence Length: How many days are you looking back? (The "Time Steps"), Input Dim: How many features per day?)
if X_train_tensor.dim() == 2:
    # unsqueeze(-1) adds a new dimension of size one at the specified position in a tensor
    X_train_tensor = X_train_tensor.unsqueeze(-1)

if X_test_tensor.dim() == 2:
    X_test_tensor = X_test_tensor.unsqueeze(-1)

model = PredictionModel(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 501

for i in range(num_epochs):
    model.train()

    # Clear the "memory" of the gradients from the last round
    optimiser.zero_grad()

    # Make a prediction (forward pass)
    y_train_pred = model(X_train_tensor)

    # Calculate loss of that prediction
    loss = criterion(y_train_pred, y_train_tensor)

    # Backpropagation - Calculate the new gradient by identifying which weights need to change and by how much
    loss.backward()

    # Optimisation - Update the weights moving the gradient in a better direction
    optimiser.step()        

    # Every 25 epochs, print the loss to check progress
    if i % 100 == 0:
        st.write(f"Epoch {i} | Loss : {loss.item():.6f}")


# Using the trained model on test data
model.eval()

with torch.no_grad():
    y_test_pred = model(X_test_tensor)

    # Convert scaled data back to it original scale 
    y_test_pred = y_scaler.inverse_transform(y_test_pred.detach().cpu().numpy()).reshape(-1)
    y_test = y_scaler.inverse_transform(y_test_tensor.detach().cpu().numpy()).reshape(-1)

    y_train_pred = y_scaler.inverse_transform(y_train_pred.detach().cpu().numpy()).reshape(-1)
    y_train = y_scaler.inverse_transform(y_train_tensor.detach().cpu().numpy()).reshape(-1)

    # Evaluating the model
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

evaluation_fig = evaluation_figure(ticker, data, y_test, y_test_pred, train_rmse, test_rmse)

st.pyplot(evaluation_fig)