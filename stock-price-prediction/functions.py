import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
import pandas as pd

import yfinance as yf # Data

import torch
import torch.nn as nn
import torch.optim as optim

# Identify which processing unit is being used: CPU or GPU (mps for Apple, cuda for NVDIA)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def get_data(ticker, date):
    data = yf.download(ticker, f'{date}')

    # Check if there are MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values("Price")   # Flatten the "Price" header

    # Reset index to make `Date` a column and not an index
    data = data.reset_index().sort_values(by="Date")

    # Remove the `Price` column header label
    data.columns.name = None

    return data 

def yf_figure(data, var, date):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(data['Date'], data[f'{var}'])

    plt.xticks(rotation=45)
    ax.set(
        xlabel = "Date",
        ylabel = f"{var}",
        title=f'Closing prices between from {date}'
    )

    return fig


class PredictionModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(PredictionModel, self).__init__()

		# Number of LSTM layers
		self.num_layers = num_layers	

		# Number of memory neurones per layer
		self.hidden_dim = hidden_dim

		# LSTM layer
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

		# Fully Connected layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	# Forward pass
	def forward(self, x):
		# Blank slate initialisation
		init_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)	# Initial short-term memory
		init_cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)	# Initial long-term memory (unique to LSTMs)

		# The LSTM takes the data batch (x) and the blank memories and cycles through the (29-day) sequence, updating its internal states at each time step
		hidden_sequences, (final_hidden_snapshot, final_cell_snapshot) = self.lstm(x, (init_hidden_state, init_cell_state))

		# Extract the final hidden state (or summary of all prior hidden states) from the very last time-step in the sequence
		final_hidden_stat = hidden_sequences[:, -1, :]

		# Condense this final hidden state with 64 hidden units into a single price prediction
		prediction = self.fc(final_hidden_stat)

		return prediction

def evaluation_figure(ticker, data, y_test, y_test_pred, train_rmse, test_rmse):
    y_test_date = data.iloc[-len(y_test): , 0]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(4, 1)

    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(y_test_date, y_test_pred, color='red', label='Predicted')
    ax1.plot(y_test_date, y_test, color='blue', label='Actual')
    ax1.set(
        # xlabel = "Date",
        ylabel = "Price"
    )
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.axhline(test_rmse, color='green', linestyle='--', label='Test RMSE')
    ax2.axhline(train_rmse, color='black', linestyle='--', label='Train RMSE')
    ax2.plot(y_test_date, abs(y_test - y_test_pred), color='red', label='Prediction error')
    ax2.set(
        xlabel = "Date",
        ylabel = "Error"
    )
    ax2.legend()

    plt.suptitle(f"{ticker} Sock Price Prediction")
    plt.tight_layout()

    return fig