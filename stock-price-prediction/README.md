# Predicting Stock Prices with a Long-Short Term Memory (LSTM) Predictive Model

> [!WARNING]
> THIS IS NOT FINANCIAL ADVICE. THIS MODEL WAS BUILT FOR EDUCATIONAL REASONS.

This project explores predictive modelling in finance using a recurrent neural network architecture known as Long Short-Term Memory.

This Streamlit app explains the LSTM model architecture, and users can interact with the model by entering their four-letter company identifier and letting the model predict future prices. The application also includes a breakdown of the PyTorch code used and model evaluation.

Data used to train and test this model is stock market data from the Yahoo Finance (`yfinance`) API. This API includes information on Open, High, Low, Close prices and Volume traded (definitions for each feature are noted in the Streamlit app).

## Running Streamlit app

1. Download Streamlit in your virtual environment.

    ```zsh
    uv add streamlit
    ```

    ```zsh
    pip install streamlit
    ```

2. Run the Streamlit app

    ```zsh
    streamlit run main.py
    ```
