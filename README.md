📌 Repository Description (README intro)

Stock Market Predictor
This project is a machine learning–based stock market prediction tool that analyzes historical price data from 2012 up to the most recent available trading day (1–2 days before the run).

📊 Data Source: Yahoo Finance (yfinance)

🛠 Features: Moving averages (MA5, MA10, MA20), returns, volatility, momentum, volume changes

🤖 Model: Random Forest (time-series aware training)

🔮 Prediction Goal: Next-day market movement (Up/Down) or next-day return

📈 Evaluation: TimeSeriesSplit cross-validation, accuracy reporting, feature importance


The tool automatically:

1. Fetches historical OHLCV data starting from 2012.


2. Builds technical features and target labels.


3. Trains a Random Forest model with walk-forward validation.


4. Saves the trained model for reuse.


5. Predicts the next trading day’s direction or return using the latest available features.



⚠ Disclaimer: This project is for educational and research purposes only. Stock market prediction is inherently uncertain. This is not financial advice.
