import yfinance as yf
import pandas as pd
import json
import warnings

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_data():
    """
    Downloads and saves yfinance data for the asset pool.
    """
    # Load the asset pool
    with open("V4-D.8(v7.0)/asset_pool.json", "r") as f:
        asset_pool = json.load(f)

    # Clean the ticker symbols
    tickers = [asset.split(":")[1].replace('.', '-') for asset in asset_pool]

    # Download 60m data
    data_60m = yf.download(
        tickers,
        period="2y",
        interval="60m",
        auto_adjust=False,
        prepost=True,  # Include pre- and post-market data
        threads=True,
    )

    # Download daily data
    data_daily = yf.download(
        tickers,
        period="10y",
        interval="1d",
        auto_adjust=False,
        threads=True,
    )

    # Format and save 60m data
    if not data_60m.empty:
        data_60m.index.name = "timestamp"
        data_60m = data_60m.stack(future_stack=True)
        data_60m.index.names = ["timestamp", "symbol"]
        data_60m = data_60m.reorder_levels(["symbol", "timestamp"])
        data_60m = data_60m[['Open', 'High', 'Low', 'Close', 'Volume']]
        data_60m.to_parquet("V4-D.8(v7.0)/raw_60m.parquet")

    # Format and save daily data
    if not data_daily.empty:
        data_daily.index.name = "timestamp"
        data_daily = data_daily.stack(future_stack=True)
        data_daily.index.names = ["timestamp", "symbol"]
        data_daily = data_daily.reorder_levels(["symbol", "timestamp"])
        data_daily = data_daily[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        data_daily.to_parquet("V4-D.8(v7.0)/raw_daily.parquet")


if __name__ == "__main__":
    get_data()
