import pandas as pd
import numpy as np
import os
import pandas_ta as ta

def load_data():
    """Loads the raw 60m and daily data from the data_acquisition directory."""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_60m_path = os.path.join(data_dir, 'raw_60m.parquet')
    data_daily_path = os.path.join(data_dir, 'raw_daily.parquet')

    try:
        data_60m = pd.read_parquet(data_60m_path)
        data_daily = pd.read_parquet(data_daily_path)
        print("Successfully loaded raw_60m.parquet and raw_daily.parquet.")
        return data_60m, data_daily
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have run '01_get_data.py' first.")
        return None, None

def calculate_label_inputs(data_daily):
    """
    Calculates T-1 inputs (p, vol) and T+1 exit price (p_exit) by iterating through each symbol.
    """
    if data_daily is None:
        return None

    # Ensure timestamp is a DatetimeIndex
    if not isinstance(data_daily.index.get_level_values('timestamp'), pd.DatetimeIndex):
        data_daily.index = data_daily.index.set_levels(pd.to_datetime(data_daily.index.get_level_values('timestamp')), level='timestamp')

    all_labels = []
    symbols = data_daily.index.get_level_values('symbol').unique()

    for symbol in symbols:
        symbol_data = data_daily.loc[symbol].copy()

        # Ensure correct data types
        for col in ['High', 'Low', 'Close', 'Open']:
            symbol_data[col] = pd.to_numeric(symbol_data[col], errors='coerce')

        # Drop rows with NaN values in the essential columns
        symbol_data.dropna(subset=['High', 'Low', 'Close'], inplace=True)

        # Calculate 'vol' (T-1 14-day ATR)
        symbol_data['vol'] = ta.atr(
            high=symbol_data['High'],
            low=symbol_data['Low'],
            close=symbol_data['Close'],
            length=14
        )

        # --- New logic (T Open entry, T+1 Open exit) ---
        symbol_data['p'] = symbol_data['Open'].shift(-1)      # p = T Open
        symbol_data['p_exit'] = symbol_data['Open'].shift(-2)  # p_exit = T+1 Open

        # Add symbol back for MultiIndex
        symbol_data['symbol'] = symbol

        all_labels.append(symbol_data[['symbol', 'p', 'vol', 'p_exit']])

    # Concatenate all results and set the correct index
    labels_df = pd.concat(all_labels)
    labels_df = labels_df.set_index(['symbol', labels_df.index])

    print("Calculated T-1 inputs (p, vol) and T+1 exit price (p_exit).")
    return labels_df

def determine_fill_status_and_calculate_y(labels_df, data_60m):
    # ... (data_60m is no longer used but kept for compatibility)

    results = []
    for (symbol, t_minus_1_timestamp), row in labels_df.iterrows():
        p = row['p']
        p_exit = row['p_exit']
        vol = row['vol']

        fill_status = 'FILLED' # Always filled for market order
        y = np.nan             # Default to NaN

        try:
            # Calculate Y only if all necessary data is valid and vol > 0
            if pd.notna(p) and pd.notna(p_exit) and pd.notna(vol) and vol > 0:
                y = (p_exit - p) / vol
        except Exception as e:
            print(f"An error occurred for {symbol} at {t_minus_1_timestamp}: {e}")
            pass

        results.append({
            'asset': symbol,
            'T-1_timestamp': t_minus_1_timestamp,
            'Y': y,
            'Fill_Status': fill_status
        })

    final_labels = pd.DataFrame(results).set_index(['asset', 'T-1_timestamp'])
    print("Fill status and Y labels calculated (Market Order simulation).")
    return final_labels

def build_labels():
    """
    Main function to build the labels.
    """
    data_60m, data_daily = load_data()
    if data_60m is None or data_daily is None:
        return

    # Calculate the primary inputs for the labels
    labels_df = calculate_label_inputs(data_daily)
    if labels_df is None:
        return

    # Determine fill status and calculate Y
    final_labels = determine_fill_status_and_calculate_y(labels_df, data_60m)
    if final_labels is None:
        return

    # --- Start of timestamp normalization ---
    # Get the current index levels
    asset_level = final_labels.index.get_level_values('asset')
    timestamp_level = final_labels.index.get_level_values('T-1_timestamp')

    # Normalize the timestamp level to .date()
    normalized_timestamp = pd.to_datetime(timestamp_level).date

    # Re-create the MultiIndex
    final_labels.index = pd.MultiIndex.from_arrays(
        [asset_level, normalized_timestamp],
        names=['asset', 'T-1_timestamp']
    )
    # --- End of timestamp normalization ---

    # Save the final labels to a parquet file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'labels_Y.parquet')
    final_labels.to_parquet(output_path)
    print(f"Successfully saved final labels to {output_path}")
    print(f"Final labels shape: {final_labels.shape}")

if __name__ == "__main__":
    build_labels()
