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

        # Assign 'p' (T-1 Close)
        symbol_data['p'] = symbol_data['Close']

        # Calculate 'p_exit' (T+1 Open)
        symbol_data['p_exit'] = symbol_data['Open'].shift(-1)

        # Add symbol back for MultiIndex
        symbol_data['symbol'] = symbol

        all_labels.append(symbol_data[['symbol', 'p', 'vol', 'p_exit']])

    # Concatenate all results and set the correct index
    labels_df = pd.concat(all_labels)
    labels_df = labels_df.set_index(['symbol', labels_df.index])

    print("Calculated T-1 inputs (p, vol) and T+1 exit price (p_exit).")
    return labels_df

def determine_fill_status_and_calculate_y(labels_df, data_60m):
    """
    Determines the fill status for day T and calculates the Y label.
    """
    if labels_df is None or data_60m is None:
        return None

    # --- Pre-computation for efficiency ---
    # Get the date part of the 60m timestamps
    data_60m['date'] = data_60m.index.get_level_values('timestamp').date
    # Find the minimum low for each symbol and day T
    min_low_t = data_60m.groupby(['symbol', 'date'])['Low'].min()

    # --- Row-by-row processing ---
    results = []
    for (symbol, t_minus_1_timestamp), row in labels_df.iterrows():
        p = row['p']
        p_exit = row['p_exit']
        vol = row['vol']

        # Determine day T by adding one day to T-1
        t_date = (t_minus_1_timestamp + pd.Timedelta(days=1)).date()

        fill_status = 'NO_FILL'
        y = 0.0

        try:
            # Check if the min low on day T was <= p
            if min_low_t.loc[(symbol, t_date)] <= p:
                fill_status = 'FILLED'
                # Calculate Y only if vol is not zero to avoid division by zero
                if vol is not None and vol > 0:
                    y = (p_exit - p) / vol
                else:
                    y = np.nan # Or some other indicator of an issue

        except KeyError:
            # This happens if there is no 60m data for day T for that symbol
            # The default of NO_FILL and Y=0 is appropriate here
            pass
        except Exception as e:
            # General exception for unexpected errors
            print(f"An error occurred for {symbol} at {t_minus_1_timestamp}: {e}")

        results.append({
            'asset': symbol,
            'T-1_timestamp': t_minus_1_timestamp,
            'Y': y,
            'Fill_Status': fill_status
        })

    final_labels = pd.DataFrame(results).set_index(['asset', 'T-1_timestamp'])
    print("Fill status and Y labels calculated.")
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
