
import pandas as pd
import os

def inspect_pickle_file(file_path):
    """Loads a pickle file and prints its info and head."""
    print(f"--- Inspecting File: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    try:
        data = pd.read_pickle(file_path)

        if isinstance(data, dict):
            # Handle the dictionary structure for ticker data
            for key, df in data.items():
                print(f"\n--- Data found for key: '{key}' ---")
                if isinstance(df, pd.DataFrame):
                    print(f"Shape: {df.shape}")
                    print("\n[INFO]")
                    df.info()
                    print("\n[HEAD]")
                    print(df.head())
                else:
                    print(f"Data for key '{key}' is not a DataFrame.")

        elif isinstance(data, pd.DataFrame):
            # Handle the DataFrame structure for macro data
            print(f"Shape: {data.shape}")
            print("\n[INFO]")
            data.info()
            print("\n[HEAD]")
            print(data.head())

        else:
            print(f"Unsupported data type in pickle file: {type(data)}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    print("-" * (len(file_path) + 20))


def main():
    """Main function to inspect the downloaded data files."""
    # --- Setup Paths ---
    # Get the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script directory
    data_dir = os.path.join(SCRIPT_DIR, 'data', 'temp_raw')

    tickers_file = os.path.join(data_dir, 'raw_tickers_data.pkl')
    macro_file = os.path.join(data_dir, 'raw_macro_data.pkl')

    inspect_pickle_file(tickers_file)
    print("\n\n")
    inspect_pickle_file(macro_file)


if __name__ == '__main__':
    main()
