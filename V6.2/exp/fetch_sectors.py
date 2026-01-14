
import json
import os
import yfinance as yf
import sys

# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
SECTOR_FILE = os.path.join(RESOURCE_DIR, 'ticker_sectors.json')

sys.path.append(BASE_DIR)
try:
    from production_daily_plan_v6_2 import clean_ticker
except ImportError:
    def clean_ticker(t): return t.split(':')[-1]

def fetch_sectors():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    with open(path, 'r') as f:
        tickers = json.load(f)

    cleaned_tickers = [clean_ticker(t) for t in tickers]

    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, 'r') as f:
            sector_map = json.load(f)
    else:
        sector_map = {}

    print(f"Fetching sectors for {len(cleaned_tickers)} tickers...")

    changed = False
    for t in cleaned_tickers:
        if t not in sector_map:
            try:
                print(f"Fetching info for {t}...")
                tick = yf.Ticker(t)
                # yfinance info can be slow.
                info = tick.info
                sector = info.get('sector', 'Unknown')
                sector_map[t] = sector
                changed = True
            except Exception as e:
                print(f"Failed to fetch {t}: {e}")
                sector_map[t] = 'Unknown'
                changed = True

    if changed:
        with open(SECTOR_FILE, 'w') as f:
            json.dump(sector_map, f, indent=2)
        print(f"Saved sectors to {SECTOR_FILE}")
    else:
        print("No new sectors to fetch.")

if __name__ == "__main__":
    fetch_sectors()
