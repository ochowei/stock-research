"""
Daily Gap Signal Generator V6.2.6 RC
------------------------------------
Integrates:
1. Robust Scanner Framework (from v6.2.1)
2. Advanced Model Logic (from v6.2.5 RC)
3. Three-Pillar Architecture (Tech, Non-Tech, Crypto) - EXP-19
4. New Gap Logic (No caps, Volume Caution) - EXP-24, EXP-25
5. Tiered Position Sizing - EXP-18

Usage: python daily_gap_signal_generator_v6_2_6_rc.py
"""

import os
import sys
import json
import time
import logging
import joblib
import warnings
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from pandas.tseries.offsets import BDay

# [V6.2 Integration] Import Regime Logic
try:
    from production_daily_plan_v6_2 import get_regime_decision, clean_ticker
except ImportError:
    # Fallback if file not found in current dir
    def clean_ticker(ticker):
        if "BRK.B" in ticker: return "BRK-B"
        if ":" in ticker: return ticker.split(":")[-1]
        return ticker

    def calculate_er(series, window=5):
        if len(series) < window + 1: return 0.0
        subset = series.tail(window + 1)
        net_change = abs(subset.iloc[-1] - subset.iloc[0])
        sum_abs_change = subset.diff().abs().sum()
        if sum_abs_change == 0: return 0.0
        return net_change / sum_abs_change

    def get_regime_decision(df_daily, ticker, window=5, threshold=0.6):
        try:
            if df_daily is None or df_daily.empty or len(df_daily) < window + 1:
                return "UNKNOWN", 0.0, "Insufficient Data"
            er_value = calculate_er(df_daily['Close'], window=window)
            is_trend_regime = er_value > threshold
            status = "🛑 BLOCK" if is_trend_regime else "✅ PASS"
            action = "SKIP (Too Trendy)" if is_trend_regime else "Trade Gaps"
            return status, er_value, action
        except Exception as e:
            return "ERROR", 0.0, f"Error: {str(e)}"

# --- Configuration ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for resource dir in standard V6.2 structure relative to this script
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'resource'))
MODEL_DIR = os.path.join(BASE_DIR, 'Sell_Model_Lab', '03_Experiments', 'EXP_18_Production_Script_Update', '03_Output')

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
GAP_THRESHOLD = 0.005
RIP_THRESHOLD = 0.03
DIP_THRESHOLD = 0.03
MOMENTUM_THRESHOLD = 0.53
DIP_CONFIDENCE_LV = 0.50

CRYPTO_TICKERS = ['COIN', 'MSTR', 'RIOT', 'MARA']
US_HOLIDAYS = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
]

# Feature definitions
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']
CRYPTO_FEATURES = ['BTC_Gap_Pct', 'BTC_RSI_14', 'BTC_Dist_MA20', 'Crypto_Corr']

# Embed Sector Map Fallback
SECTOR_MAP_FALLBACK = {
    "ABAT": "Industrials", "ABNB": "Consumer Cyclical", "ACHR": "Industrials", "ADBE": "Technology",
    "AMD": "Technology", "AMZN": "Consumer Cyclical", "ANET": "Technology", "APP": "Communication Services",
    "BE": "Industrials", "BRK-B": "Financial Services", "BROS": "Consumer Cyclical", "CCJ": "Energy",
    "CEG": "Utilities", "CELH": "Consumer Defensive", "CIFR": "Financial Services", "CRWD": "Technology",
    "DDOG": "Technology", "DOCN": "Technology", "DUOL": "Technology", "EOSE": "Industrials",
    "FUBO": "Communication Services", "GLW": "Technology", "GOOG": "Communication Services", "GRAB": "Technology",
    "HIMS": "Healthcare", "HOOD": "Financial Services", "IBM": "Technology", "IONQ": "Technology",
    "IREN": "Financial Services", "KTOS": "Industrials", "LEU": "Energy", "LRCX": "Technology",
    "LTBR": "Industrials", "LUMN": "Communication Services", "MCD": "Consumer Cyclical", "MP": "Basic Materials",
    "MRVL": "Technology", "MSFT": "Technology", "MU": "Technology", "NET": "Technology", "NVDA": "Technology",
    "NVO": "Healthcare", "NVTS": "Technology", "OKLO": "Utilities", "ON": "Technology", "ONDS": "Technology",
    "OPEN": "Real Estate", "ORCL": "Technology", "OSCR": "Healthcare", "PANW": "Technology", "PLTR": "Technology",
    "POWI": "Technology", "QBTS": "Technology", "QCOM": "Technology", "RXRX": "Healthcare", "SHOP": "Technology",
    "SMR": "Industrials", "SNOW": "Technology", "SOFI": "Financial Services", "SOUN": "Technology",
    "SPOT": "Communication Services", "STX": "Technology", "TMDX": "Healthcare", "TSLA": "Consumer Cyclical",
    "TSM": "Technology", "TTD": "Communication Services", "U": "Technology", "UAMY": "Basic Materials",
    "UBER": "Technology", "UNH": "Healthcare", "UPST": "Financial Services", "UUUU": "Energy",
    "V": "Financial Services", "VST": "Utilities", "WWR": "Basic Materials", "COIN": "Financial Services",
    "MSTR": "Technology", "RIOT": "Financial Services", "MARA": "Technology"
}

# --- Helper Functions ---

def get_model_path(filename):
    """Search for model file in likely locations."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(OUTPUT_DIR, filename),
        os.path.join(BASE_DIR, 'Sell_Model_Lab', '03_Experiments', 'EXP_18_Production_Script_Update', '03_Output', filename),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return None

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        candidates = [
            os.path.join(BASE_DIR, '..', 'resource', '2025_final_asset_pool.json'),
            os.path.join(BASE_DIR, 'resource', '2025_final_asset_pool.json')
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            tickers = raw if isinstance(raw, list) else list(raw.keys())
            return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in tickers])))
    else:
        print(f"[Warning] Asset pool not found. Using default list.")
        return sorted(['NVDA', 'TSLA', 'AMD', 'COIN', 'MSTR', 'PLTR', 'SOFI'])

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1]) if not df.empty else 20.0
    except: return 20.0

def get_calendar_status():
    try:
        today = datetime.now().date()
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5: next_day += timedelta(days=1)
        if next_day.strftime('%Y-%m-%d') in US_HOLIDAYS: return "Pre-Holiday (Bullish) 🏖️", True

        current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(0)
        last_trading_day = current_month_end.date()
        next_month_start = current_month_end + BDay(1)
        first_3_days = [(next_month_start + BDay(i)).date() for i in range(3)]

        if today == last_trading_day: return "TOTM (Month End) 🚀", True
        elif today in first_3_days: return "TOTM (Month Start) 🚀", True
        return "Normal (一般日)", False
    except: return "Unknown", False

def get_position_size(prob):
    if prob > 0.60: return "1.5x"
    if prob > 0.55: return "1.0x"
    return "0.5x"

def get_sector_type(ticker, sector_map):
    if ticker in CRYPTO_TICKERS: return 'Crypto'
    sector = sector_map.get(ticker, 'Unknown')
    if sector == 'Technology': return 'Tech'
    return 'Non-Tech'

# --- Feature Engineering ---

def prepare_benchmark(bm_df, prefix):
    df = bm_df.copy()
    if len(df) < 20: return df # Not enough data for indicators

    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    rsi = ta.rsi(df['Close'], length=14)
    df[f'{prefix}_RSI_14'] = rsi.shift(1) if rsi is not None else np.nan

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def build_features_latest(df, bm_df, prefix, bm_features, corr_name_override=None):
    if len(df) < 20: return None
    df = df.sort_index().copy()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    last_idx = df.index[-1]
    row = df.iloc[[-1]].copy()

    # BM Features Sync
    if last_idx not in bm_df.index:
        idx_loc = bm_df.index.get_indexer([last_idx], method='pad')[0]
        if idx_loc == -1: return None
        bm_row = bm_df.iloc[[idx_loc]]
    else:
        bm_row = bm_df.loc[[last_idx]]

    for f in bm_features:
        if 'Corr' not in f:
            if f in bm_row.columns: row[f] = bm_row[f].values[0]
            else: row[f] = 0.0

    # Correlation
    common_idx = df.index.intersection(bm_df.index)
    if len(common_idx) < 20: return None
    df_sub = df.loc[common_idx]
    bm_sub = bm_df.loc[common_idx]

    aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
    corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1]).shift(1)

    corr_name = corr_name_override
    if not corr_name:
        if prefix == 'QQQ': corr_name = 'Sector_Corr'
        elif prefix == 'BTC': corr_name = 'Crypto_Corr'
        else: corr_name = 'Market_Corr'

    if np.isnan(corr.iloc[-1]): return None
    row[corr_name] = corr.iloc[-1]
    return row

# --- Main Execution ---

def main():
    print(f"\n>>> V6.2.6 RC Daily Gap Signal Generator (Three-Pillar Architecture)")

    cal_status, is_bullish_cal = get_calendar_status()
    curr_vix = get_current_vix()
    print(f">>> [Market Context] 📅 Calendar: {cal_status} | VIX: {curr_vix:.1f}")

    tickers = load_tickers()

    # Load Sector Map
    sector_map_path = os.path.join(RESOURCE_DIR, '..', 'exp', 'Sell_Model_Lab', '03_Experiments', 'EXP_18_Production_Script_Update', '03_Output', 'sector_map.json')
    if not os.path.exists(sector_map_path): sector_map_path = os.path.join(BASE_DIR, 'sector_map.json')

    sector_map = SECTOR_MAP_FALLBACK.copy()
    if os.path.exists(sector_map_path):
        with open(sector_map_path, 'r') as f:
            sector_map.update(json.load(f))

    # Load Models
    print(">>> Loading Models...")
    non_tech_path = get_model_path('v6.2.4_rc_non_tech_model.joblib')
    tech_path = get_model_path('v6.2.4_rc_tech_model.joblib')
    crypto_path = get_model_path('crypto_model.joblib')
    mom_path = get_model_path('momentum_model.joblib')
    dip_path = get_model_path('dip_model.joblib')

    models = {}
    if non_tech_path: models['Non-Tech'] = joblib.load(non_tech_path)
    if tech_path: models['Tech'] = joblib.load(tech_path)
    if mom_path: models['Mom'] = joblib.load(mom_path)
    if dip_path: models['Dip'] = joblib.load(dip_path)

    crypto_model = None
    if crypto_path:
        print("    [Info] Crypto Model found.")
        crypto_model = joblib.load(crypto_path)
        models['Crypto'] = crypto_model
    else:
        print("    [Warning] Crypto Model not found. Will downgrade Crypto tickers to Non-Tech Model (SPY Context).")

    if not models.get('Non-Tech') or not models.get('Tech'):
        print("[Error] Critical models missing (Tech/Non-Tech). Exiting.")
        return

    # Data Download
    benchmarks = ['QQQ', 'SPY', 'BTC-USD']
    all_tickers = list(set(tickers + benchmarks))
    print(f">>> Fetching Market Data for {len(all_tickers)} symbols...")

    try:
        data = yf.download(all_tickers, period='3mo', interval='1d', auto_adjust=True, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            try: data = data.stack(level=1, future_stack=True)
            except TypeError: data = data.stack(level=1)
            data = data.rename_axis(['Date', 'Ticker']).reset_index()
        else:
            if 'Ticker' not in data.columns: data['Ticker'] = all_tickers[0]
            data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()
    except Exception as e:
        print(f"[Error] Data download failed: {e}")
        return

    qqq_df = data[data['Ticker'] == 'QQQ'].dropna(subset=['Close']).set_index('Date').sort_index()
    spy_df = data[data['Ticker'] == 'SPY'].dropna(subset=['Close']).set_index('Date').sort_index()
    btc_df = data[data['Ticker'] == 'BTC-USD'].dropna(subset=['Close']).set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(benchmarks)].dropna(subset=['Close'])

    # Prepare Benchmarks
    qqq_prep = prepare_benchmark(qqq_df, 'QQQ')
    spy_prep = prepare_benchmark(spy_df, 'SPY')
    btc_prep = prepare_benchmark(btc_df, 'BTC')

    # Dual Date Logic
    latest_equity = spy_df.index.max()
    latest_crypto = btc_df.index.max()
    print(f">>> Analysis Dates -> Equity: {latest_equity.date()} | Crypto: {latest_crypto.date()}")

    results = []
    processed_count = 0

    for ticker, group in stock_df.groupby('Ticker'):
        sector_type = get_sector_type(ticker, sector_map)

        # Date Sync
        target_date = latest_crypto if sector_type == 'Crypto' else latest_equity
        if group['Date'].max() != target_date:
            print(f"Date Mismatch {ticker}: {group['Date'].max().date()} vs {target_date.date()}")
            continue

        processed_count += 1
        regime_status, er_val, _ = get_regime_decision(group, ticker)

        try:
            feat_row, prob, model_used = None, 0.0, "-"

            if sector_type == 'Tech':
                feat_row = build_features_latest(group.set_index('Date'), qqq_prep, 'QQQ', TECH_FEATURES)
                if feat_row is not None:
                    prob = models['Tech'].predict_proba(feat_row[BASE_FEATURES + TECH_FEATURES])[0][1]
                    model_used = "Tech"
            elif sector_type == 'Crypto':
                if crypto_model:
                    feat_row = build_features_latest(group.set_index('Date'), btc_prep, 'BTC', CRYPTO_FEATURES)
                    if feat_row is not None:
                        prob = crypto_model.predict_proba(feat_row[BASE_FEATURES + CRYPTO_FEATURES])[0][1]
                        model_used = "Crypto"
                else:
                    feat_row = build_features_latest(group.set_index('Date'), spy_prep, 'SPY', NON_TECH_FEATURES, corr_name_override='Market_Corr')
                    if feat_row is not None:
                        prob = models['Non-Tech'].predict_proba(feat_row[BASE_FEATURES + NON_TECH_FEATURES])[0][1]
                        model_used = "Non-Tech(F)"
            else:
                feat_row = build_features_latest(group.set_index('Date'), spy_prep, 'SPY', NON_TECH_FEATURES)
                if feat_row is not None:
                    prob = models['Non-Tech'].predict_proba(feat_row[BASE_FEATURES + NON_TECH_FEATURES])[0][1]
                    model_used = "Non-Tech"

            if feat_row is None:
                # print(f"Feat Row None for {ticker}")
                continue

            gap_pct = feat_row['Gap_Pct'].iloc[0]
            price = group.iloc[-1]['Close']
            vol_ratio = feat_row['Vol_Ratio'].iloc[0]
            atr_pct = feat_row['ATR_Pct'].iloc[0]
            dist_ma20 = feat_row['Dist_MA20'].iloc[0]
            rsi_14 = feat_row['RSI_14'].iloc[0]

            # Aux Models Predictions
            mom_prob, dip_prob = 0.0, 0.0
            try:
                # Legacy Features: ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20']
                X_legacy = pd.DataFrame([[rsi_14, atr_pct, vol_ratio, gap_pct, curr_vix, dist_ma20]],
                                      columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20'])
                if 'Mom' in models: mom_prob = models['Mom'].predict_proba(X_legacy)[0][1]
                if 'Dip' in models: dip_prob = models['Dip'].predict_proba(X_legacy)[0][1]
            except Exception: pass

            # --- Decision Logic (V6.2.1 + V6.2.6 Hybrid) ---
            status, action, size = "Flat", "FLAT", "-"

            if regime_status == "🛑 BLOCK":
                status, action = "🛑 BLOCK", "SKIP (Trend)"
            else:
                # 1. GAP UP Logic
                if gap_pct > GAP_THRESHOLD:
                    is_rip = gap_pct > RIP_THRESHOLD
                    is_mom = mom_prob > MOMENTUM_THRESHOLD

                    if is_rip:
                        if is_mom:
                            status, action = "🚀 ROCKET", "WATCH (Mom)"
                        else:
                            status = "🔴 SELL RIP"
                            if prob > 0.50:
                                action = "SELL -> MOC"
                                size = get_position_size(prob)
                                if vol_ratio > 3.0: action += " (Vol Caution)"
                            else:
                                action = "WATCH (Low Prob)"
                    else: # Normal Gap Up
                        if is_mom:
                            status, action = "🟢 MOMENTUM", "WATCH (Mom)"
                        else:
                            status = "🔴 GAP UP"
                            if prob > 0.50:
                                action = "SELL -> MOC"
                                size = get_position_size(prob)
                                if vol_ratio > 3.0: action += " (Vol Caution)"
                            else:
                                action = "WATCH (Low Prob)"

                # 2. GAP DOWN Logic (Dip Buy)
                elif gap_pct < -GAP_THRESHOLD:
                    is_deep_dip = gap_pct < -DIP_THRESHOLD
                    if is_deep_dip:
                        if dip_prob > DIP_CONFIDENCE_LV:
                            status, action = "🟢 SMART DIP", "BUY DIP"
                        else:
                            status, action = "🔵 WEAK DIP", "WATCH"
                    else:
                        status, action = "🟡 GAP DOWN", "WATCH"

            results.append({
                'Ticker': ticker, 'Sector': sector_type, 'Regime': regime_status,
                'Gap%': gap_pct, 'Price': price, 'Prob': prob,
                'Mom%': mom_prob, 'Dip%': dip_prob, 'ATR%': atr_pct,
                'Model': model_used, 'Size': size, 'Action': action, 'Status': status, 'Vol_R': vol_ratio
            })

        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue

    print(f">>> Scanned {processed_count} tickers. Found {len(results)} potential setups.")

    if not results:
        print("No valid signals generated (Check Gaps/Data).")
        return

    # Sort Priority Logic (Adapted from V6.2.1)
    def get_sort_priority(r):
        action = r['Action']
        if "SELL" in action: return 0      # Top Priority: Actionable Sells
        if "BUY" in action: return 0       # Top Priority: Actionable Buys
        if "WATCH" in action: return 1     # Second Priority: Watchlist
        if "FLAT" in action: return 2      # Third Priority: Flat/Pass
        if "SKIP" in action: return 3      # Last Priority: Blocked
        return 4

    results.sort(key=lambda x: (get_sort_priority(x), -abs(x['Gap%'])))

    # Header Definition
    header = f"{'Ticker':<8} {'Sector':<10} {'Regime':<12} {'Gap%':>8} {'Price':>9} {'Status':<16} {'Action':<18} {'Model':<10} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6} {'Vol':>5} {'Size':<6} {'Note':<15}"
    print(header)
    print("-" * 185)

    last_priority = -1
    for r in results:
        curr_priority = get_sort_priority(r)

        # Section Separators
        if curr_priority != last_priority:
            if curr_priority == 0: print("-" * 60 + " [ 🚨 訊號區 (Actionable) ] " + "-" * 95)
            if curr_priority == 1: print("-" * 60 + " [ 👁️ 觀察區 (Watchlist) ] " + "-" * 96)
            if curr_priority == 2: print("-" * 60 + " [ 💤 盤整區 (Flat/Pass) ] " + "-" * 96)
            if curr_priority == 3: print("-" * 60 + " [ 🛑 禁止區 (Trend/Block) ] " + "-" * 94)
            last_priority = curr_priority

        # Markers
        marker = ""
        if "Caution" in r['Action']: marker = "<--- ⚠️ VOL"
        elif "SELL" in r['Action'] and r['Prob'] > 0.60: marker = "<--- 🔥 HOT"
        elif "BUY" in r['Action']: marker = "<--- 🟢 BUY"

        # Formatted Output
        sell_p = f"{r['Prob']:.0%}" if r['Prob'] > 0 else "-"
        mom_p = f"{r['Mom%']:.0%}" if r['Mom%'] > 0 else "-"
        dip_p = f"{r['Dip%']:.0%}" if r['Dip%'] > 0 else "-"
        atr_p = f"{r['ATR%']*100:.1f}%"
        vol_r = f"{r['Vol_R']:.1f}x"

        print(f"{r['Ticker']:<8} {r['Sector']:<10} {r['Regime']:<12} {r['Gap%']*100:>7.2f}% {r['Price']:>9.2f} {r['Status']:<16} {r['Action']:<18} {r['Model']:<10} {sell_p:>6} {mom_p:>6} {dip_p:>6} {atr_p:>6} {vol_r:>5} {r['Size']:<6} {marker}")

    out_path = os.path.join(OUTPUT_DIR, f"daily_signals_v6.2.6_{date.today()}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("-" * 140)
    print(f"Total Scanned: {processed_count} | [Saved] Report saved to: {out_path}")

if __name__ == "__main__":
    main()
