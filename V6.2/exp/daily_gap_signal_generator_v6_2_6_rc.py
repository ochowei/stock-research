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

# [V6.2 Integration] Import logic from production_daily_plan_v6_2 if available
try:
    from production_daily_plan_v6_2 import get_regime_decision, clean_ticker
except ImportError:
    # Fallback if not found (mocking for standalone execution safety)
    def get_regime_decision(df, ticker): return "✅ PASS", 0.0, {}
    def clean_ticker(t): return t.split(':')[1] if ':' in t else t

# --- Configuration ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Experiment Paths (Source of Truth)
# Correct Path Logic: BASE_DIR is V6.2/exp/. Sell_Model_Lab is inside V6.2/exp/.
EXP_DIR = os.path.join(BASE_DIR, 'Sell_Model_Lab', '03_Experiments', 'EXP_18_Production_Script_Update', '03_Output')
SECTOR_MAP_PATH = os.path.join(EXP_DIR, 'sector_map.json')

# Model Paths - Try specific RC files first, fall back to generic names
TECH_MODEL_CANDIDATES = [
    os.path.join(EXP_DIR, 'v6.2.4_rc_tech_model.joblib'),
    os.path.join(EXP_DIR, 'model_tech.joblib')
]
NON_TECH_MODEL_CANDIDATES = [
    os.path.join(EXP_DIR, 'v6.2.4_rc_non_tech_model.joblib'),
    os.path.join(EXP_DIR, 'model_non_tech.joblib')
]

def resolve_model_path(candidates):
    for path in candidates:
        if os.path.exists(path): return path
    return candidates[0] # Default to first

TECH_MODEL_PATH = resolve_model_path(TECH_MODEL_CANDIDATES)
NON_TECH_MODEL_PATH = resolve_model_path(NON_TECH_MODEL_CANDIDATES)

# Legacy Models (for console display context)
MOM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
DIP_MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# Strategy Parameters
DEFAULT_GAP_THRESHOLD = 0.005
RIP_THRESHOLD = 0.03
DIP_THRESHOLD = 0.03
MOMENTUM_THRESHOLD = 0.53
DIP_CONFIDENCE_LV = 0.50

# Feature Defs
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

# US Holidays 2026 (Updated for current context)
US_HOLIDAYS = [
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25'
]

# --- Helper Functions ---

def load_tickers_and_tags():
    tags_map = {}
    # Prioritize 2026 files if they exist, else fallback to 2025
    asset_file = '2026_final_asset_pool.json' if os.path.exists(os.path.join(RESOURCE_DIR, '2026_final_asset_pool.json')) else ASSET_POOL_FILE
    holding_file = '2026_holding_asset_pool.json' if os.path.exists(os.path.join(RESOURCE_DIR, '2026_holding_asset_pool.json')) else HOLDING_POOL_FILE

    for filename, tag in [(asset_file, 'Asset'), (holding_file, 'Held')]:
        path = os.path.join(RESOURCE_DIR, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    for t in raw:
                        clean_t = clean_ticker(t)
                        if clean_t not in tags_map: tags_map[clean_t] = set()
                        tags_map[clean_t].add(tag)
            except Exception as e: print(f"[Error] Failed to load {filename}: {e}")
    return sorted(list(tags_map.keys())), tags_map

def load_sector_map():
    if os.path.exists(SECTOR_MAP_PATH):
        try:
            with open(SECTOR_MAP_PATH, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

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

def download_data(tickers):
    # Ensure QQQ and SPY are included
    all_tickers = list(set(tickers + ['QQQ', 'SPY']))
    try:
        daily = yf.download(all_tickers, period="3mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True)
        intra = yf.download(tickers, period="5d", interval="1m", group_by='ticker', prepost=True, progress=False, auto_adjust=True)
        return daily, intra
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

def prepare_benchmark(bm_df, prefix):
    df = bm_df.copy()
    if len(df) < 50: return df # Not enough data
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df[f'{prefix}_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def build_features_latest(df, bm_df, prefix, bm_features):
    """Builds features for the LAST row only, matching EXP-18 logic"""
    if len(df) < 50: return None

    df = df.sort_index().copy()

    # Indicators
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

    # BM Features
    if last_idx not in bm_df.index:
        # Fallback to last available
        bm_row = bm_df.iloc[[-1]]
    else:
        bm_row = bm_df.loc[[last_idx]]

    for f in bm_features:
        if 'Corr' not in f and f in bm_row.columns:
            row[f] = bm_row[f].values[0]

    # Correlation
    common_idx = df.index.intersection(bm_df.index)
    df_sub = df.loc[common_idx]
    bm_sub = bm_df.loc[common_idx]

    if len(df_sub) > 20:
        aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
        corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1]).shift(1)
        corr_name = 'Sector_Corr' if prefix == 'QQQ' else 'Market_Corr'
        row[corr_name] = corr.iloc[-1]

    return row

def get_position_size(prob):
    if prob > 0.60: return "1.5x"
    if prob > 0.55: return "1.0x"
    return "0.5x"

# --- Main Program ---

def generate_report():
    cal_status, is_bullish_cal = get_calendar_status()
    current_gap_threshold = 0.01 if is_bullish_cal else DEFAULT_GAP_THRESHOLD
    curr_vix = get_current_vix()

    print(f"\n>>> V6.2.6 RC Daily Gap Scanner (Ensemble + Sizing + MOC)")
    print(f">>> [Market Context] 📅 Calendar: {cal_status} | VIX: {curr_vix:.1f}")
    print("-" * 155)

    # Load Models
    models = {
        'mom': joblib.load(MOM_MODEL_PATH) if os.path.exists(MOM_MODEL_PATH) else None,
        'dip': joblib.load(DIP_MODEL_PATH) if os.path.exists(DIP_MODEL_PATH) else None,
        'tech': joblib.load(TECH_MODEL_PATH) if os.path.exists(TECH_MODEL_PATH) else None,
        'non_tech': joblib.load(NON_TECH_MODEL_PATH) if os.path.exists(NON_TECH_MODEL_PATH) else None
    }

    if models['tech'] is None or models['non_tech'] is None:
        print("[Error] Critical models (Tech/Non-Tech) missing. Check paths.")
        print(f"Tech: {TECH_MODEL_PATH}")
        print(f"Non-Tech: {NON_TECH_MODEL_PATH}")
        return

    sector_map = load_sector_map()
    tickers, tags_map = load_tickers_and_tags()
    daily_data, intra_data = download_data(tickers)

    # Prepare Benchmark Data
    try:
        qqq_df = daily_data['QQQ'].dropna(how='all')
        spy_df = daily_data['SPY'].dropna(how='all')
        qqq_prep = prepare_benchmark(qqq_df, 'QQQ')
        spy_prep = prepare_benchmark(spy_df, 'SPY')
    except Exception as e:
        print(f"[Error] Failed to prepare benchmark data: {e}")
        return

    results = []

    for t in tickers:
        try:
            # Handle Single Level Column if only one ticker (unlikely due to QQQ/SPY but safe to check)
            if isinstance(daily_data.columns, pd.MultiIndex):
                if t not in daily_data.columns.get_level_values(0): continue
                df_t_daily = daily_data[t].copy()
            else:
                if t != daily_data.name: continue # Should not happen with multiple tickers
                df_t_daily = daily_data.copy()

            if df_t_daily.empty or len(df_t_daily) < 50: continue

            df_t_intra = pd.DataFrame()
            if not intra_data.empty and isinstance(intra_data.columns, pd.MultiIndex) and t in intra_data.columns.get_level_values(0):
                 df_t_intra = intra_data[t]

            # Basic Metrics for Display (simpler calculation like v6.2.1)
            prev_close = float(df_t_daily['Close'].iloc[-2]) # Use yesterday for ref
            curr_price = float(df_t_daily['Close'].iloc[-1]) # Today's current/close

            # If intra available, update current price
            if not df_t_intra.empty:
                df_m = df_t_intra.dropna(subset=['Close'])
                if not df_m.empty: curr_price = float(df_m['Close'].iloc[-1])

            # Use Open for Gap calculation as per model training
            open_price = float(df_t_daily['Open'].iloc[-1])
            gap_pct = (open_price - prev_close) / prev_close

            # ATR/RSI for display (using simple method from v6.2.1)
            atr_val = ta.atr(df_t_daily['High'], df_t_daily['Low'], df_t_daily['Close'], length=14).iloc[-2]
            atr_pct = atr_val / prev_close

            # Regime
            regime_status, er_val, _ = get_regime_decision(df_t_daily, t)

            # Determine Sector and Model
            sector = sector_map.get(t, 'Unknown')
            is_tech = (sector == 'Technology')

            # Build Features for Prediction (EXP-18 Logic)
            feat_row = None
            if is_tech:
                feat_row = build_features_latest(df_t_daily, qqq_prep, 'QQQ', TECH_FEATURES)
            else:
                feat_row = build_features_latest(df_t_daily, spy_prep, 'SPY', NON_TECH_FEATURES)

            if feat_row is None: continue

            # AI Prediction
            probs = {}

            # Main Sell Model
            sell_prob = 0.0
            if is_tech:
                X = feat_row[BASE_FEATURES + TECH_FEATURES]
                sell_prob = models['tech'].predict_proba(X)[0][1]
            else:
                X = feat_row[BASE_FEATURES + NON_TECH_FEATURES]
                sell_prob = models['non_tech'].predict_proba(X)[0][1]

            probs['sell'] = f"{sell_prob:.0%}"
            probs['sell_val'] = sell_prob

            # Legacy Models (Mom/Dip)
            legacy_feats = pd.DataFrame([[
                feat_row['RSI_14'].iloc[0],
                feat_row['ATR_Pct'].iloc[0],
                feat_row['Vol_Ratio'].iloc[0],
                feat_row['Gap_Pct'].iloc[0],
                curr_vix,
                feat_row['Dist_MA20'].iloc[0]
            ]], columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20'])

            if models['mom']:
                try: probs['mom'] = f"{models['mom'].predict_proba(legacy_feats.iloc[:, :5])[0][1]:.0%}"
                except: probs['mom'] = "-"
            else: probs['mom'] = "-"

            if models['dip']:
                try: probs['dip'] = f"{models['dip'].predict_proba(legacy_feats)[0][1]:.0%}"
                except: probs['dip'] = "-"
            else: probs['dip'] = "-"

            # Position Sizing
            pos_size = get_position_size(sell_prob)

            # Decision Logic
            status, action = "Flat", "-"

            # Gap Quality Filter (EXP-24)
            vol_ratio = feat_row['Vol_Ratio'].iloc[0]
            high_vol_warning = vol_ratio > 3.0

            if regime_status == "🛑 BLOCK":
                status, action = "🛑 BLOCK (Trendy)", "SKIP"
            else:
                if gap_pct > RIP_THRESHOLD:
                    # Large Gap Logic
                    status, action = "🚀 ROCKET", "HOLD/BUY" # Default for massive gaps often momentum
                    if sell_prob > 0.60:
                        status, action = "🔴 SELL RIP", "MOC (Strong)"
                    elif sell_prob > 0.50:
                        status, action = "🔴 SELL RIP", "MOC"
                elif gap_pct > current_gap_threshold:
                    if sell_prob > 0.50:
                        status, action = "🔴 GAP UP", "MOC"
                    else:
                        status, action = "🟢 MOMENTUM", "HOLD"
                elif gap_pct < -DIP_THRESHOLD:
                    status, action = "🟢 SMART DIP", "WATCH"
                elif gap_pct < -DEFAULT_GAP_THRESHOLD:
                    status, action = "🟡 GAP DOWN", "HOLD"

            is_held = 'Held' in tags_map.get(t, set())

            note = ""
            if high_vol_warning: note = "⚠️ High_Vol"
            if is_tech: note += " [Tech]"
            else: note += " [Non-Tech]"

            results.append({
                'Ticker': t, 'Tag': "[HOLD]" if is_held else "",
                'Regime': regime_status, 'Gap%': gap_pct, 'Price': curr_price,
                'Status': status, 'Action': action,
                'Sell%': probs['sell'], 'Mom%': probs['mom'], 'Dip%': probs['dip'],
                'ATR%': atr_pct, 'Vol': f"{vol_ratio:.1f}x",
                'Size': pos_size if 'SELL' in status or 'MOC' in action else "-",
                'Note': note.strip()
            })

        except Exception as e:
            # print(f"Error processing {t}: {e}")
            continue

    # --- Sorting and Output ---
    def get_sort_priority(r):
        if 'MOC' in r['Action'] or 'SELL' in r['Status']: return 0
        if r['Action'] in ['HOLD', 'WATCH']: return 1
        if r['Regime'] == "✅ PASS": return 2
        return 3

    results.sort(key=lambda x: (get_sort_priority(x), -abs(x['Gap%'])))

    header = f"{'Ticker':<8} {'Tag':<6} {'Regime':<12} {'Gap%':>8} {'Price':>9} {'Status':<16} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6} {'Vol':>5} {'Size':>5} {'Note':<15}"
    print(header)
    print("-" * 155)

    last_priority = -1
    for r in results:
        curr_priority = get_sort_priority(r)
        if curr_priority != last_priority:
            if curr_priority == 1: print("-" * 45 + " [ HOLD / WATCH ] " + "-" * 90)
            if curr_priority == 2: print("-" * 45 + " [ FLAT / NO SIGNAL ] " + "-" * 88)
            if curr_priority == 3: print("-" * 45 + " [ BLOCKED ] " + "-" * 95)
            last_priority = curr_priority

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Regime']:<12} {r['Gap%']*100:>7.2f}% {r['Price']:>9.2f} {r['Status']:<16} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}% {r['Vol']:>5} {r['Size']:>5} {r['Note']:<15}")

    print("-" * 155)
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_v6.2.6_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Total Scanned: {len(results)} | [Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()
