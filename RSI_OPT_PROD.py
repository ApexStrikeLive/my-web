import pandas as pd
import ccxt
import numpy as np
import time
import datetime
import json
import os
from tqdm import tqdm
import uuid

# --- ANSI Color Codes ---
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"

# --- Configuration ---
# Replace with your actual API key and secret
api_key = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
api_secret = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'

# --- Strategy and File Management Constants ---
STRATEGY_NAME = "RSI_OPT_PROD" # Defined globally
SIGNAL_LOG_FILE = 'RSI_OPT_PROD.json'

TIMEFRAMES = ['1h', '15m', '30m', '2h', '4h']
CANDLE_LIMIT = 250  # Number of historical candles to fetch

COINS_TO_EXCLUDE = [
    'BZRX/USDT',
    'AAVEUP/USDT',
    'LEND/USDT',
    'USDC/USDT', 'BUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'TUSD/USDT',
    'PYUSD/USDT', 'GUSD/USDT', 'USTC/USDT', 'MIM/USDT',
    'EURT/USDT', 'BKRW/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'ETHUP/USDT', 'ETHDOWN/USDT', 'BTCUP/USDT', 'BTCDOWN/USDT', 'BNBUP/USDT', 'BNBDOWN/USDT',
    'ETC/USDT', 'XEC/USDT', 'LUNC/USDT', 'TRB/USDT', 'PERP/USDT', 'FTM/USDT',
    'NEAR/USDT', 'ICP/USDT', 'RNDR/USDT', 'INJ/USDT', 'TIA/USDT', 'ALT/USDT',
    'SEI/USDT', 'PYTH/USDT', 'JUP/USDT',
    'CITY/USDT', 'PSG/USDT', 'LAZIO/USDT'
]

# Strategy Parameters (RSI focused)
RSI_PERIOD = 14  # Period for RSI calculation

# --- ADJUSTED PARAMETERS ---

# 1. Strengthen RSI Confirmation Requirements (Adjusted)
# Will be set dynamically based on volatility

# 2. Add RSI Momentum Filter (Reduced)
RSI_MIN_MOMENTUM = 3  # Reduced from 4

# 3. Enhance Volume Filtering (Reduced)
MIN_VOLUME_RATIO = 1.1  # Reduced from 1.25
VOLUME_INCREASING_BARS = 1  # Kept at 1, consider 0 if still too restrictive (logic for 0 would mean removing the explicit increasing check)

# 4. Add Price Action Confluence (Adjusted)
MIN_CANDLE_BODY_PCT = 0.002  # Keep
REQUIRE_BREAKOUT_CONFIRMATION = False  # Keep False
LOOKBACK_RESISTANCE_SUPPORT = 15  # Keep

# 5. Implement Multi-Timeframe Confirmation (Reduced Buffer)
HIGHER_TIMEFRAME_MAP = {
    '15m': '30m',
    '30m': '2h',
    '1h': '4h',
    '2h': '4h',
    '4h': '1d',
}
HTF_BUFFER = 2  # Reduced from 5

# 6. Add Market Structure Filter (Reduced Strength)
TREND_LOOKBACK = 50
MIN_TREND_STRENGTH = 0.005  # Reduced from 0.008

# 7. Increase Confirmation Period (Reduced)
RSI_ENTRY_CONFIRM_BARS = 1  # Reduced from 2

# 8. Add Time-Based Filters (Adjusted)
AVOID_MARKET_OPEN_MINUTES = 0  # Reduced from 30, set to 0 to effectively remove
AVOID_LOW_LIQUIDITY_HOURS = [3, 4]  # Keep

# 9. Implement Signal Cooldown (Reduced Timeframe Cooldown) - REMOVED
# 10. Add Volatility Regime Filter
VOLATILITY_MIN_PERCENTILE = 30  # Reduced from 40

# Scheduling
SCAN_INTERVAL_MINUTES = 15
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

# Initialize exchange
try:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        },
    })
    print("Successfully connected to Binance client via CCXT.")
except Exception as e:
    print(f"Error connecting to Binance client: {e}")
    exit()

# --- Data Fetching and Indicator Calculation ---

def fetch_binance_data(symbol, timeframe, limit=CANDLE_LIMIT):
    """Fetches OHLCV data for a given symbol and timeframe using CCXT."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)
        return df
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error fetching {symbol} {timeframe}: {e}{COLOR_RESET}")
        return None


def calculate_indicators(df, rsi_period=RSI_PERIOD):
    """Calculates RSI and other indicators for the given DataFrame."""
    if df.empty:
        return df

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    # Volatility Calculation (e.g., historical standard deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()

    # Candle Body Percentage
    df['CandleBodyPct'] = abs(df['Close'] - df['Open']) / df['Open']

    # For Market Structure Filter (e.g., Simple Moving Average)
    df['SMA_TREND'] = df['Close'].rolling(window=TREND_LOOKBACK).mean()

    return df


def get_timeframe_in_minutes(tf_str):
    if 'm' in tf_str:
        return int(tf_str.replace('m', ''))
    elif 'h' in tf_str:
        return int(tf_str.replace('h', '')) * 60
    elif 'd' in tf_str:
        return int(tf_str.replace('d', '')) * 24 * 60
    return 0


# --- Adaptive RSI Thresholds ---
def get_dynamic_rsi_thresholds(volatility_percentile):
    """Set RSI thresholds based on current volatility conditions"""
    if volatility_percentile > 70:  # High volatility
        return 28, 72  # Wider bands
    elif volatility_percentile > 40:  # Medium volatility
        return 30, 70  # Standard bands
    else:  # Low volatility
        return 32, 68  # Tighter bands


# --- RSI Signal Generation ---

def generate_trading_signals(df, symbol, timeframe):
    """Generates trading signals based on RSI strategy."""
    signals = []
    min_required_bars = max(RSI_PERIOD, TREND_LOOKBACK, LOOKBACK_RESISTANCE_SUPPORT) + RSI_ENTRY_CONFIRM_BARS + 1

    if df.empty or len(df) < min_required_bars:
        return signals

    required_cols = ['RSI', 'Close', 'Volume', 'CandleBodyPct', 'SMA_TREND', 'Volatility']
    df_cleaned = df.dropna(subset=required_cols).copy()

    if df_cleaned.empty or len(df_cleaned) < min_required_bars:
        return signals

    i = len(df_cleaned) - 1
    current_bar = df_cleaned.iloc[i]
    current_rsi = current_bar['RSI']
    current_close = current_bar['Close']
    current_open = current_bar['Open']
    current_high = current_bar['High']
    current_low = current_bar['Low']
    current_volume = current_bar['Volume']
    current_candle_body_pct = current_bar['CandleBodyPct']
    current_volatility = current_bar['Volatility']

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    current_utc_hour = signal_timestamp.hour
    current_minute = signal_timestamp.minute
    if current_utc_hour in AVOID_LOW_LIQUIDITY_HOURS:
        return signals

    if AVOID_MARKET_OPEN_MINUTES > 0 and current_minute < AVOID_MARKET_OPEN_MINUTES:
        pass

    recent_volatilities = df_cleaned['Volatility'].iloc[max(0, i - 100):i + 1].dropna()
    if not recent_volatilities.empty:
        volatility_threshold = recent_volatilities.quantile(VOLATILITY_MIN_PERCENTILE / 100)
        volatility_ok = current_volatility >= volatility_threshold
        if not volatility_ok:
            return signals
    else:
        return signals

    if not recent_volatilities.empty:
        volatility_percentile_for_threshold = (recent_volatilities < current_volatility).mean() * 100
    else:
        volatility_percentile_for_threshold = 50

    RSI_LONG_OVERSOLD_MAX, RSI_SHORT_OVERBOUGHT_MIN = get_dynamic_rsi_thresholds(volatility_percentile_for_threshold)

    volume_lookback_period = 10
    avg_volume_window = df_cleaned['Volume'].iloc[max(0, i - volume_lookback_period):i]
    avg_volume = avg_volume_window.mean() if not avg_volume_window.empty else 0
    volume_ok = (current_volume / avg_volume >= MIN_VOLUME_RATIO) if avg_volume > 0 else False

    volume_increasing = True
    if VOLUME_INCREASING_BARS > 0 and i >= VOLUME_INCREASING_BARS:
        for k in range(1, VOLUME_INCREASING_BARS + 1):
            if df_cleaned['Volume'].iloc[i - k + 1] <= df_cleaned['Volume'].iloc[i - k]:
                volume_increasing = False
                break
    elif VOLUME_INCREASING_BARS == 0:
        volume_increasing = True
    else:
        volume_increasing = False

    if not (volume_ok and volume_increasing):
        return signals

    rsi_confirm_start_idx = max(0, i - RSI_ENTRY_CONFIRM_BARS)
    price_confirm_start_idx = max(0, i - RSI_ENTRY_CONFIRM_BARS)

    rsi_movement = df_cleaned['RSI'].iloc[rsi_confirm_start_idx:i + 1]
    price_movement = df_cleaned['Close'].iloc[price_confirm_start_idx:i + 1]

    rsi_rising = current_rsi > rsi_movement.iloc[0]
    price_rising = current_close > price_movement.iloc[0]

    rsi_falling = current_rsi < rsi_movement.iloc[0]
    price_falling = current_close < price_movement.iloc[0]

    rsi_momentum_long = (current_rsi - rsi_movement.iloc[0]) >= RSI_MIN_MOMENTUM
    rsi_momentum_short = (rsi_movement.iloc[0] - current_rsi) >= RSI_MIN_MOMENTUM

    if current_candle_body_pct < MIN_CANDLE_BODY_PCT:
        return signals

    trend_start_idx = max(0, i - TREND_LOOKBACK)
    trend_slice = df_cleaned['Close'].iloc[trend_start_idx:i + 1]
    if trend_slice.empty:
        return signals

    initial_trend_price = trend_slice.iloc[0]
    is_trending_up = False
    is_trending_down = False

    if initial_trend_price != 0:
        trend_change_pct = (current_close - initial_trend_price) / initial_trend_price
        if trend_change_pct >= MIN_TREND_STRENGTH:
            is_trending_up = True
        elif trend_change_pct <= -MIN_TREND_STRENGTH:
            is_trending_down = True
    else:
        return signals

    recent_highs = df_cleaned['High'].iloc[max(0, i - LOOKBACK_RESISTANCE_SUPPORT):i].max()
    recent_lows = df_cleaned['Low'].iloc[max(0, i - LOOKBACK_RESISTANCE_SUPPORT):i].min()

    breakout_long_flag = False
    if REQUIRE_BREAKOUT_CONFIRMATION:
        if current_high > recent_highs:
            breakout_long_flag = True
    else:
        breakout_long_flag = current_close > df_cleaned['Close'].iloc[i - 1] if i > 0 else False

    breakout_short_flag = False
    if REQUIRE_BREAKOUT_CONFIRMATION:
        if current_low < recent_lows:
            breakout_short_flag = True
    else:
        breakout_short_flag = current_close < df_cleaned['Close'].iloc[i - 1] if i > 0 else False

    higher_timeframe_aligned = True
    if timeframe in HIGHER_TIMEFRAME_MAP:
        htf = HIGHER_TIMEFRAME_MAP[timeframe]
        htf_df = fetch_binance_data(symbol, htf)
        if htf_df is not None and not htf_df.empty:
            htf_df = calculate_indicators(htf_df)
            if not htf_df.dropna(subset=['RSI']).empty:
                htf_current_rsi = htf_df['RSI'].iloc[-1]
                if current_rsi <= RSI_LONG_OVERSOLD_MAX and htf_current_rsi >= (RSI_SHORT_OVERBOUGHT_MIN - HTF_BUFFER):
                    higher_timeframe_aligned = False
                if current_rsi >= RSI_SHORT_OVERBOUGHT_MIN and htf_current_rsi <= (RSI_LONG_OVERSOLD_MAX + HTF_BUFFER):
                    higher_timeframe_aligned = False
            else:
                higher_timeframe_aligned = False
        else:
            higher_timeframe_aligned = False

    if not higher_timeframe_aligned:
        return signals

    # LONG Entry Conditions
    if (current_rsi <= RSI_LONG_OVERSOLD_MAX and rsi_rising and price_rising and rsi_momentum_long and
            volume_ok and volume_increasing and current_candle_body_pct >= MIN_CANDLE_BODY_PCT and
            is_trending_up and breakout_long_flag and higher_timeframe_aligned):
        long_signal = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "BUY",
            "date": signal_date,
            "time": signal_time,
            "entry_price": round(current_close, 6),
            "strategy_name": STRATEGY_NAME
        }
        signals.append(long_signal)

    # SHORT Entry Conditions
    if (current_rsi >= RSI_SHORT_OVERBOUGHT_MIN and rsi_falling and price_falling and rsi_momentum_short and
            volume_ok and volume_increasing and current_candle_body_pct >= MIN_CANDLE_BODY_PCT and
            is_trending_down and breakout_short_flag and higher_timeframe_aligned):
        short_signal = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": "SELL",
            "date": signal_date,
            "time": signal_time,
            "entry_price": round(current_close, 6),
            "strategy_name": STRATEGY_NAME
        }
        signals.append(short_signal)

    return signals


# --- JSON File Management Functions ---

def load_existing_signals_data(filename: str = SIGNAL_LOG_FILE) -> list[dict]:
    """
    Loads existing signals from the JSON file.
    Returns a list of dictionaries, or an empty list if file is not found or corrupted.
    """
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                tqdm.write(f"{COLOR_YELLOW}Warning: {filename} has unexpected structure. Returning empty list.{COLOR_RESET}")
                return []
    except json.JSONDecodeError:
        tqdm.write(f"{COLOR_YELLOW}Warning: Could not decode JSON from {filename}. Returning empty list.{COLOR_RESET}")
        return []
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error loading signals from {filename}: {e}. Returning empty list.{COLOR_RESET}")
        return []

def save_signals_to_file(new_signals: list[dict], filename: str = SIGNAL_LOG_FILE):
    """
    Appends new unique signals to the existing JSON file.
    Uniqueness is determined by symbol, timeframe, signal, date, time, and strategy_name.
    """
    existing_signals = load_existing_signals_data(filename)

    added_count = 0
    existing_signal_keys = set()
    for s in existing_signals:
        if isinstance(s, dict) and all(k in s for k in ['symbol', 'timeframe', 'signal', 'date', 'time', 'strategy_name']):
            existing_signal_keys.add(
                (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name'])
            )

    for signal in new_signals:
        signal_key = (
            signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
            signal.get('date'), signal.get('time'), signal.get('strategy_name')
        )
        if signal_key not in existing_signal_keys:
            existing_signals.append(signal)
            added_count += 1

    try:
        with open(filename, 'w') as f:
            json.dump(existing_signals, f, indent=2)
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error saving signals to {filename}: {e}{COLOR_RESET}")
    return added_count

def print_signal_to_console(signal: dict):
    """
    Prints the simplified signal to the console.
    """
    signal_color = COLOR_GREEN if signal['signal'] == 'BUY' else COLOR_RED
    tqdm.write(
        f"SIGNAL: {signal_color}{signal['signal']}{COLOR_RESET} | PAIR: {signal['symbol']} | PRICE: {signal['entry_price']:.6f} | TIME: {signal['date']} {signal['time']}")

# --- UI Helper Functions ---

def countdown_timer(seconds: int, total_seconds: int, message_prefix: str = "Next scan in"):
    """
    Displays a countdown to the next scan with a progress bar.
    """
    while seconds > 0:
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        timer = f'{hours:02d}:{mins:02d}:{secs:02d}'

        percent = int(100 * (total_seconds - seconds) / total_seconds)
        bar_length = 50
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        print(f'\r{message_prefix}: [{bar}] {timer} remaining', end='', flush=True)
        time.sleep(1)
        seconds -= 1
    print("\n", end="", flush=True)

# --- Main Scanning Logic ---

def monitor_signals():
    """
    Main loop for monitoring trading signals, fetching data,
    calculating indicators, and saving signals to a JSON file.
    """
    print(f"Starting Binance Trading Signal Scanner ({STRATEGY_NAME})...")

    print("Loading tradable symbols from Binance...")
    try:
        exchange.load_markets()
        all_usdt_pairs = [s for s in exchange.symbols if s.endswith('/USDT') and s not in COINS_TO_EXCLUDE]
        print(f"Found {len(all_usdt_pairs)} USDT pairs to scan (after exclusions).")
    except Exception as e:
        print(f"{COLOR_RED}Error loading markets: {e}. Retrying in 60 seconds.{COLOR_RESET}")
        time.sleep(60)
        return

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n--- Starting new scan at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        signals_this_scan = []

        existing_signals_data = load_existing_signals_data(SIGNAL_LOG_FILE)
        existing_signal_keys = set()
        for s in existing_signals_data:
            if isinstance(s, dict) and all(k in s for k in ['symbol', 'timeframe', 'signal', 'date', 'time', 'strategy_name']):
                existing_signal_keys.add(
                    (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name'])
                )

        total_iterations = len(all_usdt_pairs) * len(TIMEFRAMES)
        with tqdm(total=total_iterations, desc="Analyzing Pairs", unit="pair_tf", ncols=100) as pbar:
            for pair in all_usdt_pairs:
                for timeframe in TIMEFRAMES:
                    df = fetch_binance_data(pair, timeframe)
                    if df is None or df.empty:
                        pbar.update(1)
                        continue

                    df_with_indicators = calculate_indicators(df)
                    if df_with_indicators.empty or len(df_with_indicators.dropna(subset=['RSI', 'Close', 'Volume', 'CandleBodyPct', 'SMA_TREND', 'Volatility'])) < 2:
                        pbar.update(1)
                        continue

                    newly_generated_signals = generate_trading_signals(df_with_indicators, pair, timeframe)

                    for signal in newly_generated_signals:
                        signal_key = (
                            signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
                            signal.get('date'), signal.get('time'), signal.get('strategy_name')
                        )
                        if signal_key not in existing_signal_keys:
                            signals_this_scan.append(signal)
                            existing_signals_data.append(signal)
                            existing_signal_keys.add(signal_key)
                            print_signal_to_console(signal)
                    pbar.update(1)
                    time.sleep(0.05)

        print("\nAnalysis complete.")

        if signals_this_scan:
            added_count = save_signals_to_file(signals_this_scan, SIGNAL_LOG_FILE)
            print(f"\n{COLOR_CYAN}Saved {added_count} new unique signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
        else:
            print(f"\nNo new signals found in this scan.")

        print(f"\nScan finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)

# --- Run the script ---
if __name__ == "__main__":
    monitor_signals()