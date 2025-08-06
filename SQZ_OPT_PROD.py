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
# Note: For security, consider using environment variables for API keys in production
api_key = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
api_secret = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'

# --- Strategy and File Management Constants ---
STRATEGY_NAME = "SQZ_OPT_PROD" # Defined globally
SIGNAL_LOG_FILE = 'SQZ_OPT_PROD.json'

TIMEFRAMES = ['1h', '2h', '4h', '30m']
CANDLE_LIMIT = 100

COINS_TO_EXCLUDE = [
    'BZRX/USDT', 'AAVEUP/USDT', 'LEND/USDT',
    'USDC/USDT', 'BUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'TUSD/USDT',
    'PYUSD/USDT', 'USDP/USDT', 'GUSD/USDT', 'USTC/USDT', 'MIM/USDT',
    'EURT/USDT', 'BKRW/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'ETHUP/USDT', 'ETHDOWN/USDT', 'BTCUP/USDT', 'BTCDOWN/USDT', 'BNBUP/USDT', 'BNBDOWN/USDT',
    'ETC/USDT', 'XEC/USDT', 'LUNC/USDT', 'TRB/USDT', 'PERP/USDT', 'FTM/USDT',
    'NEAR/USDT', 'ICP/USDT', 'RNDR/USDT', 'INJ/USDT', 'TIA/USDT', 'ALT/USDT',
    'SEI/USDT', 'PYTH/USDT', 'JUP/USDT',
    'CITY/USDT', 'PSG/USDT', 'LAZIO/USDT',
    # --- ADDED BASED ON YOUR LOGS AND REQUEST ---
    'ERN/USDT', # Explicitly exclude ERN/USDT as its price isn't changing and was flagged
    'ATOM/USDT', # Causing fetching errors
    'SUSD/USDT', # Causing fetching errors
    'AION/USDT', # Causing fetching errors
    'HIVE/USDT', # Causing fetching errors
    'LINA/USDT', # Causing fetching errors
    'MDT/USDT', # Causing fetching errors
    'MDX/USDT', # Causing fetching errors
    # BTC/USDT is a major pair and should ideally not cause errors.
    # If BTC/USDT fetching errors persist, it's usually a rate limit or a temporary Binance issue.
    # We will not exclude BTC/USDT by default, but you can add it if absolutely necessary.
]

# Strategy Parameters (Triple-Confirmation VWAP Breakout Strategy)
VWAP_PERIOD = 24
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
EMA_50_PERIOD = 50
VOLUME_MA_PERIOD = 20

# Entry Confirmation Filters
MIN_VOLUME_RATIO = 1.5
MIN_MACD_HIST_ABS = 0.01

# Scheduling
SCAN_INTERVAL_MINUTES = 15
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

# --- ENHANCED CONFIGURATION PARAMETERS (New) ---
ENHANCED_CONFIG = {
    'SIGNAL_COOLDOWN_HOURS': 4,
    'MIN_MOMENTUM_SCORE': 0.02,
    'MIN_TREND_STRENGTH': 0.6,
    'MIN_EMA_SLOPE': 0.001,
    'ATR_PERIOD': 14,
    'MAX_RSI_LONG': 70,
    'MIN_RSI_SHORT': 30,
    'STRONGER_VWAP_THRESHOLD': 0.002,
    'MIN_VOLUME_RATIO': MIN_VOLUME_RATIO
}

# Define required columns globally so it's accessible everywhere needed
required_cols = ['Close', 'Volume', 'VWAP', 'RSI', 'MACD_Line', 'Signal_Line', 'Histogram', 'EMA_50', 'Volume_MA',
                 'ATR']

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

def calculate_atr(df, period=ENHANCED_CONFIG['ATR_PERIOD']):
    """Calculate Average True Range for volatility-based TP"""
    if df.empty or len(df) < period:
        return pd.Series([np.nan] * len(df), index=df.index)

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_indicators(df):
    """
    Calculates VWAP, RSI, MACD, EMA_50, Volume MA, and ATR indicators for the given DataFrame.
    Periods are sourced from global configuration and ENHANCED_CONFIG.
    """
    if df.empty:
        return df.copy()

    df_copy = df.copy()

    # VWAP Calculation (Rolling VWAP)
    df_copy['Typical_Price'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
    if len(df_copy) >= VWAP_PERIOD and df_copy['Volume'].sum() > 0:
        df_copy['VWAP'] = (df_copy['Typical_Price'] * df_copy['Volume']).rolling(window=VWAP_PERIOD).sum() / \
                     df_copy['Volume'].rolling(window=VWAP_PERIOD).sum()
    else:
        df_copy['VWAP'] = np.nan

    # RSI Calculation
    delta = df_copy['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    ema_fast = df_copy['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    ema_slow = df_copy['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df_copy['MACD_Line'] = ema_fast - ema_slow
    df_copy['Signal_Line'] = df_copy['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    df_copy['Histogram'] = df_copy['MACD_Line'] - df_copy['Signal_Line']

    # EMA 50 Calculation
    df_copy['EMA_50'] = df_copy['Close'].ewm(span=EMA_50_PERIOD, adjust=False).mean()

    # Volume MA Calculation
    if len(df_copy) >= VOLUME_MA_PERIOD:
        df_copy['Volume_MA'] = df_copy['Volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    else:
        df_copy['Volume_MA'] = np.nan

    # ATR Calculation
    df_copy['ATR'] = calculate_atr(df_copy, period=ENHANCED_CONFIG['ATR_PERIOD'])

    return df_copy


# --- Enhanced Signal Quality Functions ---

def check_signal_cooldown(symbol, timeframe, signals_data: list[dict]) -> bool:
    """
    Checks if the cooldown period has passed for a given symbol and timeframe.
    `signals_data` is expected to be a list of signal dictionaries directly from the JSON.
    """
    recent_signals = [s for s in signals_data
                      if s.get('symbol') == symbol and s.get('timeframe') == timeframe]

    if not recent_signals:
        return True

    try:
        last_signal_entry = max(recent_signals,
                                key=lambda x: datetime.datetime.strptime(f"{x.get('date', '1970-01-01')} {x.get('time', '00:00:00')}", "%Y-%m-%d %H:%M:%S"))
        last_signal_time = datetime.datetime.strptime(
            f"{last_signal_entry.get('date', '1970-01-01')} {last_signal_entry.get('time', '00:00:00')}",
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return True

    time_diff = datetime.datetime.now() - last_signal_time
    return time_diff.total_seconds() >= ENHANCED_CONFIG['SIGNAL_COOLDOWN_HOURS'] * 3600

def calculate_momentum_score(df):
    """Calculate comprehensive momentum score"""
    if len(df) < 2:
        return 0.0

    current = df.iloc[-1]
    previous = df.iloc[-2]

    required_momentum_cols = ['Close', 'Volume', 'VWAP', 'RSI', 'Volume_MA']
    if not all(col in current and col in previous for col in required_momentum_cols):
        return 0.0

    if current['Volume_MA'] == 0:
        return 0.0

    price_momentum = (current['Close'] - previous['Close']) / previous['Close'] if previous['Close'] != 0 else 0
    vol_momentum = (current['Volume'] - current['Volume_MA']) / current['Volume_MA']
    vwap_momentum = (current['Close'] - current['VWAP']) / current['VWAP'] if current['VWAP'] != 0 else 0
    rsi_momentum = (current['RSI'] - 50) / 50

    momentum_score = (
            price_momentum * 0.3 +
            vol_momentum * 0.25 +
            vwap_momentum * 0.25 +
            rsi_momentum * 0.2
    )
    return momentum_score

def calculate_trend_strength(df, lookback=20):
    """Calculate trend strength using price action and EMA slope"""
    if len(df) < lookback + 5:
        return 0.0, 0.0

    closes = df['Close'].tail(lookback)
    if len(closes) < 2:
        return 0.0, 0.0

    higher_highs = sum(1 for i in range(1, len(closes)) if closes.iloc[i] > closes.iloc[i - 1])
    higher_lows = sum(1 for i in range(1, len(df['Low'].tail(lookback))) if df['Low'].tail(lookback).iloc[i] > df['Low'].tail(lookback).iloc[i - 1])

    uptrend_strength = (higher_highs + higher_lows) / (2 * lookback) if lookback > 0 else 0

    ema_slope = 0.0
    if 'EMA_50' in df and len(df) >= EMA_50_PERIOD + 5:
        if df['EMA_50'].iloc[-5] != 0:
            ema_slope = (df['EMA_50'].iloc[-1] - df['EMA_50'].iloc[-5]) / df['EMA_50'].iloc[-5]
    return uptrend_strength, ema_slope

def check_market_conditions(df):
    """Check if market conditions are favorable for trading"""
    if len(df) < 20:
        return False
    if 'Close' not in df.columns or len(df['Close']) < 2:
        return False

    volatility = df['Close'].pct_change().std() * 100

    atr_series = calculate_atr(df, period=ENHANCED_CONFIG['ATR_PERIOD'])
    if atr_series.empty or atr_series.iloc[-1] is np.nan:
        return False

    atr_market_cond = atr_series.iloc[-1]
    avg_price = df['Close'].tail(20).mean()

    if avg_price == 0:
        return False

    MIN_ATR_PERCENTAGE = 0.1
    MAX_VOLATILITY_PERCENTAGE = 5.0
    atr_percentage = (atr_market_cond / avg_price) * 100

    if volatility > MAX_VOLATILITY_PERCENTAGE or atr_percentage < MIN_ATR_PERCENTAGE:
        return False
    return True

# --- VWAP Breakout Signal Generation ---

def generate_enhanced_signals(df: pd.DataFrame, symbol: str, timeframe: str, existing_signals: list[dict]) -> list[dict]:
    """Generates trading signals based on the Triple-Confirmation VWAP Breakout Strategy."""
    signals_to_log = []
    min_required_bars = max(VWAP_PERIOD, RSI_PERIOD, MACD_SLOW_PERIOD, EMA_50_PERIOD, VOLUME_MA_PERIOD,
                            ENHANCED_CONFIG['ATR_PERIOD']) + 5

    if df.empty or len(df) < min_required_bars:
        return signals_to_log

    df_cleaned = df.dropna(subset=required_cols).copy()

    if df_cleaned.empty or len(df_cleaned) < 2:
        return signals_to_log

    current_bar = df_cleaned.iloc[-1]
    previous_bar = df_cleaned.iloc[-2]

    if any(pd.isna(current_bar[col]) for col in ['Close', 'Volume', 'VWAP', 'RSI', 'Histogram', 'EMA_50', 'Volume_MA', 'ATR']):
        return signals_to_log
    if pd.isna(previous_bar['Histogram']):
        return signals_to_log

    current_close = current_bar['Close']
    current_volume = current_bar['Volume']
    current_vwap = current_bar['VWAP']
    current_rsi = current_bar['RSI']
    current_macd_hist = current_bar['Histogram']
    previous_macd_hist = previous_bar['Histogram']
    current_ema_50 = current_bar['EMA_50']
    current_volume_ma = current_bar['Volume_MA']
    atr_value = current_bar['ATR']

    volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 0

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    # --- NEW: Calculate additional indicators for signal quality ---
    momentum_score = calculate_momentum_score(df_cleaned)
    trend_strength, ema_slope = calculate_trend_strength(df_cleaned)

    # --- Check Cooldown ---
    if not check_signal_cooldown(symbol, timeframe, existing_signals):
        return []

    # --- Check Market Conditions (New) ---
    if not check_market_conditions(df_cleaned):
        return []

    signal_type = None

    # --- LONG Entry Criteria ---
    long_conditions = (
            current_close > current_vwap and
            current_rsi > 50 and current_rsi < ENHANCED_CONFIG['MAX_RSI_LONG'] and
            current_macd_hist > 0 and
            current_macd_hist > previous_macd_hist and
            volume_ratio > ENHANCED_CONFIG['MIN_VOLUME_RATIO'] and
            current_close > current_ema_50 and
            momentum_score > ENHANCED_CONFIG['MIN_MOMENTUM_SCORE'] and
            trend_strength > ENHANCED_CONFIG['MIN_TREND_STRENGTH'] and
            ema_slope > ENHANCED_CONFIG['MIN_EMA_SLOPE'] and
            current_close > (current_vwap * (1 + ENHANCED_CONFIG['STRONGER_VWAP_THRESHOLD']))
    )

    macd_hist_strength_ok = abs(current_macd_hist) > MIN_MACD_HIST_ABS

    if long_conditions and macd_hist_strength_ok:
        signal_type = "BUY"

    # --- SHORT Entry Criteria ---
    elif (
            current_close < current_vwap and
            current_rsi < 50 and current_rsi > ENHANCED_CONFIG['MIN_RSI_SHORT'] and
            current_macd_hist < 0 and
            current_macd_hist < previous_macd_hist and
            volume_ratio > ENHANCED_CONFIG['MIN_VOLUME_RATIO'] and
            current_close < current_ema_50 and
            momentum_score < -ENHANCED_CONFIG['MIN_MOMENTUM_SCORE'] and
            trend_strength < (1 - ENHANCED_CONFIG['MIN_TREND_STRENGTH']) and
            ema_slope < -ENHANCED_CONFIG['MIN_EMA_SLOPE'] and
            current_close < (current_vwap * (1 - ENHANCED_CONFIG['STRONGER_VWAP_THRESHOLD']))
    ) and macd_hist_strength_ok:
        signal_type = "SELL"

    if signal_type:
        simplified_signal_for_json = {
            "id": str(uuid.uuid4()),
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": signal_type,
            "date": signal_date,
            "time": signal_time,
            "entry_price": round(current_close, 6),
            "strategy_name": STRATEGY_NAME # Using the global STRATEGY_NAME constant
        }
        signals_to_log.append(simplified_signal_for_json)

    return signals_to_log

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
                    if df_with_indicators.empty or len(df_with_indicators.dropna(subset=required_cols)) < 2:
                        pbar.update(1)
                        continue

                    newly_generated_signals = generate_enhanced_signals(df_with_indicators, pair, timeframe, existing_signals_data)

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