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
STRATEGY_NAME = "SQZ_OPT_TEST"  # Defined globally
SIGNAL_LOG_FILE = 'SQZ_OPT_TEST.json'

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
    'ERN/USDT',  # Explicitly exclude ERN/USDT as its price isn't changing and was flagged
    'ATOM/USDT',  # Causing fetching errors
    'SUSD/USDT',  # Causing fetching errors
    'AION/USDT',  # Causing fetching errors
    'HIVE/USDT',  # Causing fetching errors
    'LINA/USDT',  # Causing fetching errors
    'MDT/USDT',  # Causing fetching errors
    'MDX/USDT',  # Causing fetching errors
    # BTC/USDT is a major pair and should ideally not cause errors.
    # If BTC/USDT fetching errors persist, it's usually a rate limit or a temporary Binance issue.
    # We will not exclude BTC/USDT by default, but you can add it if absolutely necessary.
]

# --- Squeeze Momentum Strategy Parameters ---
BB_PERIOD = 20
BB_STD_DEV = 2.0
KC_PERIOD = 20
KC_MULTIPLIER = 1.5
LR_PERIOD = 20
SMA_PERIOD = 20

# Scheduling
SCAN_INTERVAL_MINUTES = 15
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

# --- ENHANCED CONFIGURATION PARAMETERS (New) ---
ENHANCED_CONFIG = {
    'SIGNAL_COOLDOWN_HOURS': 4,
    'MIN_VOLUME_RATIO': 1.2,
    'ATR_PERIOD': 20,
    'MIN_MOMENTUM_THRESHOLD': 0.001,
}

# Define required columns globally so it's accessible everywhere needed
required_cols = ['Close', 'High', 'Low', 'Volume', 'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'SMA_20',
                 'LR_Momentum', 'Squeeze_On', 'ATR']

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
    Calculates Squeeze Momentum indicators: BB, KC, Linear Regression Momentum, Squeeze detection
    """
    if df.empty or len(df) < max(BB_PERIOD, KC_PERIOD, LR_PERIOD):
        return df.copy()

    df_copy = df.copy()

    # Simple Moving Average (20-period)
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=SMA_PERIOD).mean()

    # Bollinger Bands (20-period, 2 std dev)
    std_dev = df_copy['Close'].rolling(window=BB_PERIOD).std()
    df_copy['BB_Upper'] = df_copy['SMA_20'] + (BB_STD_DEV * std_dev)
    df_copy['BB_Lower'] = df_copy['SMA_20'] - (BB_STD_DEV * std_dev)

    # EMA for Keltner Channels
    df_copy['EMA_20'] = df_copy['Close'].ewm(span=KC_PERIOD).mean()

    # ATR Calculation (already exists, but update period)
    df_copy['ATR'] = calculate_atr(df_copy, period=ENHANCED_CONFIG['ATR_PERIOD'])

    # Keltner Channels (20-period EMA, 1.5 ATR)
    df_copy['KC_Upper'] = df_copy['EMA_20'] + (KC_MULTIPLIER * df_copy['ATR'])
    df_copy['KC_Lower'] = df_copy['EMA_20'] - (KC_MULTIPLIER * df_copy['ATR'])

    # Squeeze Detection (BB inside KC)
    df_copy['Squeeze_On'] = (df_copy['BB_Upper'] <= df_copy['KC_Upper']) & (df_copy['BB_Lower'] >= df_copy['KC_Lower'])

    # Linear Regression Momentum
    def calculate_lr_momentum(series, period):
        momentum = []
        for i in range(len(series)):
            if i < period - 1:
                momentum.append(np.nan)
            else:
                y_values = series.iloc[i - period + 1:i + 1].values
                x_values = np.arange(period)
                if len(y_values) == period:
                    slope = np.polyfit(x_values, y_values, 1)[0]
                    momentum.append(slope)
                else:
                    momentum.append(np.nan)
        return pd.Series(momentum, index=series.index)

    df_copy['LR_Momentum'] = calculate_lr_momentum(df_copy['Close'], LR_PERIOD)

    # Basic volume moving average for filtering
    df_copy['Volume_MA'] = df_copy['Volume'].rolling(window=20).mean()

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
                                key=lambda x: datetime.datetime.strptime(
                                    f"{x.get('date', '1970-01-01')} {x.get('time', '00:00:00')}", "%Y-%m-%d %H:%M:%S"))
        last_signal_time = datetime.datetime.strptime(
            f"{last_signal_entry.get('date', '1970-01-01')} {last_signal_entry.get('time', '00:00:00')}",
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return True

    time_diff = datetime.datetime.now() - last_signal_time
    return time_diff.total_seconds() >= ENHANCED_CONFIG['SIGNAL_COOLDOWN_HOURS'] * 3600


def check_market_conditions(df):
    """Simplified market conditions check for Squeeze Momentum"""
    if len(df) < 20:
        return False

    # Basic volatility check only
    if 'ATR' not in df.columns or pd.isna(df['ATR'].iloc[-1]):
        return False

    return True


# --- Squeeze Momentum Signal Generation ---

def generate_enhanced_signals(df: pd.DataFrame, symbol: str, timeframe: str, existing_signals: list[dict]) -> list[
    dict]:
    """Generates trading signals based on Pure Squeeze Momentum Strategy with directional turn logic."""
    signals_to_log = []
    min_required_bars = max(BB_PERIOD, KC_PERIOD, LR_PERIOD) + 5
    if df.empty or len(df) < min_required_bars:
        return signals_to_log

    df_cleaned = df.dropna(subset=required_cols).copy()

    if df_cleaned.empty or len(df_cleaned) < 2:
        return signals_to_log

    current_bar = df_cleaned.iloc[-1]
    previous_bar = df_cleaned.iloc[-2]

    # Check for required data
    required_current = ['Close', 'Volume', 'LR_Momentum', 'Squeeze_On', 'Volume_MA']
    if any(pd.isna(current_bar[col]) for col in required_current) or pd.isna(previous_bar['LR_Momentum']):
        return signals_to_log
    if pd.isna(previous_bar['Squeeze_On']):
        return signals_to_log

    current_close = current_bar['Close']
    current_volume = current_bar['Volume']
    current_momentum = current_bar['LR_Momentum']
    previous_momentum = previous_bar['LR_Momentum']  # We need this for turn detection
    current_squeeze = current_bar['Squeeze_On']
    previous_squeeze = previous_bar['Squeeze_On']
    volume_ma = current_bar['Volume_MA']

    # Basic volume filter
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    # Check Cooldown
    if not check_signal_cooldown(symbol, timeframe, existing_signals):
        return []

    signal_type = None

    # SQUEEZE MOMENTUM SIGNAL LOGIC with Directional Turn
    squeeze_release = previous_squeeze == True and current_squeeze == False

    if squeeze_release:
        # BUY Signal: Squeeze OFF + Positive Momentum Turn
        if (current_momentum > ENHANCED_CONFIG['MIN_MOMENTUM_THRESHOLD'] and
                current_momentum > previous_momentum and  # This is the "bottom of bottom" turn check
                volume_ratio > ENHANCED_CONFIG['MIN_VOLUME_RATIO']):
            signal_type = "BUY"

        # SELL Signal: Squeeze OFF + Negative Momentum Turn
        elif (current_momentum < -ENHANCED_CONFIG['MIN_MOMENTUM_THRESHOLD'] and
              current_momentum < previous_momentum and  # This is the "top of top" turn check
              volume_ratio > ENHANCED_CONFIG['MIN_VOLUME_RATIO']):
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
            "strategy_name": STRATEGY_NAME,
            "momentum": round(current_momentum, 6),
            "momentum_change": round(current_momentum - previous_momentum, 6),  # Add momentum change
            "squeeze_release": True
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
                tqdm.write(
                    f"{COLOR_YELLOW}Warning: {filename} has unexpected structure. Returning empty list.{COLOR_RESET}")
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
        if isinstance(s, dict) and all(
                k in s for k in ['symbol', 'timeframe', 'signal', 'date', 'time', 'strategy_name']):
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
            if isinstance(s, dict) and all(
                    k in s for k in ['symbol', 'timeframe', 'signal', 'date', 'time', 'strategy_name']):
                existing_signal_keys.add(
                    (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name'])
                )

        current_scan_signal_keys = set()

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

                    newly_generated_signals = generate_enhanced_signals(df_with_indicators, pair, timeframe,
                                                                        existing_signals_data)

                    for signal in newly_generated_signals:
                        signal_key = (
                            signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
                            signal.get('date'), signal.get('time'), signal.get('strategy_name')
                        )

                        if signal_key not in existing_signal_keys and signal_key not in current_scan_signal_keys:
                            signals_this_scan.append(signal)
                            existing_signals_data.append(signal)
                            existing_signal_keys.add(signal_key)
                            current_scan_signal_keys.add(signal_key)
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