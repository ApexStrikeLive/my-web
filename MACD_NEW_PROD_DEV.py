import os
import time
import math
import uuid
import json
from datetime import datetime, timedelta
import pandas as pd
import ccxt
import numpy as np
from tqdm import tqdm

# --- ANSI Color Codes ---
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"

# --- Configuration ---
# IMPORTANT: Replace with your actual API Key and Secret.
# For production, consider using environment variables for security.
API_KEY = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
API_SECRET = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'

# --- Exchange Initialization ---
try:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True, # Crucial for managing API calls
        'options': {
            'defaultType': 'spot',
            'recvWindow': 10000, # Increased recvWindow for potentially faster responses
            'adjustForTimeDifference': True, # Automatically synchronizes time
        },
    })
    exchange.load_time_difference() # Synchronize time (this is good to keep)
    exchange.load_markets() # Load markets to populate exchange.symbols
    print(f"Successfully connected to Binance client via CCXT.")
except Exception as e:
    print(f"{COLOR_RED}Error connecting to Binance client: {e}{COLOR_RESET}")
    exit()

# --- Strategy and File Management Constants ---
SIGNAL_LOG_FILE = 'MACD_OPT_DEV.json'
STRATEGY_NAME = "MACD_OPT_DEV"

# --- Timeframes and Data Limits ---
TIMEFRAMES = ['1h', '2h', '4h', '30m','15m']
CANDLE_LIMIT = 100 # Number of historical candles to fetch

# --- Coins to Exclude (Updated with problematic coins from your logs) ---
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
    # --- NEWLY ADDED FROM YOUR LATEST LOGS ---
    'COS/USDT',  # Causing connection errors
    'CYBER/USDT',# Causing connection errors
    'DGB/USDT',  # Causing connection errors
]

# --- Strategy Parameters (Only MACD related ones needed for calculation) ---
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# --- Core MACD Filter ---
# Minimum absolute value for MACD Histogram for a signal to be considered strong enough.
# If histogram is too close to zero, it often means very little momentum.
MIN_MACD_HIST_ABS = 0.0001 # Can be adjusted based on asset volatility and desired strictness

# --- Time-based Filtering (Only to avoid old candles) ---
# Max age for a candle's close time (5 minutes) to be considered 'current' for signal generation.
# This is a general data integrity check, not a strategy filter.
MAX_SIGNAL_AGE_SECONDS = 300

# Define required columns for MACD calculation and basic data checks
REQUIRED_INDICATOR_COLS = ['Close', 'MACD_Line', 'Signal_Line', 'Histogram']

# --- Scan Interval ---
SCAN_INTERVAL_MINUTES = 15
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

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

# --- Data Fetching and Indicator Calculation ---

def fetch_ohlcv_data(symbol: str, timeframe: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame | None:
    """
    Fetches OHLCV data for a given symbol and timeframe using CCXT.
    Returns a DataFrame with appropriate column names, or None if an error occurs.
    Includes retry logic for common CCXT errors.
    """
    retries = 3
    for i in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv: # Check if the list is empty
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                      inplace=True)
            return df
        except ccxt.RateLimitExceeded as e:
            tqdm.write(f"{COLOR_YELLOW}Rate limit hit for {symbol} {timeframe}. Retrying in {exchange.rateLimit / 1000}s...{COLOR_RESET}")
            time.sleep(exchange.rateLimit / 1000) # CCXT's built-in should handle most, but this is a fallback if an explicit exception is caught.
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            tqdm.write(f"{COLOR_YELLOW}Connection error for {symbol} {timeframe}: {e}. Retrying in 5s...{COLOR_RESET}")
            time.sleep(5)
        except Exception as e:
            tqdm.write(f"{COLOR_RED}Error fetching {symbol} {timeframe} (attempt {i+1}/{retries}): {e}{COLOR_RESET}")
            if i < retries - 1:
                time.sleep(2)
    return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates only MACD indicators for the given DataFrame.
    """
    if df.empty:
        return df.copy()

    df_copy = df.copy()

    # MACD Calculation
    # Need sufficient data for MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD for valid values
    if len(df_copy) >= MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD:
        ema_fast = df_copy['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False, min_periods=MACD_FAST_PERIOD).mean()
        ema_slow = df_copy['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False, min_periods=MACD_SLOW_PERIOD).mean()
        df_copy['MACD_Line'] = ema_fast - ema_slow
        df_copy['Signal_Line'] = df_copy['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False, min_periods=MACD_SIGNAL_PERIOD).mean()
        df_copy['Histogram'] = df_copy['MACD_Line'] - df_copy['Signal_Line']
    else:
        df_copy['MACD_Line'] = np.nan
        df_copy['Signal_Line'] = np.nan
        df_copy['Histogram'] = np.nan

    return df_copy

# --- Signal Generation Logic (MACD Histogram Reversal with Two-Bar Confirmation) ---

def generate_pure_macd_signals(df: pd.DataFrame, symbol: str, timeframe: str, current_scan_time: datetime) -> list[dict]:
    """
    Generates trading signals based on MACD Histogram reversal with two-bar confirmation.
    Looks for a clear peak/trough in the histogram, followed by two bars confirming the reversal.
    All other confirmation indicators are removed.
    """
    signals_to_log = []

    # Minimum bars required for MACD Histogram calculation and comparison (now need 4 for 2-bar confirmation)
    min_required_bars = max(MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD, 4)

    if df.empty or len(df) < min_required_bars:
        return signals_to_log

    # Drop rows with NaN values for the required MACD columns
    df_cleaned = df.dropna(subset=REQUIRED_INDICATOR_COLS).iloc[:]

    if df_cleaned.empty or len(df_cleaned) < 4: # Ensure at least 4 valid bars after dropping NaNs
        return signals_to_log

    # Get the latest bars for comparison (from most recent to oldest)
    current_bar = df_cleaned.iloc[-1]
    previous_bar = df_cleaned.iloc[-2]
    two_bars_ago = df_cleaned.iloc[-3]
    three_bars_ago = df_cleaned.iloc[-4] # Need one more historical bar

    # --- Crucial Check: Ensure the signal is from the most recent completed candle ---
    # This is a data integrity check to ensure we're not using stale data.
    if (current_scan_time - current_bar.name).total_seconds() > MAX_SIGNAL_AGE_SECONDS:
        return signals_to_log

    # Validate that critical MACD values are not NaN before proceeding
    if pd.isna(current_bar.get('Histogram')) or \
       pd.isna(previous_bar.get('Histogram')) or \
       pd.isna(two_bars_ago.get('Histogram')) or \
       pd.isna(three_bars_ago.get('Histogram')):
        return signals_to_log

    current_close = current_bar['Close']
    current_macd_hist = current_bar['Histogram']
    previous_macd_hist = previous_bar['Histogram']
    two_bars_ago_macd_hist = two_bars_ago['Histogram']
    three_bars_ago_macd_hist = three_bars_ago['Histogram']

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    signal_type = None

    # --- MACD Histogram Long Signal (Two-Bar Bottom Confirmation) ---
    # Pattern: three_bars_ago -> two_bars_ago (trough) -> previous (rising 1) -> current (rising 2)
    # Trough should ideally be in negative territory for a long signal.
    macd_hist_long_reversal = (
        # Ensure we are below zero or very close to it for a potential bottom
        two_bars_ago_macd_hist <= 0 and
        # 'two_bars_ago' is a local trough (lowest among the four)
        two_bars_ago_macd_hist < three_bars_ago_macd_hist and
        two_bars_ago_macd_hist < previous_macd_hist and
        # 'previous_bar' starts rising from the trough
        previous_macd_hist > two_bars_ago_macd_hist and
        # 'current_bar' confirms the rise (second rising bar)
        current_macd_hist > previous_macd_hist
    )

    # --- MACD Histogram Short Signal (Two-Bar Top Confirmation) ---
    # Pattern: three_bars_ago -> two_bars_ago (peak) -> previous (falling 1) -> current (falling 2)
    # Peak should ideally be in positive territory for a short signal.
    macd_hist_short_reversal = (
        # Ensure we are above zero or very close to it for a potential top
        two_bars_ago_macd_hist >= 0 and
        # 'two_bars_ago' is a local peak (highest among the four)
        two_bars_ago_macd_hist > three_bars_ago_macd_hist and
        two_bars_ago_macd_hist > previous_macd_hist and
        # 'previous_bar' starts falling from the peak
        previous_macd_hist < two_bars_ago_macd_hist and
        # 'current_bar' confirms the fall (second falling bar)
        current_macd_hist < previous_macd_hist
    )


    # --- Core MACD Histogram Strength Check ---
    # Signal should only be considered if the current histogram has sufficient absolute value.
    macd_hist_strength_ok = abs(current_macd_hist) >= MIN_MACD_HIST_ABS

    # --- Signal Generation ---
    if macd_hist_long_reversal and macd_hist_strength_ok:
        signal_type = "BUY"
    elif macd_hist_short_reversal and macd_hist_strength_ok:
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
            "strategy_name": STRATEGY_NAME
        }
        signals_to_log.append(simplified_signal_for_json)

    return signals_to_log

# --- JSON File Management Functions ---

def load_existing_signals(filename: str = SIGNAL_LOG_FILE) -> list[dict]:
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

def save_signals_to_file(new_signals_batch: list[dict], filename: str = SIGNAL_LOG_FILE):
    """
    Appends a batch of new unique signals to the existing JSON file.
    Uniqueness is determined by symbol, timeframe, signal, date, time, and strategy_name.
    """
    if not new_signals_batch: # Nothing to save
        return 0

    existing_signals = load_existing_signals(filename)

    added_count = 0
    existing_signal_keys = set()
    for s in existing_signals:
        if isinstance(s, dict) and all(k in s for k in ['symbol', 'timeframe', 'signal', 'date', 'time', 'strategy_name']):
            existing_signal_keys.add(
                (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name'])
            )

    signals_to_append = []
    for signal in new_signals_batch:
        signal_key = (
            signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
            signal.get('date'), signal.get('time'), signal.get('strategy_name')
        )
        if signal_key not in existing_signal_keys:
            signals_to_append.append(signal)
            added_count += 1

    if signals_to_append:
        existing_signals.extend(signals_to_append)
        existing_signals.sort(key=lambda x: (x.get('date', ''), x.get('time', '')), reverse=True)
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
        f"SIGNAL: {signal_color}{signal['signal']}{COLOR_RESET} | PAIR: {signal['symbol']} | TF: {signal['timeframe']} | PRICE: {signal['entry_price']:.6f} | TIME: {signal['date']} {signal['time']}")

# --- Main Scanning Logic ---

def monitor_signals():
    """
    Main loop for monitoring trading signals, fetching data,
    calculating indicators, and saving signals to a JSON file.
    """
    print(f"Starting Binance Trading Signal Scanner ({STRATEGY_NAME})...")

    print("Loading tradable symbols from Binance...")
    try:
        all_usdt_pairs = [
            s for s in exchange.symbols
            if s.endswith('/USDT') and
            exchange.markets[s]['active'] and # Ensure market is active
            exchange.markets[s]['spot'] # Ensure it's a spot market
        ]
        all_usdt_pairs = [pair for pair in all_usdt_pairs if pair not in COINS_TO_EXCLUDE]
        print(f"Found {len(all_usdt_pairs)} USDT spot pairs to scan (after exclusions).")
    except Exception as e:
        print(f"{COLOR_RED}Error loading markets or filtering symbols: {e}. Exiting.{COLOR_RESET}")
        exit()

    # Pre-calculate min_required_bars for efficiency
    min_bars_for_strategy = max(MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD, 4) # Ensure at least 4 bars for the new logic

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        current_scan_start_time = datetime.now()
        print(f"\n--- Starting new scan at {current_scan_start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")

        # Create a list to hold all new signals found in this scan cycle
        new_signals_this_cycle = []

        # Load existing signals once per scan cycle for uniqueness checks
        existing_signals_data = load_existing_signals(SIGNAL_LOG_FILE)
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
                    pbar.set_description(f"Scanning: {pair} {timeframe}")

                    df = fetch_ohlcv_data(pair, timeframe)
                    if df is None or df.empty:
                        pbar.update(1)
                        continue

                    df_with_indicators = calculate_indicators(df)

                    # Check for sufficient data after indicator calculation and NaN removal
                    if len(df_with_indicators) < min_bars_for_strategy or \
                       df_with_indicators.iloc[-min_bars_for_strategy:].dropna(subset=REQUIRED_INDICATOR_COLS).shape[0] < min_bars_for_strategy:
                        pbar.update(1)
                        continue

                    # Generate signals using the pure MACD logic with 2-bar confirmation
                    newly_generated_signals_for_pair_tf = generate_pure_macd_signals(
                        df_with_indicators, pair, timeframe, current_scan_start_time
                    )

                    if newly_generated_signals_for_pair_tf:
                        for signal in newly_generated_signals_for_pair_tf:
                            signal_key = (
                                signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
                                signal.get('date'), signal.get('time'), signal.get('strategy_name')
                            )
                            if signal_key not in existing_signal_keys:
                                # Add to the batch for this scan cycle, don't save individually yet
                                new_signals_this_cycle.append(signal)
                                # Add to the in-memory set to prevent duplicates within this cycle
                                existing_signal_keys.add(signal_key)
                                print_signal_to_console(signal) # Print immediately for user feedback

                    pbar.update(1)
                    # IMPORTANT: Removed the manual time.sleep here. Rely on CCXT's rate limiting.

        # After scanning all pairs and timeframes, save all new signals found in this cycle
        if new_signals_this_cycle:
            total_saved_signals_this_scan = save_signals_to_file(new_signals_this_cycle, SIGNAL_LOG_FILE)
            print(f"\n{COLOR_CYAN}Saved {total_saved_signals_this_scan} new unique signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
        else:
            print(f"\nNo new signals found in this scan.")

        print(f"\nScan finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)

# --- Run the script ---
if __name__ == "__main__":
    monitor_signals()