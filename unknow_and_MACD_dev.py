import os
import time
import math
import uuid
import json
from datetime import datetime
import pandas as pd
import ccxt
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from prettytable import PrettyTable

# --- ANSI Color Codes ---
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"

# --- Configuration ---
# IMPORTANT: Double-check these are EXACTLY correct from your Binance API management.
# For production, consider using environment variables for security.
API_KEY = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
API_SECRET = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'  # Placeholder

# Define the list of timeframes to scan
TIMEFRAMES = ['1h', '2h', '4h', '30m']  # New: Multiple timeframes
CANDLE_LIMIT = 250  # Enough for all calculations (26 for MACD, 50 for EMA, 24 for VWAP)

# --- Merged Exclusion List ---
# Combine and ensure unique entries, using BASE/QUOTE format for consistency
COINS_TO_EXCLUDE_S1 = [
    'BZRXUSDT', 'AAVEUPUSDT', 'LENDUSDT',
    'USDCUSDT', 'BUSDUSDT', 'DAIUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'PYUSDUSDT', 'USDPUSDT', 'GUSDUSDT', 'USTCUSDT',
    'MIMUSDT',
    'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT',
    'ETHUPUSDT', 'ETHDOWNUSDT', 'BTCUPUSDT', 'BTCDOWNUSDT', 'BNBUPUSDT', 'BNBDOWNUSDT',
    'TRBUSDT', 'LUNCUSDT', 'ETCUSDT',
]
COINS_TO_EXCLUDE_S2 = [
    'BZRX/USDT', 'AAVEUP/USDT', 'LEND/USDT',
    'USDC/USDT', 'BUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'TUSD/USDT',
    'PYUSD/USDT', 'USDP/USDT', 'GUSD/USDT', 'USTC/USDT', 'MIM/USDT',
    'EURT/USDT', 'BKRW/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'ETHUP/USDT', 'ETHDOWN/USDT', 'BTCUP/USDT', 'BTCDOWN/USDT', 'BNBUP/USDT',
    'BNBDOWN/USDT',
    'ETC/USDT', 'XEC/USDT', 'LUNC/USDT', 'TRB/USDT', 'PERP/USDT',
    'FTM/USDT', 'NEAR/USDT', 'ICP/USDT', 'RNDR/USDT', 'INJ/USDT',
    'TIA/USDT', 'ALT/USDT', 'SEI/USDT', 'PYTH/USDT', 'JUP/USDT',
    'CITY/USDT', 'PSG/USDT', 'LAZIO/USDT',
]

# Convert S1 list to BASE/QUOTE format and combine with S2 list for a unified set
ALL_COINS_TO_EXCLUDE_FORMATTED = set(s.replace('USDT', '/USDT') for s in COINS_TO_EXCLUDE_S1)
ALL_COINS_TO_EXCLUDE_FORMATTED.update(COINS_TO_EXCLUDE_S2)
COINS_TO_EXCLUDE_FINAL = list(ALL_COINS_TO_EXCLUDE_FORMATTED)

# --- Strategy Parameters (Combined from both codes) ---
# VWAP period is 24
# RSI period is 14
# MACD periods 12, 26, 9
# EMA 50

# Strategy 2 Parameters
MACD_FAST_PERIOD_S2 = 12
MACD_SLOW_PERIOD_S2 = 26
MACD_SIGNAL_PERIOD_S2 = 9
RSI_PERIOD_S2 = 14  # Same as S1

HIST_PEAK_TROUGH_LOOKBACK = 5  # How many bars back to look for the peak/trough
REVERSAL_CONFIRMATION_BARS = 2  # NEW: Number of bars *after* the peak/trough that must confirm the reversal direction

RSI_LONG_OVERSOLD_MAX = 40
RSI_SHORT_OVERBOUGHT_MIN = 60
MIN_VOLUME_RATIO_S2 = 0.8  # Renamed to avoid clash, as S1 also has volume ratio filter

# --- File Names & Strategy Info ---
SIGNAL_LOG_FILE = 'merged_crypto_signals.json'
STRATEGY_NAME = "Combined_Binance_MACD_OPT_PROD"
# --- Scan Interval (from Code 1, 45 minutes) ---
SCAN_INTERVAL_MINUTES = 30
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

# --- Initialize CCXT Exchange (Focus on Connection Method) ---
# This block is crucial for successful connection with CCXT
try:
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            # Important for avoiding "Timestamp for this request is outside of the recvWindow"
            'adjustForTimeDifference': True,
        },
        'recvWindow': 10000  # Increased recvWindow to 10 seconds (10000 ms)
    })

    print("Attempting to synchronize time with Binance...")
    # It's better to call load_time_difference() and then check for the attribute
    # or just trust 'adjustForTimeDifference': True if it doesn't need to be printed.
    exchange.load_time_difference()  # This call is what populates the time_difference attribute

    # Now, check if the attribute exists before trying to print it
    if hasattr(exchange, 'time_difference'):
        print(f"Time difference adjusted: {exchange.time_difference} ms")
    else:
        print(
            f"{COLOR_YELLOW}Warning: 'time_difference' attribute not directly available after load_time_difference(). Time synchronization should still be active via 'adjustForTimeDifference'.{COLOR_RESET}")

    print("Successfully connected to Binance client via CCXT.")

except ccxt.ExchangeError as e:
    # Catch specific CCXT exchange errors (e.g., API key, permissions, recvWindow)
    print(f"{COLOR_RED}Error connecting to Binance client (CCXT ExchangeError): {e}{COLOR_RESET}")
    if "Signature for this request is not valid" in str(e):
        print(
            f"{COLOR_YELLOW}Hint: Double-check your API_KEY and API_SECRET. Ensure they are correct and have 'Enable Reading' permission.{COLOR_RESET}")
    elif "Timestamp for this request is outside of the recvWindow" in str(e):
        print(
            f"{COLOR_YELLOW}Hint: Your system's time might be out of sync. Please synchronize your computer's clock with an NTP server.{COLOR_RESET}")
    import sys

    sys.exit("Exiting due to Binance connection error.")
except Exception as e:
    # Catch any other unexpected errors during connection
    print(f"{COLOR_RED}An unexpected error occurred during Binance connection: {e}{COLOR_RESET}")
    import sys

    sys.exit("Exiting due to unexpected connection error.")


# --- UI Helper Functions ---
def countdown_timer(seconds, total_seconds, message_prefix="Next scan in"):
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
        seconds -= 1  # Decrement seconds inside the loop
    print("\n", end="", flush=True)


# --- Data Fetching ---
def fetch_ohlcv(symbol, interval, limit):
    """
    Pobiera dane OHLCV dla danego symbolu (w formacie BASE/QUOTE), interwału i limitu using CCXT.
    Returns DataFrame with columns 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error fetching {symbol} {interval}: {e}{COLOR_RESET}")
        return None


def get_all_usdt_pairs():
    """
    Pobiera wszystkie aktywne pary handlowe z Binance, które kończą się na 'USDT'.
    Zwraca symbole w formacie 'BASE/QUOTE' (np. 'BTC/USDT').
    """
    try:
        markets = exchange.load_markets()
        usdt_pairs = [
            symbol for symbol, market_data in markets.items()
            if market_data['quote'] == 'USDT' and market_data['active'] and market_data['spot']
        ]
        return usdt_pairs
    except Exception as e:
        # Improved error message for getting pairs
        tqdm.write(f"{COLOR_RED}Error loading markets from Binance: {e}{COLOR_RESET}")
        return []


# --- Indicator Calculation (Combined and Aligned) ---

def calculate_vwap(df):
    """
    Oblicza VWAP dla ostatnich 24 okresów. (From Strategy 1)
    """
    if len(df) < 24:
        return None
    df_vwap = df.tail(24).copy()
    df_vwap['typical_price'] = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3
    df_vwap['price_volume'] = df_vwap['typical_price'] * df_vwap['volume']
    if df_vwap['volume'].sum() == 0:
        return 0  # If no volume, VWAP is undefined, return 0 or handle as None
    vwap = df_vwap['price_volume'].sum() / df_vwap['volume'].sum()
    return vwap


def calculate_rsi(series, period=14):
    """
    Oblicza wskaźnik RSI. (Shared logic, periods can differ)
    """
    if len(series) < period * 2:  # Ensure enough data for initial average calculation
        return np.nan  # Use np.nan for consistency with pandas
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero for RS
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_macd(series, fast_period, slow_period, signal_period):
    """
    Oblicza wskaźnik MACD, linię sygnału i histogram. (Used by both, parameters passed)
    """
    if len(series) < max(fast_period, slow_period, signal_period) + 1:
        return None

    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    if len(macd_histogram) < 2:  # Need at least current and previous for some checks
        return None

    return {
        'macd_line': macd_line,  # Return full series for S2 checks
        'signal_line': signal_line,  # Return full series for S2 checks
        'macd_histogram': macd_histogram,  # Return full series for S2 checks
    }


def calculate_volume_ratio(series, period=20):
    """
    Oblicza stosunek bieżącego wolumenu do 20-okresowej średniej kroczącej wolumenu. (From Strategy 1)
    """
    if len(series) < period:
        return None
    volume_ma = series.rolling(window=period).mean().iloc[-1]
    if volume_ma == 0:
        return 0
    current_volume = series.iloc[-1]
    return current_volume / volume_ma


def calculate_ema(series, period):
    """
    Oblicza wykładniczą średnią kroczącą (EMA). (From Strategy 1)
    """
    if len(series) < period:
        return None
    return series.ewm(span=period, adjust=False).mean().iloc[-1]


# --- Combined Signal Logic ---

def check_combined_signal(pair, data, timeframe):  # Added timeframe parameter
    """
    Checks if conditions for BOTH strategies are met for LONG or SHORT signals.
    """
    # Ensure enough data for all calculations, including MACD history for S2
    min_required_s2_bars = max(MACD_SLOW_PERIOD_S2 + MACD_SIGNAL_PERIOD_S2,
                               HIST_PEAK_TROUGH_LOOKBACK + REVERSAL_CONFIRMATION_BARS + 2)
    if data is None or data.empty or len(data) < CANDLE_LIMIT or len(data) < min_required_s2_bars:
        # print(f"Not enough data for {pair} on {timeframe}. Required {min_required_s2_bars}, got {len(data)}") # Debugging line
        return None

    # Calculate indicators for Strategy 1
    current_price = data['close'].iloc[-1]
    vwap = calculate_vwap(data)
    rsi_s1 = calculate_rsi(data['close'], period=14)
    macd_data_s1_raw = calculate_macd(data['close'], 12, 26, 9)  # Returns full series
    volume_ratio_s1 = calculate_volume_ratio(data['volume'], period=20)
    ema_50_1h = calculate_ema(data['close'], 50)

    # Pre-check S1 basic requirements (filters)
    if vwap is None or rsi_s1 is None or macd_data_s1_raw is None or volume_ratio_s1 is None or ema_50_1h is None:
        # print(f"S1 indicator is None for {pair} on {timeframe}") # Debugging line
        return None

    macd_histogram_s1 = macd_data_s1_raw['macd_histogram'].iloc[-1]
    previous_macd_histogram_s1 = macd_data_s1_raw['macd_histogram'].iloc[-2]  # Access previous from series

    # Apply S1 initial filters
    vwap_distance = abs(current_price - vwap) / vwap * 100
    if vwap_distance < 0.2:
        return None
    if not (rsi_s1 < 45 or rsi_s1 > 55):  # RSI must be in extreme zones for S1, not middle
        return None
    if abs(macd_histogram_s1) < 0.01:
        return None
    if volume_ratio_s1 < 1.5:
        return None

    # Calculate indicators for Strategy 2
    df_s2 = data.copy()

    macd_data_s2 = calculate_macd(df_s2['close'], MACD_FAST_PERIOD_S2, MACD_SLOW_PERIOD_S2, MACD_SIGNAL_PERIOD_S2)
    if macd_data_s2 is None:  # Not enough data for MACD S2
        return None

    df_s2['MACD_Line'] = macd_data_s2['macd_line']
    df_s2['Signal_Line'] = macd_data_s2['signal_line']
    df_s2['Histogram'] = macd_data_s2['macd_histogram']
    df_s2['RSI'] = calculate_rsi(df_s2['close'], period=RSI_PERIOD_S2)  # Calculate RSI for S2 directly on this df

    # Drop NaNs after all calculations to ensure clean data for slicing
    df_cleaned_s2 = df_s2.dropna(subset=['MACD_Line', 'Signal_Line', 'Histogram', 'RSI', 'close', 'volume']).copy()

    # Re-check data length after dropping NaNs for S2 specific conditions
    if df_cleaned_s2.empty or len(df_cleaned_s2) < min_required_s2_bars:
        # print(f"S2 data too short after cleanup for {pair} on {timeframe}. Length: {len(df_cleaned_s2)}") # Debugging line
        return None

    # Ensure we have enough data for lookback + confirmation bars
    if len(df_cleaned_s2) < HIST_PEAK_TROUGH_LOOKBACK + REVERSAL_CONFIRMATION_BARS + 1:
        # print(f"Not enough data for S2 peak/trough reversal logic for {pair} on {timeframe}.")
        return None

    i_s2 = len(df_cleaned_s2) - 1  # Index of the current bar
    # Adjusted window for peak/trough search
    hist_window_s2 = df_cleaned_s2['Histogram'].iloc[max(0,
                                                         i_s2 - HIST_PEAK_TROUGH_LOOKBACK):i_s2 + 1]  # Include current bar in lookback for peak/trough

    if hist_window_s2.empty:
        return None

    current_bar_s2 = df_cleaned_s2.iloc[i_s2]
    current_hist_s2 = current_bar_s2['Histogram']
    current_rsi_s2 = current_bar_s2['RSI']
    current_close_s2 = current_bar_s2['close']  # Use 'close' as column name
    current_volume_s2 = current_bar_s2['volume']  # Use 'volume' as column name

    # Calculate average volume for S2's volume ratio, considering the lookback window
    avg_volume_s2 = df_cleaned_s2['volume'].iloc[
                    max(0, i_s2 - HIST_PEAK_TROUGH_LOOKBACK):i_s2].mean()  # Average volume BEFORE current bar
    volume_ratio_s2 = current_volume_s2 / avg_volume_s2 if avg_volume_s2 > 0 else 0
    price_volatility_s2 = df_cleaned_s2['close'].iloc[max(0, i_s2 - HIST_PEAK_TROUGH_LOOKBACK):i_s2].std() / \
                          df_cleaned_s2['close'].iloc[max(0, i_s2 - HIST_PEAK_TROUGH_LOOKBACK):i_s2].mean() if \
        df_cleaned_s2['close'].iloc[max(0, i_s2 - HIST_PEAK_TROUGH_LOOKBACK):i_s2].mean() > 0 else 0

    signal_type = None

    # --- Combined LONG Conditions ---
    long_condition_s1 = (
            current_price > vwap and
            rsi_s1 > 50 and  # From S1 logic, not necessarily oversold
            macd_histogram_s1 > 0 and
            macd_histogram_s1 > previous_macd_histogram_s1 and  # MACD Hist growing
            volume_ratio_s1 > 1.5 and
            current_price > ema_50_1h
    )

    long_condition_s2 = False
    # Find the trough (lowest negative point or lowest point if all positive)
    potential_troughs = hist_window_s2[hist_window_s2 <= 0]  # Consider non-positive histograms for a trough
    if not potential_troughs.empty:
        trough_val_s2 = potential_troughs.min()
        trough_idx_s2_ts = potential_troughs.idxmin()  # Get timestamp index
        trough_pos_in_df_s2 = df_cleaned_s2.index.get_indexer([trough_idx_s2_ts])[0]

        # Ensure there are enough bars *after* the trough for confirmation
        if i_s2 - trough_pos_in_df_s2 >= REVERSAL_CONFIRMATION_BARS:
            # Check the 'REVERSAL_CONFIRMATION_BARS' bars *after* the trough, up to and including current bar
            confirmation_segment = df_cleaned_s2['Histogram'].iloc[
                                   trough_pos_in_df_s2 + 1: i_s2 + 1]  # From bar AFTER trough to current

            # Check if all bars in the confirmation segment are increasing
            is_momentum_up_s2 = (confirmation_segment.diff().dropna() > 0).all()

            # Additional checks for S2
            hist_rising_from_trough_s2 = current_hist_s2 > trough_val_s2  # Current histogram is higher than the trough
            rsi_oversold_s2 = (current_rsi_s2 <= RSI_LONG_OVERSOLD_MAX)
            volume_ok_s2 = (volume_ratio_s2 >= MIN_VOLUME_RATIO_S2)

            # Compare current close to close at trough
            price_rising_s2 = current_close_s2 > df_cleaned_s2['close'].iloc[trough_pos_in_df_s2]

            volatility_ok_s2 = price_volatility_s2 > 0.01

            macd_line_recent_s2 = df_cleaned_s2['MACD_Line'].iloc[-3:]
            macd_trending_up_s2 = False
            if len(macd_line_recent_s2) == 3 and \
                    (macd_line_recent_s2.iloc[2] > macd_line_recent_s2.iloc[1] and
                     macd_line_recent_s2.iloc[1] > macd_line_recent_s2.iloc[0]):
                macd_trending_up_s2 = True

            long_condition_s2 = (
                    is_momentum_up_s2 and hist_rising_from_trough_s2 and rsi_oversold_s2 and
                    volume_ok_s2 and price_rising_s2 and volatility_ok_s2 and macd_trending_up_s2
            )
            # print(f"LONG S2 for {pair} on {timeframe}: {is_momentum_up_s2}, {hist_rising_from_trough_s2}, {rsi_oversold_s2}, {volume_ok_s2}, {price_rising_s2}, {volatility_ok_s2}, {macd_trending_up_s2}") # Debugging

    if long_condition_s1 and long_condition_s2:
        signal_type = 'BUY'

    # --- Combined SHORT Conditions ---
    short_condition_s1 = (
            current_price < vwap and
            rsi_s1 < 50 and  # From S1 logic, not necessarily overbought
            macd_histogram_s1 < 0 and
            macd_histogram_s1 < previous_macd_histogram_s1 and  # MACD Hist falling
            volume_ratio_s1 > 1.5 and
            current_price < ema_50_1h
    )

    short_condition_s2 = False
    # Find the peak (highest positive point or highest point if all negative)
    potential_peaks = hist_window_s2[hist_window_s2 >= 0]  # Consider non-negative histograms for a peak
    if not potential_peaks.empty:
        peak_val_s2 = potential_peaks.max()
        peak_idx_s2_ts = potential_peaks.idxmax()  # Get timestamp index
        peak_pos_in_df_s2 = df_cleaned_s2.index.get_indexer([peak_idx_s2_ts])[0]

        # Ensure there are enough bars *after* the peak for confirmation
        if i_s2 - peak_pos_in_df_s2 >= REVERSAL_CONFIRMATION_BARS:
            # Check the 'REVERSAL_CONFIRMATION_BARS' bars *after* the peak, up to and including current bar
            confirmation_segment = df_cleaned_s2['Histogram'].iloc[
                                   peak_pos_in_df_s2 + 1: i_s2 + 1]  # From bar AFTER peak to current

            # Check if all bars in the confirmation segment are decreasing
            is_momentum_down_s2 = (confirmation_segment.diff().dropna() < 0).all()

            # Additional checks for S2
            hist_falling_from_peak_s2 = current_hist_s2 < peak_val_s2  # Current histogram is lower than the peak
            rsi_overbought_s2 = (current_rsi_s2 >= RSI_SHORT_OVERBOUGHT_MIN)
            volume_ok_s2 = (volume_ratio_s2 >= MIN_VOLUME_RATIO_S2)

            # Compare current close to close at peak
            price_falling_s2 = current_close_s2 < df_cleaned_s2['close'].iloc[peak_pos_in_df_s2]

            volatility_ok_s2 = price_volatility_s2 > 0.01

            macd_line_recent_s2 = df_cleaned_s2['MACD_Line'].iloc[-3:]
            macd_trending_down_s2 = False
            if len(macd_line_recent_s2) == 3 and \
                    (macd_line_recent_s2.iloc[2] < macd_line_recent_s2.iloc[1] and
                     macd_line_recent_s2.iloc[1] < macd_line_recent_s2.iloc[0]):
                macd_trending_down_s2 = True

            short_condition_s2 = (
                    is_momentum_down_s2 and hist_falling_from_peak_s2 and rsi_overbought_s2 and
                    volume_ok_s2 and price_falling_s2 and volatility_ok_s2 and macd_trending_down_s2
            )
            # print(f"SHORT S2 for {pair} on {timeframe}: {is_momentum_down_s2}, {hist_falling_from_peak_s2}, {rsi_overbought_s2}, {volume_ok_s2}, {price_falling_s2}, {volatility_ok_s2}, {macd_trending_down_s2}") # Debugging

    if short_condition_s1 and short_condition_s2:
        signal_type = 'SELL'

    if signal_type:
        now = datetime.now()
        signal_date = now.strftime("%Y-%m-%d")
        signal_time = now.strftime("%H:%M:%S")

        return {
            "id": str(uuid.uuid4()),
            "symbol": pair,
            "timeframe": timeframe,  # Use the dynamic timeframe
            "signal": signal_type,
            "date": signal_date,
            "time": signal_time,
            "entry_price": round(current_price, 6),
            "strategy_name": STRATEGY_NAME
        }
    return None


def print_signal_to_console(signal):
    """
    Prints the simplified signal to the console.
    """
    signal_color = COLOR_GREEN if signal['signal'] == 'BUY' else COLOR_RED
    tqdm.write(  # Use tqdm.write to print without breaking the progress bar
        f"SIGNAL: {signal_color}{signal['signal']}{COLOR_RESET} | PAIR: {signal['symbol']} | TF: {signal['timeframe']} | PRICE: {signal['entry_price']:.6f} | TIME: {signal['date']} {signal['time']}")


# --- File Handling ---
def load_existing_signals_data(filename: str = SIGNAL_LOG_FILE):
    """Loads existing signals from the JSON file (expected to be a list)."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                tqdm.write(
                    f"{COLOR_YELLOW}Warning: {filename} has unexpected structure. Re-initializing to empty list.{COLOR_RESET}")
                return []
            return data
    except json.JSONDecodeError:
        tqdm.write(
            f"{COLOR_YELLOW}Warning: Could not decode JSON from {filename}. Re-initializing to empty list.{COLOR_RESET}")
        return []
    except Exception as e:
        tqdm.write(
            f"{COLOR_RED}Error loading signals from {filename}: {e}. Re-initializing to empty list.{COLOR_RESET}")
        return []


def save_signals_to_file(new_signals: list, filename: str = SIGNAL_LOG_FILE):
    """Appends new signals to the existing JSON file (expected to be a list)."""
    existing_signals = load_existing_signals_data(filename)

    added_count = 0
    existing_signal_keys = set()
    # Ensure 'strategy_name' is included in the uniqueness check
    for s in existing_signals:
        if isinstance(s,
                      dict) and 'symbol' in s and 'timeframe' in s and 'signal' in s and 'date' in s and 'time' in s and 'strategy_name' in s:
            existing_signal_keys.add(
                (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name']))

    for signal in new_signals:
        # Use symbol, timeframe, signal, date, time, AND strategy_name for uniqueness check
        signal_key = (signal['symbol'], signal['timeframe'], signal['signal'], signal['date'], signal['time'],
                      signal.get('strategy_name'))

        if signal_key not in existing_signal_keys:
            existing_signals.append(signal)
            existing_signal_keys.add(signal_key)
            added_count += 1

    existing_signals.sort(key=lambda x: (x['date'], x['time']), reverse=True)

    try:
        with open(filename, 'w') as f:
            json.dump(existing_signals, f, indent=2)
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error saving signals to {filename}: {e}{COLOR_RESET}")

    return added_count


# --- Main Monitoring Function ---
def monitor_combined_signals():
    """
    Main loop to monitor for combined trading signals across multiple timeframes.
    """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\nStarting new combined strategy scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("Fetching list of all USDT pairs...")
        all_usdt_pairs = get_all_usdt_pairs()
        if not all_usdt_pairs:
            print(f"{COLOR_RED}Failed to fetch pairs. Retrying in 1 minute.{COLOR_RESET}")
            time.sleep(60)
            continue

        # Filter out excluded coins
        pairs_to_scan = [p for p in all_usdt_pairs if p not in COINS_TO_EXCLUDE_FINAL]
        pairs_to_scan.sort()  # Sort for consistent processing order

        signals_this_scan = []

        existing_logged_signals = load_existing_signals_data(SIGNAL_LOG_FILE)
        existing_logged_signal_details = set()
        for s in existing_logged_signals:
            if isinstance(s,
                          dict) and 'symbol' in s and 'timeframe' in s and 'signal' in s and 'date' in s and 'time' in s and 'strategy_name' in s:
                existing_logged_signal_details.add(
                    (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name']))

        # Outer loop for timeframes
        for timeframe in TIMEFRAMES:
            total_pairs = len(pairs_to_scan)
            print(f"\nScanning {total_pairs} USDT pairs on {timeframe} timeframe...")

            with tqdm(total=total_pairs, desc=f"Analyzing pairs on {timeframe}", unit="pair", ncols=100) as pbar:
                for i, pair in enumerate(pairs_to_scan):
                    pbar.set_description(f"Analyzing {pair} ({timeframe})")
                    # Pass the current timeframe to fetch_ohlcv
                    data = fetch_ohlcv(pair, timeframe, CANDLE_LIMIT)

                    # Adjusted check for minimum required bars for S2
                    min_required_s2_bars = max(MACD_SLOW_PERIOD_S2 + MACD_SIGNAL_PERIOD_S2,
                                               HIST_PEAK_TROUGH_LOOKBACK + REVERSAL_CONFIRMATION_BARS + 2)
                    if data is None or data.empty or len(data) < CANDLE_LIMIT or len(data) < min_required_s2_bars:
                        pbar.update(1)
                        continue

                    # Pass the current timeframe to check_combined_signal
                    new_signal = check_combined_signal(pair, data, timeframe)
                    if new_signal:
                        signal_key = (new_signal['symbol'], new_signal['timeframe'], new_signal['signal'],
                                      new_signal['date'], new_signal['time'], new_signal.get('strategy_name'))

                        if signal_key not in existing_logged_signal_details:
                            signals_this_scan.append(new_signal)
                            existing_logged_signal_details.add(signal_key)
                            print_signal_to_console(new_signal)
                    pbar.update(1)
                    # Use exchange.rateLimit which is directly from CCXT for better rate limit compliance
                    time.sleep(exchange.rateLimit / 1000 * 0.1)  # Small delay to respect rate limits, 10% of rate limit

        print("\nAnalysis complete for all timeframes.")

        if signals_this_scan:
            added_count = save_signals_to_file(signals_this_scan, SIGNAL_LOG_FILE)
            print(f"\n{COLOR_CYAN}Saved {added_count} new unique combined signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
        else:
            print(f"\nNo new combined signals found in this scan across all timeframes.")

        print(f"\nScan finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)


# --- Run the script ---
if __name__ == "__main__":
    print("Starting Combined Crypto Trading Signal Logger...")
    monitor_combined_signals()