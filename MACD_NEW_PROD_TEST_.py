# MACD Histogram Reversal Strategy - Optimized for Signal Generation

import pandas as pd
import ccxt
import numpy as np
import time
import datetime
import json
import os
from prettytable import PrettyTable
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
# IMPORTANT: Replace with your actual API Key and Secret.
# For production, consider using environment variables for security.
api_key = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
api_secret = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'

TIMEFRAMES = ['1h', '15m', '30m', '2h', '4h']
CANDLE_LIMIT = 250

COINS_TO_EXCLUDE = [
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

# Strategy Parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
RSI_PERIOD = 14

# Enhanced parameters
HIST_PEAK_TROUGH_LOOKBACK = 5
HIST_MOMENTUM_CHECK_BARS = 3
RSI_LONG_OVERSOLD_MAX = 40
RSI_SHORT_OVERBOUGHT_MIN = 60
MIN_VOLUME_RATIO = 0.8
REVERSAL_BARS_AGO_MAX = 2

# File Names
SIGNAL_LOG_FILE = 'MACD_NEW_OPT_PROD.json'
STRATEGY_NAME = "MACD Histogram_OPT_PROD"  # New: Strategy Name for JSON output

# --- Scan Interval (Standardized) ---
SCAN_INTERVAL_MINUTES = 15
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 30

# Initialize exchange
try:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,  # Automatyczna synchronizacja czasu
            'recvWindow': 10000  # Większe okno akceptowalnego odchylenia
        },
    })

    # Synchronizacja czasu z serwerem Binance
    exchange.load_time_difference()
    print(f"Successfully connected to Binance client via CCXT. Time diff: {exchange.time_difference} ms")

except Exception as e:
    print(f"Error connecting to Binance client: {e}")



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
        seconds -= 1
    print("\n", end="", flush=True)


# --- Data Fetching and Indicator Calculation ---

def fetch_binance_data(symbol, timeframe, limit=CANDLE_LIMIT):
    """Fetches OHLCV data for a given symbol and timeframe using CCXT."""
    try:
        # Since defaultType is 'spot' in client options, this should primarily fetch spot data.
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'open': 'Open', 'high': 'Hhigh', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                  inplace=True)
        return df
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error fetching {symbol} {timeframe}: {e}{COLOR_RESET}")
        return None


def calculate_indicators(df, macd_fast=MACD_FAST_PERIOD, macd_slow=MACD_SLOW_PERIOD,
                         macd_signal=MACD_SIGNAL_PERIOD, rsi_period=RSI_PERIOD):
    """Calculates MACD and RSI indicators for the given DataFrame."""
    if df.empty:
        return df

    # MACD Calculation
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['Signal_Line'] = df['MACD_Line'].ewm(span=macd_signal, adjust=False).mean()
    df['Histogram'] = df['MACD_Line'] - df['Signal_Line']

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


# --- Signal Generation ---

def generate_trading_signals(df, symbol, timeframe):
    """Generates trading signals based on the MACD Histogram Reversal Strategy."""
    signals = []
    min_required_bars = max(HIST_PEAK_TROUGH_LOOKBACK + 1, HIST_MOMENTUM_CHECK_BARS + 1)

    if df.empty or len(df) < min_required_bars:
        return signals

    required_cols = ['MACD_Line', 'Signal_Line', 'Histogram', 'RSI', 'Close', 'Volume']
    df_cleaned = df.dropna(subset=required_cols).copy()

    if df_cleaned.empty or len(df_cleaned) < min_required_bars:
        return signals

    i = len(df_cleaned) - 1
    window_start_idx = max(0, i - HIST_PEAK_TROUGH_LOOKBACK)
    hist_window = df_cleaned['Histogram'].iloc[window_start_idx:i]

    if hist_window.empty:
        return signals

    current_bar = df_cleaned.iloc[i]
    current_hist = current_bar['Histogram']
    current_rsi = current_bar['RSI']
    current_close = current_bar['Close']
    current_volume = current_bar['Volume']

    avg_volume = df_cleaned['Volume'].iloc[window_start_idx:i].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

    price_volatility = df_cleaned['Close'].iloc[window_start_idx:i].std() / df_cleaned['Close'].iloc[
                                                                            window_start_idx:i].mean()

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    # LONG Entry Conditions (enhanced)
    if not hist_window[hist_window < 0].empty:
        trough_idx = hist_window[hist_window < 0].idxmin()
        trough_pos_in_df = df_cleaned.index.get_loc(trough_idx)
        trough_bar = df_cleaned.iloc[trough_pos_in_df]
        trough_hist = trough_bar['Histogram']
        trough_bars_ago = i - trough_pos_in_df

        if trough_bars_ago <= REVERSAL_BARS_AGO_MAX:
            momentum_bars = df_cleaned['Histogram'].iloc[max(0, trough_pos_in_df):i + 1]  # Check momentum from trough
            is_momentum_up = (momentum_bars.diff().dropna() > 0).all()
            hist_rising = current_hist > trough_hist  # Current histogram value is higher than trough
            rsi_oversold = (current_rsi <= RSI_LONG_OVERSOLD_MAX)  # RSI is oversold
            volume_ok = (volume_ratio >= MIN_VOLUME_RATIO)
            price_rising = current_close > trough_bar['Close']  # Current price is higher than price at trough

            volatility_ok = price_volatility > 0.01  # Some volatility needed
            macd_trending_up = False
            if len(df_cleaned) >= 3:
                macd_line_recent = df_cleaned['MACD_Line'].iloc[-3:]
                if (macd_line_recent.iloc[2] > macd_line_recent.iloc[1] and
                        macd_line_recent.iloc[1] > macd_line_recent.iloc[0]):
                    macd_trending_up = True  # MACD line itself is trending up for last 3 bars

            if (is_momentum_up and hist_rising and rsi_oversold and
                    volume_ok and price_rising and volatility_ok and macd_trending_up):
                long_signal = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": "BUY",
                    "date": signal_date,
                    "time": signal_time,
                    "entry_price": round(current_close, 6),
                    "strategy_name": STRATEGY_NAME  # Added strategy name
                }
                signals.append(long_signal)

    # SHORT Entry Conditions (enhanced)
    if not hist_window[hist_window > 0].empty:
        peak_idx = hist_window[hist_window > 0].idxmax()
        peak_pos_in_df = df_cleaned.index.get_loc(peak_idx)
        peak_bar = df_cleaned.iloc[peak_pos_in_df]
        peak_hist = peak_bar['Histogram']
        peak_bars_ago = i - peak_pos_in_df

        if peak_bars_ago <= REVERSAL_BARS_AGO_MAX:
            momentum_bars = df_cleaned['Histogram'].iloc[max(0, peak_pos_in_df):i + 1]  # Check momentum from peak
            is_momentum_down = (momentum_bars.diff().dropna() < 0).all()

            hist_falling = current_hist < peak_hist  # Current histogram value is lower than peak
            rsi_overbought = (current_rsi >= RSI_SHORT_OVERBOUGHT_MIN)  # RSI is overbought
            volume_ok = (volume_ratio >= MIN_VOLUME_RATIO)
            price_falling = current_close < peak_bar['Close']  # Current price is lower than price at peak

            volatility_ok = price_volatility > 0.01  # Some volatility needed
            macd_trending_down = False
            if len(df_cleaned) >= 3:
                macd_line_recent = df_cleaned['MACD_Line'].iloc[-3:]
                if (macd_line_recent.iloc[2] < macd_line_recent.iloc[1] and
                        macd_line_recent.iloc[1] < macd_line_recent.iloc[0]):
                    macd_trending_down = True  # MACD line itself is trending down for last 3 bars

            if (is_momentum_down and hist_falling and rsi_overbought and
                    volume_ok and price_falling and volatility_ok and macd_trending_down):
                short_signal = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signal": "SELL",
                    "date": signal_date,
                    "time": signal_time,
                    "entry_price": round(current_close, 6),
                    "strategy_name": STRATEGY_NAME  # Added strategy name
                }
                signals.append(short_signal)

    return signals


# --- File Handling ---

def _get_default_signal_data():
    """Returns the default structure for the JSON file (an empty list)."""
    return []


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


# --- Main Function ---

def main():
    """Main function to fetch data, generate signals, and log them."""
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console at the start of each main loop
    print(f"\n{'=' * 80}")
    print(f"MACD Histogram Reversal Strategy - Signal Generation Only (at {current_time_str})")
    print(f"{'=' * 80}")

    try:
        markets = exchange.load_markets()
        # Explicitly filter for spot markets that are active and have USDT as quote asset.
        # This aligns symbol fetching with Codes 1 & 2.
        all_active_symbols = [
            symbol for symbol, market_data in markets.items()
            if market_data['quote'] == 'USDT' and market_data['active'] and market_data['spot']
        ]
        active_symbols = [s for s in all_active_symbols if
                          s not in COINS_TO_EXCLUDE]  # COINS_TO_EXCLUDE is already in BASE/QUOTE
        active_symbols.sort()

        if not active_symbols:
            tqdm.write("No active USDT spot symbols found.")
            return

    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error fetching symbols: {e}{COLOR_RESET}")
        return

    new_signals_table = PrettyTable()
    new_signals_table.field_names = ["🪙 Symbol", "⏰ TF", "🌊 Signal", "💰 Entry Price"]
    new_signals_table.align["🪙 Symbol"] = "l"

    total_combinations = len(active_symbols) * len(TIMEFRAMES)
    generated_signals_this_scan = []

    print(f"\n--- Scanning {len(active_symbols)} symbols across {len(TIMEFRAMES)} timeframes ---")

    existing_logged_signals = load_existing_signals_data(SIGNAL_LOG_FILE)
    existing_logged_signal_details = set()
    # Ensure 'strategy_name' is included in the uniqueness check
    for s in existing_logged_signals:
        if isinstance(s,
                      dict) and 'symbol' in s and 'timeframe' in s and 'signal' in s and 'date' in s and 'time' in s and 'strategy_name' in s:
            existing_logged_signal_details.add(
                (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name']))

    with tqdm(total=total_combinations, desc="Scanning for Signals", unit="pair", ncols=100) as pbar:
        for symbol in active_symbols:
            for timeframe in TIMEFRAMES:
                pbar.set_description(f"Scanning: {symbol} {timeframe}")

                try:
                    df = fetch_binance_data(symbol, timeframe)
                    if df is not None and not df.empty:
                        df = calculate_indicators(df)

                        if not df.dropna(subset=['MACD_Line', 'Signal_Line', 'Histogram', 'RSI']).empty and len(
                                df) >= HIST_PEAK_TROUGH_LOOKBACK + HIST_MOMENTUM_CHECK_BARS:
                            signals_for_pair = generate_trading_signals(df, symbol, timeframe)

                            for signal in signals_for_pair:
                                # Ensure 'strategy_name' is included in the uniqueness check for new signals
                                signal_key = (signal['symbol'], signal['timeframe'], signal['signal'], signal['date'],
                                              signal['time'], signal.get('strategy_name'))

                                if signal_key not in existing_logged_signal_details:
                                    generated_signals_this_scan.append(signal)
                                    existing_logged_signal_details.add(signal_key)  # Add to the set for current scan

                                    signal_color = COLOR_GREEN if signal['signal'] == 'BUY' else COLOR_RED
                                    # Use tqdm.write for printing signals found during the scan
                                    tqdm.write(
                                        f"SIGNAL: {signal_color}{signal['signal']}{COLOR_RESET} | PAIR: {signal['symbol']} | PRICE: {signal['entry_price']:.6f} | TIME: {signal['date']} {signal['time']}")
                                    new_signals_table.add_row([
                                        signal['symbol'],
                                        signal['timeframe'],
                                        f"{signal_color}{signal['signal']}{COLOR_RESET}",
                                        f"{signal['entry_price']:.6f}",
                                    ])

                except Exception as e:
                    tqdm.write(f"\n{COLOR_RED}Error processing {symbol}-{timeframe}: {e}{COLOR_RESET}")

                pbar.update(1)
                time.sleep(exchange.rateLimit / 1000)

    print(f"\n{COLOR_GREEN}Scan Complete!{COLOR_RESET}")

    if new_signals_table.rows:
        print("\n--- NEW SIGNALS DETECTED ---")
        print(new_signals_table)
        added_count = save_signals_to_file(generated_signals_this_scan)
        print(f"\n{COLOR_CYAN}Added {added_count} new signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
    else:
        print(f"\nNo new signals detected in this scan.")


if __name__ == "__main__":
    while True:
        main()
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)  # Use countdown_timer for next scan