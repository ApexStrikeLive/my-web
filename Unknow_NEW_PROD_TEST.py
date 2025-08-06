import os
import time
import math
import uuid
import json
from datetime import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from tqdm import tqdm  # Import tqdm for progress bar

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
API_SECRET = 'OZteDha0ZWmuxyOQHgp9EiK0CjVohFgZSIC6dLCx33HyJ0bKDenFFXQrLXjrTS8E'  # Placeholder

client = Client(API_KEY, API_SECRET)

# --- Stałe dla strategii i zarządzania plikami ---
SIGNAL_LOG_FILE = 'Binance_OPT_PROD.json'
STRATEGY_NAME = "Binance_OPT_PROD"  # New: Strategy Name for JSON output

# --- Coins to Exclude ---
COINS_TO_EXCLUDE = [
    'BZRXUSDT', 'AAVEUPUSDT', 'LENDUSDT',
    # Common Stablecoins (using the original format for exclusion list, will convert fetched symbols)
    'USDCUSDT', 'BUSDUSDT', 'DAIUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'PYUSDUSDT', 'USDPUSDT', 'GUSDUSDT', 'USTCUSDT',
    'MIMUSDT',
    # Highly Volatile / Meme Coins / Leveraged Tokens
    'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT',
    'ETHUPUSDT', 'ETHDOWNUSDT', 'BTCUPUSDT', 'BTCDOWNUSDT', 'BNBUPUSDT', 'BNBDOWNUSDT',
    # Some typically less liquid or highly speculative altcoins
    'TRBUSDT', 'LUNCUSDT', 'ETCUSDT',
]
# Convert exclusion list to BASE/QUOTE format for consistent comparison
COINS_TO_EXCLUDE_FORMATTED = [s.replace('USDT', '/USDT') for s in COINS_TO_EXCLUDE]

# --- Scan Interval (Standardized) ---
SCAN_INTERVAL_MINUTES = 30
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60


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

        # Use print with \r to overwrite the line
        print(f'\r{message_prefix}: [{bar}] {timer} remaining', end='', flush=True)
        time.sleep(1)
        seconds -= 1
    print("\n", end="", flush=True)  # Ensure a new line after countdown finishes


# --- Funkcje pobierania danych ---

def get_all_usdt_pairs():
    """
    Pobiera wszystkie aktywne pary handlowe z Binance, które kończą się na 'USDT'.
    Zwraca symbole w formacie 'BASE/QUOTE' (np. 'BTC/USDT').
    """
    try:
        exchange_info = client.get_exchange_info()
        usdt_pairs = []
        for s in exchange_info['symbols']:
            # Ensure it's a trading pair and its quote asset is USDT
            # And standardize symbol format to BASE/QUOTE
            if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT':
                usdt_pairs.append(f"{s['baseAsset']}/{s['quoteAsset']}")
        return usdt_pairs
    except BinanceAPIException as e:
        tqdm.write(f"{COLOR_RED}Błąd podczas pobierania par z Binance: {e}{COLOR_RESET}")  # Use tqdm.write
        return []
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Wystąpił nieoczekiwany błąd: {e}{COLOR_RESET}")  # Use tqdm.write
        return []


def fetch_ohlcv(symbol, interval, limit):
    """
    Pobiera dane OHLCV dla danego symbolu (w formacie BASE/QUOTE), interwału i limitu.
    Zwraca DataFrame z kolumnami 'open', 'high', 'low', 'close', 'volume', 'timestamp'.
    Konwertuje symbol z powrotem do formatu 'BASEQUOTE' dla API Binance.
    """
    binance_symbol = symbol.replace('/', '')  # Convert 'BTC/USDT' to 'BTCUSDT' for binance.client
    try:
        klines = client.get_klines(symbol=binance_symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        if len(df) < limit or df.isnull().values.any() or not (df['volume'] >= 0).all():
            return None

        return df
    except BinanceAPIException as e:
        return None
    except Exception as e:
        return None


# --- Funkcje obliczania wskaźników ---

def calculate_vwap(df):
    """
    Oblicza VWAP dla ostatnich 24 okresów.
    """
    if len(df) < 24:
        return None
    df_vwap = df.tail(24).copy()
    df_vwap['typical_price'] = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3
    df_vwap['price_volume'] = df_vwap['typical_price'] * df_vwap['volume']
    if df_vwap['volume'].sum() == 0:
        return None
    vwap = df_vwap['price_volume'].sum() / df_vwap['volume'].sum()
    return vwap


def calculate_rsi(series, period=14):
    """
    Oblicza wskaźnik RSI.
    """
    if len(series) < period * 2:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    if avg_loss.iloc[-1] == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Oblicza wskaźnik MACD, linię sygnału i histogram.
    """
    if len(series) < max(fast_period, slow_period, signal_period) + 1:
        return None

    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    if len(macd_histogram) < 2:
        return None

    return {
        'macd_line': macd_line.iloc[-1],
        'signal_line': signal_line.iloc[-1],
        'macd_histogram': macd_histogram.iloc[-1],
        'previous_macd_histogram': macd_histogram.iloc[-2]
    }


def calculate_volume_ratio(series, period=20):
    """
    Oblicza stosunek bieżącego wolumenu do 20-okresowej średniej kroczącej wolumenu.
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
    Oblicza wykładniczą średnią kroczącą (EMA).
    """
    if len(series) < period:
        return None
    return series.ewm(span=period, adjust=False).mean().iloc[-1]


# --- Logika wykrywania sygnałów ---

def check_signal(pair, data, vwap, rsi, macd_data, volume_ratio):
    """
    Sprawdza, czy spełnione są kryteria dla sygnałów LONG lub SHORT,
    i formatuje je do uproszczonej struktury JSON.
    """
    if data is None or vwap is None or rsi is None or macd_data is None or volume_ratio is None:
        return None

    current_price = data['close'].iloc[-1]

    ema_50_1h = calculate_ema(data['close'], 50)
    if ema_50_1h is None:
        return None

    macd_histogram = macd_data['macd_histogram']
    previous_macd_histogram = macd_data['previous_macd_histogram']

    # Filters (kept from original strategy logic)
    vwap_distance = abs(current_price - vwap) / vwap * 100
    if vwap_distance < 0.2:
        return None

    if not (rsi < 45 or rsi > 55):
        return None

    if abs(macd_histogram) < 0.01:
        return None

    if volume_ratio < 1.5:
        return None

    signal_type = None

    long_condition = (
            current_price > vwap and
            rsi > 50 and
            macd_histogram > 0 and
            macd_histogram > previous_macd_histogram and
            volume_ratio > 1.5 and
            current_price > ema_50_1h
    )

    short_condition = (
            current_price < vwap and
            rsi < 50 and
            macd_histogram < 0 and
            macd_histogram < previous_macd_histogram and
            volume_ratio > 1.5 and
            current_price < ema_50_1h
    )

    if long_condition:
        signal_type = 'BUY'
    elif short_condition:
        signal_type = 'SELL'

    if signal_type:
        now = datetime.now()
        signal_date = now.strftime("%Y-%m-%d")
        signal_time = now.strftime("%H:%M:%S")

        return {
            "id": str(uuid.uuid4()),
            "symbol": pair,  # Symbol is already in BASE/QUOTE format here
            "timeframe": "1h",
            "signal": signal_type,
            "date": signal_date,
            "time": signal_time,
            "entry_price": round(current_price, 6),
            "strategy_name": STRATEGY_NAME  # Added strategy name
        }
    return None


def print_signal_to_console(signal):
    """
    Prints the simplified signal to the console.
    """
    signal_color = COLOR_GREEN if signal['signal'] == 'BUY' else COLOR_RED
    tqdm.write(  # Use tqdm.write to print without breaking the progress bar
        f"SIGNAL: {signal_color}{signal['signal']}{COLOR_RESET} | PAIR: {signal['symbol']} | PRICE: {signal['entry_price']:.6f} | TIME: {signal['date']} {signal['time']}")


# --- Funkcje zarządzania plikami JSON (uproszczone dla sygnałów) ---

def load_existing_signals(filename: str = SIGNAL_LOG_FILE):
    """Ładuje istniejące sygnały z pliku JSON."""
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


def save_signals_to_file(new_signals: list, filename: str = SIGNAL_LOG_FILE):
    """Dodaje nowe sygnały do istniejącego pliku JSON."""
    existing_signals = load_existing_signals(filename)

    added_count = 0
    existing_signal_keys = set()
    # Check for symbol, timeframe, signal, date, time, AND strategy_name for uniqueness
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

    try:
        with open(filename, 'w') as f:
            json.dump(existing_signals, f, indent=2)
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error saving signals to {filename}: {e}{COLOR_RESET}")
    return added_count


# --- Główna funkcja monitorowania ---

def monitor_signals():
    """
    Główna pętla monitorująca sygnały transakcyjne, z zapisem do JSON.
    """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\nStarting new scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("Fetching list of all USDT pairs...")
        all_usdt_pairs = get_all_usdt_pairs()  # This now returns BASE/QUOTE format
        if not all_usdt_pairs:
            print(f"{COLOR_RED}Failed to fetch pairs. Retrying in 1 minute.{COLOR_RESET}")
            time.sleep(60)  # Short wait if API fails
            continue

        # Filter out excluded coins (already formatted in COINS_TO_EXCLUDE_FORMATTED)
        pairs_to_scan = [p for p in all_usdt_pairs if p not in COINS_TO_EXCLUDE_FORMATTED]

        signals_this_scan = []
        total_pairs = len(pairs_to_scan)
        print(f"Found {total_pairs} USDT pairs to scan (after exclusion).")

        existing_logged_signals = load_existing_signals(SIGNAL_LOG_FILE)
        existing_logged_signal_details = set()
        # Ensure 'strategy_name' is included in the uniqueness check
        for s in existing_logged_signals:
            if isinstance(s,
                          dict) and 'symbol' in s and 'timeframe' in s and 'signal' in s and 'date' in s and 'time' in s and 'strategy_name' in s:
                existing_logged_signal_details.add(
                    (s['symbol'], s['timeframe'], s['signal'], s['date'], s['time'], s['strategy_name']))

        # Using tqdm for progress bar during analysis
        with tqdm(total=total_pairs, desc="Analyzing pairs", unit="pair", ncols=100) as pbar:
            for i, pair in enumerate(pairs_to_scan):
                # pair is already in BASE/QUOTE format, fetch_ohlcv will convert it back for binance.client
                data = fetch_ohlcv(pair, '1h', 100)

                if data is None:
                    pbar.update(1)
                    continue

                vwap = calculate_vwap(data)
                rsi = calculate_rsi(data['close'])
                macd_data = calculate_macd(data['close'])
                volume_ratio = calculate_volume_ratio(data['volume'])

                new_signal = check_signal(pair, data, vwap, rsi, macd_data, volume_ratio)
                if new_signal:
                    # Ensure 'strategy_name' is included in the uniqueness check for new signals
                    signal_key = (
                    new_signal['symbol'], new_signal['timeframe'], new_signal['signal'], new_signal['date'],
                    new_signal['time'], new_signal.get('strategy_name'))

                    if signal_key not in existing_logged_signal_details:
                        signals_this_scan.append(new_signal)
                        existing_logged_signal_details.add(signal_key)
                        print_signal_to_console(new_signal)  # This now uses tqdm.write internally
                pbar.update(1)
                time.sleep(0.05)  # Small delay to manage API limits

        print("\nAnalysis complete.")

        if signals_this_scan:
            added_count = save_signals_to_file(signals_this_scan, SIGNAL_LOG_FILE)
            print(f"\n{COLOR_CYAN}Saved {added_count} new unique signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
        else:
            print(f"\nNo new signals found in this scan.")

        print(f"\nScan finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)  # Use countdown_timer for next scan


# --- Run the script ---
if __name__ == "__main__":
    print("Starting Binance Trading Signal Logger (Entry Signals Only)...")
    monitor_signals()