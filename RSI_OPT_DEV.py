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
api_key = 'lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW'
api_secret = 'OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E'

STRATEGY_NAME = "RSI_OPT_DEV" # Defined globally
SIGNAL_LOG_FILE = 'RSI_OPT_DEV.json'
# Focus on higher timeframes for better signals
TIMEFRAMES = ['1h', '2h', '4h']
CANDLE_LIMIT = 100

COINS_TO_EXCLUDE = [
    'BZRX/USDT', 'AAVEUP/USDT', 'LEND/USDT',
    'USDC/USDT', 'BUSD/USDT', 'DAI/USDT', 'FDUSD/USDT', 'TUSD/USDT',
    'PYUSD/USDT', 'GUSD/USDT', 'USTC/USDT', 'MIM/USDT',
    'EURT/USDT', 'BKRW/USDT',
    'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT', 'WIF/USDT',
    'ETHUP/USDT', 'ETHDOWN/USDT', 'BTCUP/USDT', 'BTCDOWN/USDT', 'BNBUP/USDT', 'BNBDOWN/USDT',
    'ETC/USDT', 'XEC/USDT', 'LUNC/USDT', 'TRB/USDT', 'PERP/USDT', 'FTM/USDT',
    'CITY/USDT', 'PSG/USDT', 'LAZIO/USDT'
]

# RSI PURE STRATEGY PARAMETERS
RSI_PERIOD = 14
RSI_OVERSOLD = 25  # More extreme oversold (was 30)
RSI_OVERBOUGHT = 75  # More extreme overbought (was 70)

# RSI Divergence and Momentum Filters
RSI_DIVERGENCE_LOOKBACK = 5
MIN_RSI_MOMENTUM = 8  # RSI must move at least 8 points
RSI_TREND_BARS = 3  # RSI must trend for 3 bars

# Volume confirmation (minimal but important)
MIN_VOLUME_MULTIPLIER = 1.5

# Scheduling
SCAN_INTERVAL_MINUTES = 30  # Scan every hour for higher TF strategy
SCAN_INTERVAL_SECONDS = SCAN_INTERVAL_MINUTES * 60

# Initialize exchange
try:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    print("Successfully connected to Binance client via CCXT.")
except Exception as e:
    print(f"Error connecting to Binance client: {e}")
    exit()


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


def calculate_rsi(df, period=RSI_PERIOD):
    """Calculate RSI with proper handling."""
    if df.empty or len(df) < period + 1:
        return df

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Use Wilder's smoothing (proper RSI calculation)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Continue with exponential smoothing after initial period
    for i in range(period, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI

    return df


def check_rsi_divergence(df, current_idx, signal_type):
    """
    Check for RSI divergence - when price makes new highs/lows but RSI doesn't.
    This is a strong reversal signal.
    """
    if current_idx < RSI_DIVERGENCE_LOOKBACK:
        return False

    lookback_start = max(0, current_idx - RSI_DIVERGENCE_LOOKBACK)
    price_slice = df['Close'].iloc[lookback_start:current_idx + 1]
    rsi_slice = df['RSI'].iloc[lookback_start:current_idx + 1]

    if signal_type == 'BUY':
        # Bullish divergence: price making lower lows, RSI making higher lows
        price_min_idx = price_slice.idxmin()
        rsi_min_idx = rsi_slice.idxmin()

        # Check if we have a recent low in price but RSI is trending up
        recent_price_low = price_slice.iloc[-1] <= price_slice.min()
        recent_rsi_higher = rsi_slice.iloc[-1] > rsi_slice.min()

        return recent_price_low and recent_rsi_higher

    elif signal_type == 'SELL':
        # Bearish divergence: price making higher highs, RSI making lower highs
        price_max_idx = price_slice.idxmax()
        rsi_max_idx = rsi_slice.idxmax()

        # Check if we have a recent high in price but RSI is trending down
        recent_price_high = price_slice.iloc[-1] >= price_slice.max()
        recent_rsi_lower = rsi_slice.iloc[-1] < rsi_slice.max()

        return recent_price_high and recent_rsi_lower

    return False


def check_rsi_momentum(df, current_idx):
    """Check if RSI has sufficient momentum for the signal."""
    if current_idx < RSI_TREND_BARS:
        return False, 0

    current_rsi = df['RSI'].iloc[current_idx]
    past_rsi = df['RSI'].iloc[current_idx - RSI_TREND_BARS]

    rsi_change = abs(current_rsi - past_rsi)

    return rsi_change >= MIN_RSI_MOMENTUM, rsi_change


def check_rsi_trend_consistency(df, current_idx, signal_type):
    """Check if RSI trend is consistent for the signal direction."""
    if current_idx < RSI_TREND_BARS:
        return False

    rsi_values = df['RSI'].iloc[current_idx - RSI_TREND_BARS:current_idx + 1].values

    if signal_type == 'BUY':
        # For buy signals, we want RSI to be trending up from oversold
        return all(rsi_values[i] <= rsi_values[i + 1] for i in range(len(rsi_values) - 1))

    elif signal_type == 'SELL':
        # For sell signals, we want RSI to be trending down from overbought
        return all(rsi_values[i] >= rsi_values[i + 1] for i in range(len(rsi_values) - 1))

    return False


def generate_pure_rsi_signals(df, symbol, timeframe):
    """
    Pure RSI strategy with strict conditions for high-probability signals.
    """
    signals = []

    if df.empty or len(df) < RSI_PERIOD + RSI_TREND_BARS + RSI_DIVERGENCE_LOOKBACK:
        return signals

    df = calculate_rsi(df)

    # Get current bar data
    current_idx = len(df) - 1
    current_bar = df.iloc[current_idx]

    current_rsi = current_bar['RSI']
    current_close = current_bar['Close']
    current_volume = current_bar['Volume']

    # Volume filter - must be above average
    avg_volume = df['Volume'].iloc[-20:].mean()
    if current_volume < avg_volume * MIN_VOLUME_MULTIPLIER:
        return signals

    signal_timestamp = current_bar.name
    signal_date = signal_timestamp.strftime("%Y-%m-%d")
    signal_time = signal_timestamp.strftime("%H:%M:%S")

    # BUY SIGNAL CONDITIONS
    if current_rsi <= RSI_OVERSOLD:
        # Check all conditions for buy signal
        has_momentum, momentum_value = check_rsi_momentum(df, current_idx)
        has_trend_consistency = check_rsi_trend_consistency(df, current_idx, 'BUY')
        has_divergence = check_rsi_divergence(df, current_idx, 'BUY')

        # All conditions must be met for a signal
        if has_momentum and has_trend_consistency:
            # Divergence is a bonus but not required
            confidence = "HIGH" if has_divergence else "MEDIUM"

            buy_signal = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": "BUY",
                "date": signal_date,
                "time": signal_time,
                "entry_price": round(current_close, 6),
                "rsi_value": round(current_rsi, 2),
                "rsi_momentum": round(momentum_value, 2),
                "has_divergence": has_divergence,
                "confidence": confidence,
                "strategy_name": STRATEGY_NAME
            }
            signals.append(buy_signal)

    # SELL SIGNAL CONDITIONS
    elif current_rsi >= RSI_OVERBOUGHT:
        # Check all conditions for sell signal
        has_momentum, momentum_value = check_rsi_momentum(df, current_idx)
        has_trend_consistency = check_rsi_trend_consistency(df, current_idx, 'SELL')
        has_divergence = check_rsi_divergence(df, current_idx, 'SELL')

        # All conditions must be met for a signal
        if has_momentum and has_trend_consistency:
            # Divergence is a bonus but not required
            confidence = "HIGH" if has_divergence else "MEDIUM"

            sell_signal = {
                "id": str(uuid.uuid4()),
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": "SELL",
                "date": signal_date,
                "time": signal_time,
                "entry_price": round(current_close, 6),
                "rsi_value": round(current_rsi, 2),
                "rsi_momentum": round(momentum_value, 2),
                "has_divergence": has_divergence,
                "confidence": confidence,
                "strategy_name": STRATEGY_NAME
            }
            signals.append(sell_signal)

    return signals


def load_existing_signals_data(filename: str = SIGNAL_LOG_FILE) -> list[dict]:
    """Load existing signals from JSON file."""
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []

    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception) as e:
        tqdm.write(f"{COLOR_YELLOW}Warning: Could not load {filename}: {e}{COLOR_RESET}")
        return []


def save_signals_to_file(new_signals: list[dict], filename: str = SIGNAL_LOG_FILE):
    """Save new unique signals to JSON file."""
    existing_signals = load_existing_signals_data(filename)

    # Create set of existing signal keys for deduplication
    existing_keys = set()
    for s in existing_signals:
        if isinstance(s, dict):
            key = (s.get('symbol'), s.get('timeframe'), s.get('signal'),
                   s.get('date'), s.get('time'), s.get('strategy_name'))
            existing_keys.add(key)

    added_count = 0
    for signal in new_signals:
        key = (signal.get('symbol'), signal.get('timeframe'), signal.get('signal'),
               signal.get('date'), signal.get('time'), signal.get('strategy_name'))

        if key not in existing_keys:
            existing_signals.append(signal)
            existing_keys.add(key)
            added_count += 1

    try:
        with open(filename, 'w') as f:
            json.dump(existing_signals, f, indent=2)
    except Exception as e:
        tqdm.write(f"{COLOR_RED}Error saving to {filename}: {e}{COLOR_RESET}")

    return added_count


def print_signal_to_console(signal: dict):
    """Print signal to console with enhanced info."""
    signal_color = COLOR_GREEN if signal['signal'] == 'BUY' else COLOR_RED
    confidence_color = COLOR_CYAN if signal.get('confidence') == 'HIGH' else COLOR_YELLOW
    divergence_text = " [DIVERGENCE]" if signal.get('has_divergence') else ""

    tqdm.write(
        f"{signal_color}SIGNAL: {signal['signal']}{COLOR_RESET} | "
        f"PAIR: {signal['symbol']} | "
        f"PRICE: {signal['entry_price']:.6f} | "
        f"RSI: {signal['rsi_value']} | "
        f"MOMENTUM: {signal['rsi_momentum']} | "
        f"{confidence_color}CONF: {signal.get('confidence', 'N/A')}{COLOR_RESET}"
        f"{COLOR_MAGENTA}{divergence_text}{COLOR_RESET} | "
        f"TF: {signal['timeframe']} | "
        f"TIME: {signal['date']} {signal['time']}"
    )


def countdown_timer(seconds: int, total_seconds: int, message_prefix: str = "Next scan in"):
    """Display countdown timer with progress bar."""
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


def monitor_signals():
    """Main monitoring loop."""
    print(f"Starting Pure RSI Strategy Scanner ({STRATEGY_NAME})...")
    print(f"RSI Thresholds: Oversold <= {RSI_OVERSOLD}, Overbought >= {RSI_OVERBOUGHT}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Scan Interval: {SCAN_INTERVAL_MINUTES} minutes")

    try:
        exchange.load_markets()
        all_usdt_pairs = [s for s in exchange.symbols if s.endswith('/USDT') and s not in COINS_TO_EXCLUDE]
        print(f"Monitoring {len(all_usdt_pairs)} USDT pairs.")
    except Exception as e:
        print(f"{COLOR_RED}Error loading markets: {e}{COLOR_RESET}")
        return

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n--- Pure RSI Scan Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        signals_this_scan = []
        total_iterations = len(all_usdt_pairs) * len(TIMEFRAMES)

        with tqdm(total=total_iterations, desc="Scanning", unit="pair", ncols=120) as pbar:
            for pair in all_usdt_pairs:
                for timeframe in TIMEFRAMES:
                    df = fetch_binance_data(pair, timeframe)
                    if df is not None and not df.empty:
                        new_signals = generate_pure_rsi_signals(df, pair, timeframe)

                        for signal in new_signals:
                            signals_this_scan.append(signal)
                            print_signal_to_console(signal)

                    pbar.update(1)
                    time.sleep(0.1)  # Rate limiting

        print(f"\n{COLOR_CYAN}Analysis Complete{COLOR_RESET}")

        if signals_this_scan:
            added_count = save_signals_to_file(signals_this_scan, SIGNAL_LOG_FILE)
            print(f"{COLOR_GREEN}Saved {added_count} new signals to {SIGNAL_LOG_FILE}{COLOR_RESET}")
        else:
            print("No new signals found in this scan.")

        print(f"\nNext scan in {SCAN_INTERVAL_MINUTES} minutes...")
        countdown_timer(SCAN_INTERVAL_SECONDS, SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    monitor_signals()