import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from datetime import datetime, timedelta
import uuid
import random  # For simulating price changes if Binance API fails or is not configured
import threading
import time
from collections import defaultdict  # For easier aggregation

# Attempt to import Binance client, provide a warning if not available
try:
    from binance.client import Client

    BINANCE_CLIENT_AVAILABLE = True
except ImportError:
    BINANCE_CLIENT_AVAILABLE = False
    print("Warning: 'python-binance' library not found. Live price fetching from Binance will be disabled.")
    print("Please install it using: pip install python-binance")

# --- Configuration ---
JSON_FILES_TO_MONITOR = [
    'MACD_NEW_OPT_TEST.json',
    'MACD_NEW_OPT_PROD.json',
    'Binance_OPT_TEST.json',
    'Binance_OPT_PROD.json',
    'SQZ_OPT_PROD.json',  # Added missing JSON files based on your logs
    'SQZ_OPT_TEST.json',
    'RSI_OPT_TEST.json',
    'RSI_OPT_PROD.json',
    'RSI_OPT_DEV.json',
    'Binance_OPT_DEV.json',
    'MACD_OPT_DEV.json',
    'merged_crypto_signals.json'
]
TRACED_SIGNALS_FILE = 'traced_signals.json'  # Renamed for clarity
SIGNAL_REFRESH_INTERVAL_MS = 1 * 30 * 1000  # 5 minutes in milliseconds
TRADE_MONITOR_INTERVAL_MS = 1 * 1000  # Changed to 1 second in milliseconds
# IMPORTANT: Replace with your actual Binance API Key and Secret
# For security, consider using environment variables in a real application
BINANCE_API_KEY = "lPh0cFLS7iB8pmBksBIeNEsbH0QqCBEsKMFQtnCPy4IPbu1cU1su4iorf6OisOGW"
BINANCE_API_SECRET = "OZteDha0ZWmuxwOVHgp9EiK0CjVohFgZSIC6dLCu33HyJ0bKDenFFXQrLXjrTS8E"

# --- Global Binance Client Instance ---
binance_client = None
if BINANCE_CLIENT_AVAILABLE and BINANCE_API_KEY != "YOUR_BINANCE_API_KEY":
    try:
        binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        # Test connection (optional but good practice)
        binance_client.get_server_time()
        print("Successfully connected to Binance API.")
    except Exception as e:
        print(f"Error connecting to Binance API: {e}. Live price fetching will be disabled.")
        binance_client = None
else:
    print("Binance API client not initialized. Using simulated prices.")


# --- Helper Functions for File Operations ---
def load_json_file(filepath):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}. Skipping.")
        return []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: JSON file {filepath} does not contain a list. Skipping.")
                return []
            print(f"Successfully loaded {len(data)} signals from {filepath}")  # Added confirmation print
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}. Skipping.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}. Skipping.")
        return []


def save_json_file(filepath, data):
    """Saves data to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")


# --- Data Processing and Trade Monitoring Logic ---

def load_all_signals_from_files():
    """
    Loads trade signals from the configured JSON files.
    Returns a sorted list of all signals.
    """
    all_signals = []
    for filepath in JSON_FILES_TO_MONITOR:
        signals = load_json_file(filepath)
        all_signals.extend(signals)

    # Add datetime objects for sorting and ensure required fields exist
    for signal in all_signals:
        try:
            # Ensure 'date' and 'time' are present before forming datetime string
            signal_date_str = signal.get('date', datetime.now().strftime("%Y-%m-%d"))
            signal_time_str = signal.get('time', datetime.now().strftime("%H:%M:%S"))
            signal['datetime_obj'] = datetime.strptime(f"{signal_date_str} {signal_time_str}", "%Y-%m-%d %H:%M:%S")
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not parse datetime for signal {signal.get('id', 'N/A')}: {e}. Assigning min date.")
            signal['datetime_obj'] = datetime.min  # Assign a minimum date to signals with bad date/time

    # Sort by datetime in descending order (most recent first)
    all_signals.sort(key=lambda s: s.get('datetime_obj', datetime.min), reverse=True)
    return all_signals


def load_traced_signals():
    """Loads all traced signals (monitored and non-monitored) from the dedicated file."""
    traced_signals = load_json_file(TRACED_SIGNALS_FILE)
    # Convert datetime strings back to datetime objects for internal use
    for signal_entry in traced_signals:
        if 'entry_time' in signal_entry:
            signal_entry['entry_time_obj'] = datetime.fromisoformat(signal_entry['entry_time'])
        if 'exit_time' in signal_entry and signal_entry['exit_time']:
            signal_entry['exit_time_obj'] = datetime.fromisoformat(signal_entry['exit_time'])
        # Ensure partial_closes is a list
        signal_entry['partial_closes'] = signal_entry.get('partial_closes', [])
        # Ensure initial_investment exists for older entries if not present
        # Set a default initial_investment for old entries if it's missing or 0
        if 'initial_investment' not in signal_entry or signal_entry['initial_investment'] == 0.0:
            # This will be overridden by DEFAULT_TRADE_AMOUNT in TradeDashboard init if needed
            signal_entry['initial_investment'] = 100.0  # A default for PnL % calculation

        # Ensure new flags for partial TP are set for existing entries
        signal_entry.setdefault('achieved_25_percent_tp', False)
        signal_entry.setdefault('achieved_50_percent_tp', False)
        # Ensure new list for activated TP levels is set for existing entries
        signal_entry.setdefault('activated_tp_levels', [])

    return traced_signals


def save_traced_signals(traced_signals):
    """Saves all traced signals to the dedicated file."""
    # Convert datetime objects to ISO format strings for JSON serialization
    serializable_signals = []
    for signal_entry in traced_signals:
        temp_signal_entry = signal_entry.copy()
        if 'entry_time_obj' in temp_signal_entry:
            temp_signal_entry['entry_time'] = temp_signal_entry['entry_time_obj'].isoformat()
            del temp_signal_entry['entry_time_obj']
        if 'exit_time_obj' in temp_signal_entry and temp_signal_entry['exit_time_obj']:
            temp_signal_entry['exit_time'] = temp_signal_entry['exit_time_obj'].isoformat()
            del temp_signal_entry['exit_time_obj']
        serializable_signals.append(temp_signal_entry)
    save_json_file(TRACED_SIGNALS_FILE, serializable_signals)


def get_live_current_price(symbol):
    """
    Fetches the current price for a given symbol from Binance.
    Returns None if API is not available or fails.
    """
    if binance_client:
        try:
            # Binance symbols are typically uppercase and without slashes (e.g., BTCUSDT)
            binance_symbol = symbol.replace('/', '').upper()
            ticker = binance_client.get_symbol_ticker(symbol=binance_symbol)
            return float(ticker['price'])
        except Exception as e:
            # print(f"Error fetching live price for {symbol} from Binance: {e}.") # Too verbose for every second
            return None
    return None


def get_simulated_price(entry_price):
    """
    Simulates fetching the current price for a given symbol.
    It introduces a small random fluctuation around the entry price.
    """
    fluctuation_percentage = random.uniform(-0.03, 0.03)  # -3% to +3%
    return entry_price * (1 + fluctuation_percentage)


def create_traced_signal_entry_template(signal, default_trade_amount):
    """Initializes a new entry template for the traced_signals.json."""
    # Use signal's original date/time for the template, will be overridden for new monitored trades
    entry_datetime = datetime.strptime(f"{signal['date']} {signal['time']}", "%Y-%m-%d %H:%M:%S")

    trade_obj = {
        "signal_id": signal.get('id'),
        "symbol": signal['symbol'],
        "timeframe": signal['timeframe'],
        "signal_type": signal['signal'],
        "entry_price": signal['entry_price'],
        "entry_time": entry_datetime.isoformat(),  # Store as ISO string
        "entry_time_obj": entry_datetime,  # Keep object for internal use
        "is_monitored": False,  # Default to False, set to True when added to GUI monitoring
        "monitoring_status": "NOT_MONITORED",  # Default status
        "current_price": signal['entry_price'],
        "profit_loss_percentage": 0.0,
        "realized_pnl": 0.0,  # Keep realized_pnl for internal calculation of PnL % for history tab
        "unrealized_pnl": 0.0,
        "stop_loss_price": 0.0,  # Will be calculated by monitoring logic
        "take_profit_target": 0.0,
        "closed_percentage": 0.0,
        "exit_price": None,
        "exit_time": None,
        "exit_time_obj": None,
        "partial_closes": [],
        "notes": "",
        "strategy_name": signal.get('strategy_name', 'N/A'),
        "initial_investment": default_trade_amount,  # Use the default trade amount
        "achieved_25_percent_tp": False,  # New flag to track if 25% TP was achieved
        "achieved_50_percent_tp": False,  # New flag to track if 50% TP was achieved
        "activated_tp_levels": []  # New list to track all activated TP levels
    }
    return trade_obj


# --- GUI Application Class ---
class TradeDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trade Signal Dashboard")
        self.geometry("1200x800")  # Set initial window size
        self.style = ttk.Style(self)
        self.configure_styles()

        self.all_signals = []  # Raw signals from input JSONs
        self.traced_signals = []  # All signals, with monitoring status and trade details
        self._monitoring_thread = None
        self._stop_event = threading.Event()  # Event to signal thread to stop

        self.DEFAULT_TRADE_AMOUNT = 100.0  # Nominal amount for a trade, used for PnL % calculation base
        self.COMMISSION_RATE = 0.001  # 0.1% commission per side (buy and sell)

        # Tracks financial state per strategy (simplified to only trade counts and percentages)
        self.strategy_financials = defaultdict(lambda: {
            'successful_trades_100_tp': 0,  # Trades that closed 100% at TP
            'unsuccessful_trades_100_sl': 0,  # Trades that closed 100% at initial SL (no partial closes)
            'partially_closed_25_plus': 0,  # Trades that closed >= 25% via TP
            'partially_closed_50_plus': 0,  # Trades that closed >= 50% via TP
            'mixed_outcome_trades': 0,  # Trades that had partial closes then hit SL
            'open_trades_count': 0  # New: Count of currently open trades for this strategy
        })

        # IMPORTANT FIX: Create widgets BEFORE loading and populating data
        self.create_widgets()
        self.load_initial_data()  # This will now also initialize budget based on loaded trades

        # Start periodic data refresh and trade monitoring
        self.after(1000, self.start_timers)  # Start timers after GUI is ready

        # Bind closing event to stop the monitoring thread
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def configure_styles(self):
        """Confgures the ttk styles for a modern look."""
        self.style.theme_use('clam')

        self.style.configure('TNotebook', background='#f0f0f0', borderwidth=0)
        self.style.configure('TNotebook.Tab',
                             background='#e0e0e0',
                             foreground='#333333',
                             padding=[10, 5],
                             font=('Inter', 10, 'bold'))
        self.style.map('TNotebook.Tab',
                       background=[('selected', '#4A90E2'), ('active', '#5c9ceb')],
                       foreground=[('selected', 'white')])

        self.style.configure('TFrame', background='#ffffff')

        self.style.configure('Treeview',
                             background='#ffffff',
                             foreground='#333333',
                             rowheight=25,
                             fieldbackground='#ffffff',
                             font=('Inter', 9))
        self.style.configure('Treeview.Heading',
                             background='#e9ecef',
                             foreground='#495057',
                             font=('Inter', 10, 'bold'))
        self.style.map('Treeview',
                       background=[('selected', '#b3d4fc')],
                       foreground=[('selected', '#1a1a1a')])

        self.style.configure('TLabel', background='#ffffff', foreground='#333333', font=('Inter', 10))
        self.style.configure('Header.TLabel', font=('Inter', 14, 'bold'), foreground='#333333')
        self.style.configure('SubHeader.TLabel', font=('Inter', 12, 'bold'), foreground='#4A90E2')
        self.style.configure('Value.TLabel', font=('Inter', 16, 'bold'), foreground='#007bff')

    def load_initial_data(self):
        """
        Loads all signals and traced signals at startup.
        Initializes strategy budgets and reconstructs their state from traced_signals.
        """
        self.all_signals = load_all_signals_from_files()
        self.traced_signals = load_traced_signals()

        # 1. Identify all unique strategies from both new and traced signals
        all_strategy_names = set()
        for signal in self.all_signals:
            all_strategy_names.add(signal.get('strategy_name', 'Unknown Strategy'))
        for signal_entry in self.traced_signals:
            all_strategy_names.add(signal_entry.get('strategy_name', 'Unknown Strategy'))

        # 2. Initialize or reset strategy financials for reconstruction
        self.strategy_financials = defaultdict(lambda: {
            'successful_trades_100_tp': 0,  # Trades that closed 100% at TP
            'unsuccessful_trades_100_sl': 0,  # Trades that closed 100% at initial SL (no partial closes)
            'partially_closed_25_plus': 0,  # Trades that closed >= 25% via TP
            'partially_closed_50_plus': 0,  # Trades that closed >= 50% via TP
            'mixed_outcome_trades': 0,  # Trades that had partial closes then hit SL
            'open_trades_count': 0  # New: Count of currently open trades for this strategy
        })

        # 3. Reconstruct state from existing traced signals (only trade counts)
        for signal_entry in self.traced_signals:
            strategy_name = signal_entry.get('strategy_name', 'Unknown Strategy')
            monitoring_status = signal_entry.get('monitoring_status', 'N/A')

            # Ensure initial_investment is set for older entries for consistent PnL % calculation
            signal_entry.setdefault('initial_investment', self.DEFAULT_TRADE_AMOUNT)

            # Ensure new flags for partial TP and activated levels are set for existing entries (for backward compatibility)
            signal_entry.setdefault('achieved_25_percent_tp', False)
            signal_entry.setdefault('achieved_50_percent_tp', False)
            signal_entry.setdefault('activated_tp_levels', [])

            # Update strategy financials based on historical trade outcomes
            if signal_entry.get('is_monitored', False):
                base_status = monitoring_status.split(' ')[0]  # e.g., "CLOSED_SL" from "CLOSED_SL (-3.00%)"

                if base_status == "CLOSED_TP":
                    self.strategy_financials[strategy_name]['successful_trades_100_tp'] += 1
                elif base_status == "CLOSED_SL":
                    if not signal_entry['partial_closes']:
                        self.strategy_financials[strategy_name]['unsuccessful_trades_100_sl'] += 1
                    else:
                        self.strategy_financials[strategy_name]['mixed_outcome_trades'] += 1
                elif base_status in ["OPEN", "PARTIALLY_CLOSED"]:  # Reconstruct open trades count
                    self.strategy_financials[strategy_name]['open_trades_count'] += 1

            # Reconstruct partial TP hit counters based on historical flags AND closed_percentage
            if signal_entry.get('is_monitored', False):  # Only count for monitored trades
                if signal_entry['closed_percentage'] >= 25.0 and not signal_entry['achieved_25_percent_tp']:
                    self.strategy_financials[strategy_name]['partially_closed_25_plus'] += 1
                    signal_entry['achieved_25_percent_tp'] = True  # Set the flag for future saves
                if signal_entry['closed_percentage'] >= 50.0 and not signal_entry['achieved_50_percent_tp']:
                    self.strategy_financials[strategy_name]['partially_closed_50_plus'] += 1
                    signal_entry['achieved_50_percent_tp'] = True  # Set the flag for future saves

        # 4. Process any new signals from source files that are not yet traced
        self.update_traced_signals_from_new_signals()
        self.update_gui_data()

    def update_traced_signals_from_new_signals(self):
        """
        Compares all_signals with traced_signals and adds/initializes new entries
        in traced_signals if they don't exist.
        For newly added signals, sets entry_time to current time and monitors them.
        """
        traced_signal_ids = {entry.get('signal_id') for entry in self.traced_signals if entry.get('signal_id')}

        # Define new_entry_datetime here so it's available for new entries
        new_entry_datetime = datetime.now()

        for signal in self.all_signals:
            signal_id = signal.get('id')
            strategy_name = signal.get('strategy_name', 'Unknown Strategy')

            if signal_id and signal_id not in traced_signal_ids:
                # This is a new signal not yet in traced_signals.json
                # Create a new traced signal entry template
                new_entry = create_traced_signal_entry_template(signal, self.DEFAULT_TRADE_AMOUNT)

                # Set entry_time to current time when it's first observed by the GUI
                new_entry['entry_time'] = new_entry_datetime.isoformat()
                new_entry['entry_time_obj'] = new_entry_datetime

                # Always monitor new signals and set status to OPEN
                new_entry['is_monitored'] = True
                new_entry['monitoring_status'] = "OPEN"
                # Increment open trades count for the strategy
                self.strategy_financials[strategy_name]['open_trades_count'] += 1

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] New signal {new_entry['signal_id'][:8]} ({new_entry['symbol']}) for strategy '{strategy_name}' added.")

                self.traced_signals.append(new_entry)
                traced_signal_ids.add(signal_id)  # Add to set to avoid duplicates in this run

        save_traced_signals(self.traced_signals)

    def _monitor_active_trades_logic(self, traced_signals_copy):
        """
        Applies stop-loss and dynamic take-profit logic to actively monitored trades.
        Fetches prices once for all unique active symbols.
        Returns the updated list of traced signals.
        """
        updated_traced_signals = []

        # 1. Identify actively monitored trades and collect their unique symbols
        active_monitored_trades = [
            signal_entry for signal_entry in traced_signals_copy
            if
            signal_entry.get('is_monitored') and signal_entry.get('monitoring_status') in ["OPEN", "PARTIALLY_CLOSED"]
        ]
        unique_active_symbols = {trade['symbol'] for trade in active_monitored_trades}

        # 2. Fetch current prices for all unique active symbols
        current_prices_cache = {}
        for symbol in unique_active_symbols:
            live_price = get_live_current_price(symbol)
            if live_price is not None:
                current_prices_cache[symbol] = live_price
            else:
                pass  # print(f"[{datetime.now().strftime('%H:%M:%S')}] Using simulated price for {symbol} as live data is unavailable.")

        # 3. Iterate through all traced signals and update actively monitored ones
        for signal_entry in traced_signals_copy:
            if signal_entry.get('is_monitored') and signal_entry.get('monitoring_status') in ["OPEN",
                                                                                              "PARTIALLY_CLOSED"]:
                # This is an actively monitored trade, apply logic
                current_price = current_prices_cache.get(signal_entry['symbol'])
                if current_price is None:
                    current_price = get_simulated_price(signal_entry['entry_price'])  # Fallback to simulation
                signal_entry['current_price'] = current_price

                # Calculate PnL factor based on signal type (pure price movement)
                if signal_entry['signal_type'] == 'BUY':
                    pnl_factor = (current_price - signal_entry['entry_price']) / signal_entry['entry_price']
                elif signal_entry['signal_type'] == 'SELL':
                    pnl_factor = (signal_entry['entry_price'] - current_price) / signal_entry['entry_price']
                else:
                    pnl_factor = 0.0

                # Calculate gross PnL in USD based on initial_investment
                gross_pnl_usd_current = pnl_factor * signal_entry['initial_investment']

                # Calculate total estimated commissions for the entire trade (entry + potential exit)
                # This is used to make the displayed P/L % reflect commission from the start
                total_estimated_commissions = signal_entry['initial_investment'] * self.COMMISSION_RATE * 2

                # Calculate net PnL in USD for display, including commissions
                net_pnl_usd_for_display = gross_pnl_usd_current - total_estimated_commissions

                # Update profit_loss_percentage to reflect net PnL including commissions
                signal_entry['profit_loss_percentage'] = (net_pnl_usd_for_display / signal_entry[
                    'initial_investment']) * 100

                # Unrealized PnL is still calculated for display in Active Trades tab (net of commissions for remaining portion)
                # For simplicity, let's keep it based on gross PnL for now, as the main PnL % covers the "lost from start"
                signal_entry['unrealized_pnl'] = gross_pnl_usd_current * (
                            (100 - signal_entry['closed_percentage']) / 100)

                # --- Stop Loss Logic (-3% from entry) ---
                if signal_entry['stop_loss_price'] == 0.0:
                    if signal_entry['signal_type'] == 'BUY':
                        initial_stop_loss_price = signal_entry['entry_price'] * (1 - 0.03)
                    elif signal_entry['signal_type'] == 'SELL':
                        initial_stop_loss_price = signal_entry['entry_price'] * (1 + 0.03)
                    else:
                        initial_stop_loss_price = 0.0  # Should not happen
                    signal_entry['stop_loss_price'] = initial_stop_loss_price
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - Initial SL set to {signal_entry['stop_loss_price']:.6f} (-3%)")

                # Check for Stop Loss hit
                sl_hit = False
                if signal_entry['signal_type'] == 'BUY' and current_price <= signal_entry['stop_loss_price']:
                    sl_hit = True
                elif signal_entry['signal_type'] == 'SELL' and current_price >= signal_entry['stop_loss_price']:
                    sl_hit = True

                if sl_hit:
                    # Calculate PnL for the remaining open portion when SL is hit
                    remaining_percentage = (100 - signal_entry['closed_percentage'])
                    gross_realized_pnl_on_sl = pnl_factor * signal_entry['initial_investment'] * (
                                remaining_percentage / 100)

                    # Deduct commissions for this closed portion (entry and exit commissions for this portion)
                    commission_cost_on_sl = (signal_entry['initial_investment'] * (
                                remaining_percentage / 100)) * self.COMMISSION_RATE * 2
                    realized_pnl_on_sl = gross_realized_pnl_on_sl - commission_cost_on_sl

                    signal_entry['realized_pnl'] += realized_pnl_on_sl  # Add to total realized PnL for this trade

                    # Update strategy financials (only counters, not cash)
                    strategy_name = signal_entry.get('strategy_name', 'Unknown Strategy')

                    # Decrement open trades count when a trade closes
                    if signal_entry.get('monitoring_status') in ["OPEN", "PARTIALLY_CLOSED"]:
                        self.strategy_financials[strategy_name]['open_trades_count'] -= 1

                    signal_entry['exit_price'] = current_price
                    signal_entry['exit_time_obj'] = datetime.now()
                    signal_entry['closed_percentage'] = 100.0  # Mark as fully closed
                    signal_entry['notes'] = "Closed by Stop Loss"

                    # Update successful/unsuccessful/mixed counters
                    if not signal_entry['partial_closes']:  # If no partial closes, it's a pure SL loss
                        self.strategy_financials[strategy_name]['unsuccessful_trades_100_sl'] += 1
                    else:  # If partial closes exist, it's a mixed outcome
                        self.strategy_financials[strategy_name]['mixed_outcome_trades'] += 1

                    # Update monitoring_status string with actual PnL % at closure (Point 2.1 fix)
                    # Calculate the final PnL % for the entire trade based on total realized PnL
                    final_pnl_percentage = (signal_entry['realized_pnl'] / signal_entry['initial_investment']) * 100
                    signal_entry['monitoring_status'] = f"CLOSED_SL ({final_pnl_percentage:.2f}%)"

                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - STOP LOSS HIT! Closed at {current_price:.6f}.")
                    print(
                        f"  PnL on this close: {final_pnl_percentage:.2f}%. Total Trade PnL (from initial investment): {signal_entry['realized_pnl']:.2f}.")  # Updated print
                    updated_traced_signals.append(signal_entry)
                    continue  # Move to next signal, this trade is closed

                # --- Dynamic Take Profit Logic ---
                tp_levels = [
                    (0.7, 25, 0.0),  # Level 1: TP 0.7% (net), close 25%, move SL to Entry Price (0% profit net)
                    (0.8, 0, 0.2),  # Level 2: TP 0.8% (net), move SL to 0.2% profit (net) (no close)
                    (1.2, 25, 0.6),  # Level 3: TP 1.2% (net), close 25%, move SL to 0.6% profit (net)
                    (1.5, 0, 0.8),  # Level 4: TP 1.5% (net), move SL to 0.8% profit (net) (no close)
                    (1.8, 0, 1.0),  # Level 5: TP 1.8% (net), move SL to 1.0% profit (net) (no close)
                    (2.0, 25, 1.4),  # Level 6: TP 2.0% (net), close 25%, move SL to 1.4% profit (net)
                    (2.4, 0, 1.6),  # Level 7: TP 2.4% (net), move SL to 1.6% profit (net) (no close)
                    (2.6, 0, 1.8),  # Level 8: TP 2.6% (net), move SL to 1.8% profit (net) (no close)
                    (2.8, 0, 2.0),  # Level 9: TP 2.8% (net), move SL to 2.0% profit (net) (no close)
                    (3.0, 0, 2.5),  # Level 10: TP 3.0% (net), move SL to 2.5% profit (net) (no close)
                    (3.3, 100, None)  # Level 11: TP 3.3% (net), take full profit (close remaining)
                ]

                remaining_position_percentage = 100.0 - signal_entry['closed_percentage']
                if remaining_position_percentage <= 0:  # If position is fully closed, skip TP logic
                    updated_traced_signals.append(signal_entry)
                    continue

                for tp_threshold, close_percent, sl_profit_percent in tp_levels:
                    # Only trigger if profit is reached AND the level hasn't been activated yet
                    triggered_level_key = f"tp_{tp_threshold}"

                    # Use the NET profit_loss_percentage for triggering TP levels
                    if signal_entry['profit_loss_percentage'] >= tp_threshold and triggered_level_key not in \
                            signal_entry['activated_tp_levels']:

                        actual_close_amount_percentage = 0
                        if close_percent > 0:
                            actual_close_amount_percentage = min(close_percent, remaining_position_percentage)

                        if actual_close_amount_percentage > 0:
                            # Calculate gross PnL for this partial close
                            gross_realized_pnl_on_partial = pnl_factor * signal_entry['initial_investment'] * (
                                        actual_close_amount_percentage / 100)

                            # Deduct commissions for this closed portion (entry and exit commissions)
                            commission_cost_on_partial = (signal_entry['initial_investment'] * (
                                        actual_close_amount_percentage / 100)) * self.COMMISSION_RATE * 2
                            realized_pnl_on_partial = gross_realized_pnl_on_partial - commission_cost_on_partial

                            # Record partial close
                            signal_entry['partial_closes'].append({
                                'level': triggered_level_key,
                                'percentage': actual_close_amount_percentage,
                                'price': current_price,
                                'time': datetime.now().isoformat(),
                                'realized_pnl_usd': realized_pnl_on_partial  # Store PnL for this slice
                            })
                            signal_entry['closed_percentage'] += actual_close_amount_percentage
                            signal_entry[
                                'realized_pnl'] += realized_pnl_on_partial  # Add to total realized PnL for this trade

                            # Update strategy financials (only counters, not cash)
                            strategy_name = signal_entry.get('strategy_name', 'Unknown Strategy')

                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - TP {tp_threshold}% hit! Closed {actual_close_amount_percentage:.2f}% at {current_price:.6f}.")
                            print(
                                f"  PnL on this partial close: {signal_entry['profit_loss_percentage']:.2f}%. Total Trade PnL (from initial investment): {signal_entry['realized_pnl']:.2f}.")  # Updated print

                            # Update flags and counters for 25% and 50% TP hits
                            if signal_entry['closed_percentage'] >= 25.0 and not signal_entry['achieved_25_percent_tp']:
                                self.strategy_financials[strategy_name]['partially_closed_25_plus'] += 1
                                signal_entry['achieved_25_percent_tp'] = True
                                print(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - Reached 25% TP closure threshold.")

                            if signal_entry['closed_percentage'] >= 50.0 and not signal_entry['achieved_50_percent_tp']:
                                self.strategy_financials[strategy_name]['partially_closed_50_plus'] += 1
                                signal_entry['achieved_50_percent_tp'] = True
                                print(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - Reached 50% TP closure threshold.")

                        # Add the triggered level to activated_tp_levels
                        signal_entry['activated_tp_levels'].append(triggered_level_key)

                        # Update Stop Loss for the remaining position
                        if sl_profit_percent is not None:
                            # Calculate the new SL price to achieve the desired NET profit percentage
                            # This accounts for the round-trip commission for the initial investment
                            target_net_profit_ratio = sl_profit_percent / 100
                            commission_factor = 2 * self.COMMISSION_RATE  # Round-trip commission as a decimal

                            calculated_new_sl_price = 0.0
                            if signal_entry['signal_type'] == 'BUY':
                                # To achieve target_net_profit_ratio, the price needs to move by target_net_profit_ratio + commission_factor
                                price_movement_ratio = target_net_profit_ratio + commission_factor
                                calculated_new_sl_price = signal_entry['entry_price'] * (1 + price_movement_ratio)
                                # Ensure SL only moves up or stays put for BUY trades
                                signal_entry['stop_loss_price'] = max(signal_entry['stop_loss_price'],
                                                                      calculated_new_sl_price)
                            elif signal_entry['signal_type'] == 'SELL':
                                # To achieve target_net_profit_ratio, the price needs to move by target_net_profit_ratio + commission_factor
                                price_movement_ratio = target_net_profit_ratio + commission_factor
                                calculated_new_sl_price = signal_entry['entry_price'] * (1 - price_movement_ratio)
                                # Ensure SL only moves down or stays put for SELL trades
                                signal_entry['stop_loss_price'] = min(signal_entry['stop_loss_price'],
                                                                      calculated_new_sl_price)

                            signal_entry['notes'] = f"SL moved to {sl_profit_percent}% profit (net)"
                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - SL moved to {signal_entry['stop_loss_price']:.6f} ({sl_profit_percent}% profit net)")

                        # After processing a TP level, re-evaluate remaining position
                        remaining_position_percentage = 100.0 - signal_entry['closed_percentage']
                        if remaining_position_percentage <= 0:
                            signal_entry['monitoring_status'] = "CLOSED_TP"
                            signal_entry['exit_price'] = current_price
                            signal_entry['exit_time_obj'] = datetime.now()
                            signal_entry['notes'] = "Fully closed by Take Profit"

                            # Increment successful trades counter
                            self.strategy_financials[signal_entry.get('strategy_name', 'Unknown Strategy')][
                                'successful_trades_100_tp'] += 1
                            # Decrement open trades count when a trade closes
                            if signal_entry.get('monitoring_status') in ["OPEN",
                                                                         "PARTIALLY_CLOSED"]:  # Check before decrementing
                                self.strategy_financials[signal_entry.get('strategy_name', 'Unknown Strategy')][
                                    'open_trades_count'] -= 1

                            # Update monitoring_status string with actual PnL % at closure (Point 2.1 fix)
                            # Calculate the final PnL % for the entire trade based on total realized PnL
                            final_pnl_percentage = (signal_entry['realized_pnl'] / signal_entry[
                                'initial_investment']) * 100
                            signal_entry['monitoring_status'] = f"CLOSED_TP ({final_pnl_percentage:.2f}%)"

                            print(
                                f"[{datetime.now().strftime('%H:%M:%S')}] Trade {signal_entry['signal_id'][:8]} ({signal_entry['symbol']}) - FULLY CLOSED by TP at {current_price:.6f}.")
                            print(
                                f"  Total Trade PnL (from initial investment): {signal_entry['realized_pnl']:.2f}.")  # Updated print
                            break  # Exit TP loop if fully closed

                        # Update status if partially closed (and not fully closed yet)
                        if signal_entry['closed_percentage'] > 0 and signal_entry['closed_percentage'] < 100.0:
                            signal_entry['monitoring_status'] = "PARTIALLY_CLOSED"
            updated_traced_signals.append(signal_entry)

        return updated_traced_signals

    def start_timers(self):
        """Starts the periodic refresh and monitoring timers in a separate thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()  # Clear stop event for new thread
            self._monitoring_thread = threading.Thread(target=self._monitoring_thread_function, daemon=True)
            self._monitoring_thread.start()
            print("Monitoring thread started.")

    def _monitoring_thread_function(self):
        """Function to run in the background thread for periodic tasks."""
        last_signal_refresh_time = time.time()
        last_trade_monitor_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()

            # Check for signal refresh and update traced signals
            if (current_time - last_signal_refresh_time) * 1000 >= SIGNAL_REFRESH_INTERVAL_MS:
                self.refresh_all_signals_and_update_traced_in_thread()
                last_signal_refresh_time = current_time

            # Check for trade monitoring (now every second)
            if (current_time - last_trade_monitor_time) * 1000 >= TRADE_MONITOR_INTERVAL_MS:
                # Pass a copy of traced_signals to the monitoring logic to avoid race conditions
                self._monitor_trades_in_thread(list(self.traced_signals))
                last_trade_monitor_time = current_time

            time.sleep(0.1)  # Sleep for a short duration to not busy-wait, but still allow frequent checks

    def refresh_all_signals_and_update_traced_in_thread(self):
        """Loads new signals from files and updates the traced_signals (called from thread)."""
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing all signals and updating traced_signals (from thread)...")
        new_all_signals = load_all_signals_from_files()

        # Update main thread data and GUI via self.after
        self.after(0, self._update_all_signals_and_traced_from_thread, new_all_signals)

    def _update_all_signals_and_traced_from_thread(self, new_all_signals):
        """Updates main thread's all_signals and traced_signals (called via self.after)."""
        self.all_signals = new_all_signals
        self.update_traced_signals_from_new_signals()  # Re-process new signals into traced_signals
        self.update_gui_data()

    def _monitor_trades_in_thread(self, traced_signals_copy):
        """Monitors active trades and applies SL/TP logic (called from thread)."""
        print(f"[{datetime.now().strftime('%H:%M:%M')}] Monitoring active trades (from thread)...")
        updated_traced_signals = self._monitor_active_trades_logic(traced_signals_copy)
        # Update main thread data and GUI via self.after
        self.after(0, self._update_monitored_trades_from_thread, updated_traced_signals)

    def _update_monitored_trades_from_thread(self, updated_traced_signals):
        """Updates main thread's traced_signals and GUI (called via self.after)."""
        self.traced_signals = updated_traced_signals
        save_traced_signals(self.traced_signals)
        self.update_gui_data()

    def update_gui_data(self):
        """Clears and repopulates the Treeviews on all tabs."""
        # Clear existing data
        for tree in [getattr(self, 'active_tree', None), getattr(self, 'all_tree', None),
                     getattr(self, 'analysis_tree', None)]:
            if tree:
                for item in tree.get_children():
                    tree.delete(item)

        # Repopulate Active Trades Tab
        self.populate_active_trades_tab()
        # Repopulate All Trades Tab
        self.populate_all_trades_tab()
        # Refresh Strategy Analysis Tab
        self.populate_strategy_analysis_tab()

    def create_widgets(self):
        """Creates the main GUI widgets (notebook, tabs)."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.active_trades_frame = ttk.Frame(self.notebook, style='TFrame')
        self.strategy_analysis_frame = ttk.Frame(self.notebook, style='TFrame')
        self.all_trades_frame = ttk.Frame(self.notebook, style='TFrame')  # This will become 'Trade History'

        self.notebook.add(self.active_trades_frame, text='Active Trades')
        self.notebook.add(self.strategy_analysis_frame, text='Strategy Analysis')
        self.notebook.add(self.all_trades_frame, text='Trade History')  # Renamed for clarity

        # Initial setup of tab contents (will be populated by update_gui_data)
        self.setup_active_trades_tab_structure()
        self.setup_strategy_analysis_tab_structure()
        self.setup_all_trades_tab_structure()  # This now sets up Trade History

    def setup_active_trades_tab_structure(self):
        """Sets up the static structure for the 'Active Trades' tab."""
        tk.Label(self.active_trades_frame, text="Current Monitored Trades",
                 font=('Inter', 16, 'bold'), bg='white', fg='#333333', pady=10).pack(pady=(0, 10))

        # Dynamic TP/SL Setup Info
        tp_sl_info_text = (
            "Dynamic Take Profit & Stop Loss Setup:\n"
            "  - Initial Stop Loss: -3% from Entry Price.\n"
            "  - Level 1 (0.7% Profit): Close 25% of position, move SL to Entry Price (0% profit).\n"
            "  - Level 2 (0.8% Profit): Move SL to +0.2% profit (no close).\n"
            "  - Level 3 (1.2% Profit): Close 25% of position, move SL to +0.6% profit.\n"
            "  - Level 4 (1.5% Profit): Move SL to +0.8% profit (no close).\n"
            "  - Level 5 (1.8% Profit): Move SL to +1.0% profit (no close).\n"
            "  - Level 6 (2.0% Profit): Close 25% of position, move SL to +1.4% profit.\n"
            "  - Level 7 (2.4% Profit): Move SL to +1.6% profit (no close).\n"
            "  - Level 8 (2.6% Profit): Move SL to +1.8% profit (no close).\n"
            "  - Level 9 (2.8% Profit): Move SL to +2.0% profit (no close).\n"
            "  - Level 10 (3.0% Profit): Move SL to +2.5% profit (no close).\n"
            "  - Level 11 (3.3% Profit): Close remaining position (full profit)."
        )
        self.tp_sl_info_label = tk.Label(self.active_trades_frame, text=tp_sl_info_text,
                                         font=('Inter', 9), bg='white', fg='#555555',
                                         justify=tk.LEFT, anchor='w')
        self.tp_sl_info_label.pack(pady=(0, 10), padx=10, fill='x')

        # Updated columns for Active Trades (removed "Signal ID")
        columns = ("Strategy Name", "Symbol", "Type", "Entry Price", "Current Price",
                   "P/L %", "Status", "Closed %", "Stop Loss", "Entry Time", "Notes")
        self.active_tree = ttk.Treeview(self.active_trades_frame, columns=columns, show='headings')
        self.active_tree.pack(fill='both', expand=True, padx=10, pady=10)

        for col in columns:
            self.active_tree.heading(col, text=col, anchor='w')
            self.active_tree.column(col, width=100, anchor='w')

        # Adjust specific column widths
        self.active_tree.column("Strategy Name", width=150)  # Increased width
        self.active_tree.column("Symbol", width=70)
        self.active_tree.column("Type", width=60)
        self.active_tree.column("Entry Price", width=90)
        self.active_tree.column("Current Price", width=90)
        self.active_tree.column("P/L %", width=70)
        self.active_tree.column("Status", width=90)
        self.active_tree.column("Closed %", width=70)
        self.active_tree.column("Stop Loss", width=90)
        self.active_tree.column("Entry Time", width=150)
        self.active_tree.column("Notes", width=150)

        # New styles for Active Trades tab - configured directly on the Treeview
        self.active_tree.tag_configure('pnl_minus_pink', foreground='#FF007F')  # Bright Pink
        self.active_tree.tag_configure('pnl_plus_yellow', foreground='#CCCC00')  # Dark Yellow
        self.active_tree.tag_configure('partial_closed_25_blue', foreground='#0000FF')  # Blue
        self.active_tree.tag_configure('partial_closed_50_green', foreground='#008000')  # Green

        # Removed scrollbar
        # scrollbar = ttk.Scrollbar(self.active_trades_frame, orient="vertical", command=self.active_tree.yview)
        # self.active_tree.configure(yscrollcommand=scrollbar.set)
        # scrollbar.pack(side="right", fill="y")

    def populate_active_trades_tab(self):
        """Populates the 'Active Trades' tab with current monitored trade data."""
        # Clear existing data
        for item in self.active_tree.get_children():
            self.active_tree.delete(item)

        if not self.traced_signals:
            self.active_tree.insert("", "end",
                                    values=("No monitored trades found.",) * len(self.active_tree['columns']))
            return

        # Filter trades to only show 'OPEN' or 'PARTIALLY_CLOSED' AND 'is_monitored' is True
        active_filtered_trades = [
            trade for trade in self.traced_signals
            if trade.get('is_monitored') and trade.get('monitoring_status') in ["OPEN", "PARTIALLY_CLOSED"]
        ]

        if not active_filtered_trades:
            self.active_tree.insert("", "end", values=("No active trades found.",) * len(self.active_tree['columns']))
            return

        for trade in active_filtered_trades:  # Iterate over filtered trades
            status = trade.get('monitoring_status', 'UNKNOWN')  # Use monitoring_status
            pnl_percentage = trade.get('profit_loss_percentage', 0)
            closed_percentage = trade.get('closed_percentage', 0)

            # Determine the tag based on new coloring rules
            tag_to_apply = ''
            if pnl_percentage < 0:
                tag_to_apply = 'pnl_minus_pink'
            elif closed_percentage >= 50:
                tag_to_apply = 'partial_closed_50_green'
            elif closed_percentage >= 25:
                tag_to_apply = 'partial_closed_25_blue'
            elif pnl_percentage >= 0:  # This condition catches positive PnL
                tag_to_apply = 'pnl_plus_yellow'

            self.active_tree.insert("", "end", values=(
                trade.get('strategy_name', 'N/A'),  # Display Strategy Name
                trade.get('symbol'),
                trade.get('signal_type'),
                f"{trade.get('entry_price', 0):.6f}",
                f"{trade.get('current_price', 0):.6f}",
                f"{pnl_percentage:.2f}%",  # Use the calculated PnL
                status,
                f"{closed_percentage:.2f}%",  # Use the calculated closed percentage
                f"{trade.get('stop_loss_price', 0):.6f}",
                trade.get('entry_time', 'N/A').split('T')[0] + ' ' + trade.get('entry_time', 'N/A').split('T')[1][
                                                                     :8] if trade.get('entry_time') else 'N/A',
                trade.get('notes', '')
            ), tags=(tag_to_apply,))

    def setup_strategy_analysis_tab_structure(self):
        """Sets up the static structure for the 'Strategy Analysis' tab."""
        tk.Label(self.strategy_analysis_frame, text="Strategy Performance Analysis",
                 font=('Inter', 16, 'bold'), bg='white', fg='#333333', pady=10).pack(pady=(0, 10))

        # Updated columns for Strategy Analysis (added new percentages and shortened names)
        columns = ("Strategy Name", "Total Signals",
                   "Open Trades", "Open Trades %",  # New columns for open trades
                   "Full TP Trades", "Full TP %",  # Shortened name
                   "TP 25% Trades", "TP 25% %",  # Shortened name
                   "TP 50% Trades", "TP 50% %",  # Shortened name
                   "SL 100% Trades", "SL 100% %",  # Shortened name for 100% SL
                   "Total SL Trades", "Total SL %",  # Shortened combined SL metrics
                   "Success Rate (Full TP)", "Partial Success Rate (%)")  # Shortened success rate name
        self.analysis_tree = ttk.Treeview(self.strategy_analysis_frame, columns=columns, show='headings')
        self.analysis_tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure column headings and widths
        for col in columns:
            self.analysis_tree.heading(col, text=col, anchor='w')
            # Set default width, then adjust specific ones
            self.analysis_tree.column(col, width=100, anchor='w')

        self.analysis_tree.column("Strategy Name", width=150)
        self.analysis_tree.column("Total Signals", width=100)
        self.analysis_tree.column("Open Trades", width=100)  # New
        self.analysis_tree.column("Open Trades %", width=120)  # New
        self.analysis_tree.column("Full TP Trades", width=130)  # Shortened
        self.analysis_tree.column("Full TP %", width=100)  # Shortened
        self.analysis_tree.column("TP 25% Trades", width=130)  # Shortened
        self.analysis_tree.column("TP 25% %", width=100)  # Shortened
        self.analysis_tree.column("TP 50% Trades", width=130)  # Shortened
        self.analysis_tree.column("TP 50% %", width=100)  # Shortened
        self.analysis_tree.column("SL 100% Trades", width=130)  # Shortened
        self.analysis_tree.column("SL 100% %", width=100)  # Shortened
        self.analysis_tree.column("Total SL Trades", width=130)  # Shortened
        self.analysis_tree.column("Total SL %", width=100)  # Shortened
        self.analysis_tree.column("Success Rate (Full TP)", width=160)  # Shortened
        self.analysis_tree.column("Partial Success Rate (%)", width=160)

        # Removed scrollbar
        # scrollbar = ttk.Scrollbar(self.strategy_analysis_frame, orient="vertical", command=self.analysis_tree.yview)
        # self.analysis_tree.configure(yscrollcommand=scrollbar.set)
        # scrollbar.pack(side="right", fill="y")

    def get_total_budget(self):
        """Calculates the sum of all individual strategy budgets. (No longer used for display)"""
        # This function is retained for completeness but its display is removed as per user request
        # If user wants to re-add total budget, this function would be useful.
        return 0.0  # Or remove entirely if no future use is foreseen

    def populate_strategy_analysis_tab(self):
        """Populates the 'Strategy Analysis' tab with aggregated data."""
        # Removed current total budget display update
        # self.current_budget_label.config(text=f"Current Total Budget: ${self.get_total_budget():.2f}")

        # Clear existing data
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)

        if not self.traced_signals:
            self.analysis_tree.insert("", "end", values=("No signals available for analysis.",) * len(
                self.analysis_tree['columns']))
            return

        for strategy_name, data in self.strategy_financials.items():
            total_signals = sum(1 for s in self.traced_signals if s.get('strategy_name') == strategy_name)

            successful_100_tp = data['successful_trades_100_tp']
            unsuccessful_100_sl = data['unsuccessful_trades_100_sl']
            mixed_outcome = data['mixed_outcome_trades']
            partially_closed_25_plus = data['partially_closed_25_plus']
            partially_closed_50_plus = data['partially_closed_50_plus']
            open_trades_count = data['open_trades_count']  # Get open trades count

            # New: Total trades closed by SL (sum of 100% SL and mixed outcome)
            total_sl_trades = unsuccessful_100_sl + mixed_outcome

            # Total closed trades for success rate calculations (used for overall rates)
            total_closed_for_rates = successful_100_tp + total_sl_trades

            # Calculate 100% TP Success Rate
            success_rate_100_tp = 0.0
            if total_closed_for_rates > 0:
                success_rate_100_tp = (successful_100_tp / total_closed_for_rates) * 100

            # Calculate Partial Success Rate (trades that had any TP or closed 100% TP)
            total_partially_successful = successful_100_tp + mixed_outcome
            partial_success_rate = 0.0
            if total_closed_for_rates > 0:
                partial_success_rate = (total_partially_successful / total_closed_for_rates) * 100

            # New Percentage Calculations for Strategy Analysis Tab
            open_trades_percent = 0.0
            if total_signals > 0:
                open_trades_percent = (open_trades_count / total_signals) * 100

            successful_100_tp_percent = 0.0
            if total_signals > 0:
                successful_100_tp_percent = (successful_100_tp / total_signals) * 100

            tp_25_hit_percent = 0.0
            if total_signals > 0:
                tp_25_hit_percent = (partially_closed_25_plus / total_signals) * 100

            tp_50_hit_percent = 0.0
            if total_signals > 0:
                tp_50_hit_percent = (partially_closed_50_plus / total_signals) * 100

            unsuccessful_100_sl_percent = 0.0
            if total_signals > 0:
                unsuccessful_100_sl_percent = (unsuccessful_100_sl / total_signals) * 100

            total_sl_trades_percent = 0.0
            if total_signals > 0:
                total_sl_trades_percent = (total_sl_trades / total_signals) * 100

            self.analysis_tree.insert("", "end", values=(
                strategy_name,
                total_signals,
                open_trades_count,  # New
                f"{open_trades_percent:.2f}%",  # New
                successful_100_tp,
                f"{successful_100_tp_percent:.2f}%",
                partially_closed_25_plus,
                f"{tp_25_hit_percent:.2f}%",
                partially_closed_50_plus,
                f"{tp_50_hit_percent:.2f}%",
                unsuccessful_100_sl,
                f"{unsuccessful_100_sl_percent:.2f}%",
                total_sl_trades,
                f"{total_sl_trades_percent:.2f}%",
                f"{success_rate_100_tp:.2f}%",
                f"{partial_success_rate:.2f}%"
            ))

    def setup_all_trades_tab_structure(self):
        """Sets up the static structure for the 'Trade History' tab."""
        tk.Label(self.all_trades_frame, text="Trade History (All Signals & Monitored Trades)",
                 font=('Inter', 16, 'bold'), bg='white', fg='#333333', pady=10).pack(pady=(0, 10))

        # Updated columns for Trade History (removed financial metrics)
        columns = ("Strategy", "Symbol", "Type", "Entry Price", "Signal Time",
                   "Monitored?", "Monitoring Status", "P/L %", "Exit Price",
                   "Exit Time")  # Removed Realized PnL, Initial Inv.
        self.all_tree = ttk.Treeview(self.all_trades_frame, columns=columns, show='headings')
        self.all_tree.pack(fill='both', expand=True, padx=10, pady=10)

        for col in columns:
            self.all_tree.heading(col, text=col, anchor='w')
            self.all_tree.column(col, width=100, anchor='w')

        # Adjust specific column widths
        self.all_tree.column("Strategy", width=150)  # Increased width
        self.all_tree.column("Symbol", width=70)
        self.all_tree.column("Type", width=60)
        self.all_tree.column("Entry Price", width=90)
        self.all_tree.column("Signal Time", width=150)
        self.all_tree.column("Monitored?", width=80)
        self.all_tree.column("Monitoring Status", width=120)  # Renamed from Monitored Status
        self.all_tree.column("P/L %", width=90)  # Renamed from Monitored P/L %
        self.all_tree.column("Exit Price", width=90)
        self.all_tree.column("Exit Time", width=150)

        self.all_tree.tag_configure('buy', foreground='green')
        self.all_tree.tag_configure('sell', foreground='red')
        self.all_tree.tag_configure('monitored_success', background='#e6ffe6',
                                    foreground='green')  # Light green background for success
        self.all_tree.tag_configure('monitored_failure', background='#ffe6e6',
                                    foreground='red')  # Light red background for failure
        self.all_tree.tag_configure('monitored_open', foreground='blue')  # For trades still open/partial
        self.all_tree.tag_configure('not_monitored', foreground='gray')  # For signals not actively monitored

        # Removed scrollbar
        # scrollbar = ttk.Scrollbar(self.all_trades_frame, orient="vertical", command=self.all_tree.yview)
        # self.all_tree.configure(yscrollcommand=scrollbar.set)
        # scrollbar.pack(side="right", fill="y")

    def populate_all_trades_tab(self):
        """Populates the 'Trade History' tab with all signals and their monitoring status."""
        # Clear existing data
        for item in self.all_tree.get_children():
            self.all_tree.delete(item)

        if not self.traced_signals:  # Use traced_signals for this tab
            self.all_tree.insert("", "end",
                                 values=("No signals found in source JSONs.",) * len(self.all_tree['columns']))
            return

        for signal_entry in self.traced_signals:  # Iterate over all traced signals
            signal_id = signal_entry.get(
                'signal_id')  # Keep signal_id for internal logic if needed, but not for display
            signal_type = signal_entry.get('signal_type', 'N/A')

            monitored_status_flag = signal_entry.get('is_monitored', False)
            display_monitoring_status = signal_entry.get('monitoring_status', 'N/A')

            # Calculate PnL percentage for display in Trade History
            display_pnl_percentage = 'N/A'
            if monitored_status_flag:
                # For closed trades, use the PnL % stored at closure
                if display_monitoring_status.startswith("CLOSED_SL") or display_monitoring_status.startswith(
                        "CLOSED_TP"):
                    # Ensure there's a space before splitting
                    if ' ' in display_monitoring_status:
                        display_pnl_percentage = display_monitoring_status.split(' ')[1].strip(
                            '()')  # Extract percentage from status string
                    else:  # Fallback if for some reason the format is not as expected
                        display_pnl_percentage = f"{signal_entry.get('profit_loss_percentage', 0):.2f}%"
                else:  # For OPEN or PARTIALLY_CLOSED trades, show current profit_loss_percentage
                    display_pnl_percentage = f"{signal_entry.get('profit_loss_percentage', 0):.2f}%"

            display_exit_price = 'N/A'
            if signal_entry.get('exit_price') is not None and monitored_status_flag:
                display_exit_price = f"{signal_entry.get('exit_price', 0):.6f}"

            display_exit_time = 'N/A'
            if signal_entry.get('exit_time') is not None and monitored_status_flag:
                display_exit_time = signal_entry.get('exit_time', 'N/A').split('T')[0] + ' ' + \
                                    signal_entry.get('exit_time', 'N/A').split('T')[1][:8]

            tag = 'not_monitored'
            if monitored_status_flag:
                if display_monitoring_status.startswith('CLOSED_TP'):
                    tag = 'monitored_success'
                elif display_monitoring_status.startswith('CLOSED_SL'):
                    tag = 'monitored_failure'
                elif display_monitoring_status in ['OPEN', 'PARTIALLY_CLOSED']:
                    tag = 'monitored_open'

            self.all_tree.insert("", "end", values=(
                signal_entry.get('strategy_name', 'N/A'),
                signal_entry.get('symbol'),
                signal_type,
                f"{signal_entry.get('entry_price', 0):.6f}",
                f"{signal_entry.get('entry_time', 'N/A').split('T')[0]} {signal_entry.get('entry_time', 'N/A').split('T')[1][:8]}",
                # Signal Time
                "Yes" if monitored_status_flag else "No",
                display_monitoring_status,
                display_pnl_percentage,
                display_exit_price,
                display_exit_time
            ), tags=(tag,))

    def on_closing(self):
        """Handles the window closing event to stop the monitoring thread."""
        print("Closing application. Stopping monitoring thread...")
        self._stop_event.set()  # Signal the thread to stop
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)  # Give thread some time to finish
            if self._monitoring_thread.is_alive():
                print("Warning: Monitoring thread did not terminate gracefully.")
        self.destroy()  # Destroy the Tkinter window


# --- Main execution block ---
if __name__ == "__main__":
    # In a real scenario, these JSON files are expected to be generated by your trading bot.
    # The application will read from them if they exist, or start with empty data otherwise.
    # For initial testing without a bot, you would manually create these files.

    app = TradeDashboard()
    app.mainloop()
