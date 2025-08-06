import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set up matplotlib for better looking charts
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10


def calculate_indicators(df):
    """Calculate all technical indicators"""
    # VWAP (simplified for last 24 periods)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['price_volume'] = df['typical_price'] * df['volume']
    df['vwap'] = df['price_volume'].rolling(24).sum() / df['volume'].rolling(24).sum()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14).mean()
    avg_loss = loss.ewm(span=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    # EMA50
    df['ema50'] = df['close'].ewm(span=50).mean()

    # Volume ratio
    df['volume_avg'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_avg']

    return df


def check_signal_conditions(row, prev_row):
    """Check if signal conditions are met (same logic as your code)"""
    current_price = row['close']
    vwap = row['vwap']
    rsi = row['rsi']
    macd_hist = row['macd_histogram']
    prev_macd_hist = prev_row['macd_histogram'] if prev_row is not None else 0
    volume_ratio = row['volume_ratio']
    ema50 = row['ema50']

    # Filters from your code
    vwap_distance = abs(current_price - vwap) / vwap * 100
    if vwap_distance < 0.2:
        return None, "VWAP distance too small"

    if not (rsi < 45 or rsi > 55):
        return None, "RSI in neutral zone"

    if abs(macd_hist) < 0.01:
        return None, "MACD histogram too small"

    if volume_ratio < 1.5:
        return None, "Volume too low"

    # BUY conditions
    buy_conditions = (
            current_price > vwap and
            rsi > 50 and
            macd_hist > 0 and
            macd_hist > prev_macd_hist and
            volume_ratio > 1.5 and
            current_price > ema50
    )

    # SELL conditions
    sell_conditions = (
            current_price < vwap and
            rsi < 50 and
            macd_hist < 0 and
            macd_hist < prev_macd_hist and
            volume_ratio > 1.5 and
            current_price < ema50
    )

    if buy_conditions:
        return "BUY", "All BUY conditions met"
    elif sell_conditions:
        return "SELL", "All SELL conditions met"
    else:
        return None, "Signal conditions not aligned"


def create_sample_data(pair_name, scenario, hours=100):
    """Create realistic sample data"""
    np.random.seed(42)

    # Base prices
    if "BTC" in pair_name:
        base_price = 42000
    elif "ETH" in pair_name:
        base_price = 2600
    else:  # ADA
        base_price = 0.485

    dates = [datetime.now() - timedelta(hours=hours - i - 1) for i in range(hours)]

    if scenario == "buy_signal":
        # Create uptrend that triggers BUY
        trend = np.linspace(-0.05, 0.08, hours)  # Upward trend
        noise = np.random.normal(0, 0.02, hours)
        price_changes = trend + noise

        # High volume near the end
        volumes = [150 + np.random.randint(-30, 30) for _ in range(hours - 10)]
        volumes += [200 + np.random.randint(-20, 40) for _ in range(10)]  # Higher volume

    elif scenario == "sell_signal":
        # Create downtrend that triggers SELL
        trend = np.linspace(0.05, -0.08, hours)  # Downward trend
        noise = np.random.normal(0, 0.02, hours)
        price_changes = trend + noise

        # High volume near the end
        volumes = [150 + np.random.randint(-30, 30) for _ in range(hours - 10)]
        volumes += [220 + np.random.randint(-20, 40) for _ in range(10)]  # Higher volume

    else:  # no_signal
        # Sideways movement with low volume
        price_changes = np.random.normal(0, 0.015, hours)
        volumes = [120 + np.random.randint(-20, 20) for _ in range(hours)]  # Low volume

    # Generate OHLCV data
    data = []
    current_price = base_price

    for i in range(hours):
        current_price *= (1 + price_changes[i])
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = data[-1]['close'] if data else current_price

        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': max(high, current_price, open_price),
            'low': min(low, current_price, open_price),
            'close': current_price,
            'volume': volumes[i]
        })

    return pd.DataFrame(data)


def plot_trading_signals(pair_name, scenario_name, scenario_type):
    """Create comprehensive trading visualization"""
    print(f"\n🔍 Analyzing {pair_name} - {scenario_name}")
    print("=" * 60)

    # Create data and calculate indicators
    df = create_sample_data(pair_name, scenario_type)
    df = calculate_indicators(df)

    # Find signals
    signals = []
    for i in range(50, len(df)):  # Start after indicators are stable
        prev_row = df.iloc[i - 1] if i > 0 else None
        signal_type, reason = check_signal_conditions(df.iloc[i], prev_row)

        if signal_type:
            signals.append({
                'timestamp': df.iloc[i]['timestamp'],
                'price': df.iloc[i]['close'],
                'type': signal_type,
                'index': i
            })
            print(
                f"🚨 {signal_type} SIGNAL at {df.iloc[i]['timestamp'].strftime('%H:%M')} - Price: ${df.iloc[i]['close']:.4f}")

    if not signals:
        print("❌ No signals found - conditions not met")

    # Create the visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'{pair_name} - {scenario_name}', fontsize=16, fontweight='bold')

    # 1. Price Chart with VWAP and EMA50
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['close'], color='#00D4AA', linewidth=2, label='Price')
    ax1.plot(df['timestamp'], df['vwap'], color='#FF6B6B', linestyle='--', linewidth=1.5, label='VWAP')
    ax1.plot(df['timestamp'], df['ema50'], color='#4ECDC4', linestyle='--', linewidth=1.5, label='EMA50')

    # Mark signals on price chart
    for signal in signals:
        color = '#00FF00' if signal['type'] == 'BUY' else '#FF0000'
        marker = '^' if signal['type'] == 'BUY' else 'v'
        ax1.scatter(signal['timestamp'], signal['price'], color=color, s=200, marker=marker,
                    label=f"{signal['type']} Signal", zorder=5)

    ax1.set_title('Price Action with Indicators')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price ($)')

    # 2. RSI
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['rsi'], color='#9B59B6', linewidth=2, label='RSI')
    ax2.axhline(y=70, color='#FF6B6B', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=50, color='#95A5A6', linestyle='-', alpha=0.5, label='Neutral (50)')
    ax2.axhline(y=30, color='#2ECC71', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI (14)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('RSI')

    # 3. MACD Histogram
    ax3 = axes[2]
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in df['macd_histogram']]
    ax3.bar(df['timestamp'], df['macd_histogram'], color=colors, alpha=0.8, width=0.02)
    ax3.axhline(y=0, color='white', linestyle='-', alpha=0.8)
    ax3.plot(df['timestamp'], df['macd_line'], color='#3498DB', linewidth=1, label='MACD Line')
    ax3.plot(df['timestamp'], df['macd_signal'], color='#F39C12', linewidth=1, label='Signal Line')
    ax3.set_title('MACD Histogram')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('MACD')

    # 4. Volume Ratio
    ax4 = axes[3]
    volume_colors = ['#2ECC71' if x > 1.5 else '#95A5A6' for x in df['volume_ratio']]
    ax4.bar(df['timestamp'], df['volume_ratio'], color=volume_colors, alpha=0.8, width=0.02)
    ax4.axhline(y=1.5, color='#FF6B6B', linestyle='--', linewidth=2, label='Min Threshold (1.5x)')
    ax4.set_title('Volume Ratio (Current / 20-period Average)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel('Volume Ratio')
    ax4.set_xlabel('Time')

    # Format x-axis for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    return fig, signals


def run_visualization_demo():
    """Run the complete visualization demo"""
    print("🤖 TRADING SIGNAL VISUALIZATION DEMO")
    print("=" * 80)
    print("This creates charts showing exactly how your trading bot works\n")

    # Three scenarios to demonstrate
    scenarios = [
        ("BTC/USDT", "BUY Signal Formation", "buy_signal"),
        ("ETH/USDT", "SELL Signal Formation", "sell_signal"),
        ("ADA/USDT", "No Signal (Low Volume)", "no_signal")
    ]

    all_signals = []

    for pair, name, scenario_type in scenarios:
        fig, signals = plot_trading_signals(pair, name, scenario_type)
        all_signals.extend(signals)
        plt.show()

        # Print signal analysis
        if signals:
            for signal in signals:
                print(f"✅ {signal['type']} SIGNAL: {pair} at ${signal['price']:.6f}")
        else:
            print(f"❌ No signals for {pair} - this is typical!")
        print()

    # Summary
    print("📊 VISUALIZATION SUMMARY")
    print("=" * 40)
    print(f"Total signals found: {len(all_signals)}")
    print(f"Signal rate: {len(all_signals) / 3 * 100:.1f}% (3 pairs scanned)")
    print("\n💡 Key Observations:")
    print("• Green/Red arrows show exact signal points")
    print("• Volume bars turn green when >1.5x (required for signals)")
    print("• MACD histogram color shows momentum direction")
    print("• All 5 conditions must align perfectly for signals")
    print("• Your real bot scans 800+ pairs - finding 0-2 signals is normal!")


if __name__ == "__main__":
    # Install required packages if needed
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Please install required packages:")
        print("pip install matplotlib pandas numpy")
        exit(1)

    run_visualization_demo()