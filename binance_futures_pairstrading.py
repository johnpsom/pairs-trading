

import os
import time
from datetime import datetime
import warnings
from itertools import combinations
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from binance.exceptions import BinanceAPIException
from binance.client import Client

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# API keys - Ensure these are set as environment variables or replace with your keys carefully
api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')
if not api_key or not api_secret:
    print("Error: Binance API keys not found in environment variables.")
    exit()

# Initialize Binance client
client = Client(api_key, api_secret)


def get_usdc_perpetual_symbols():
    """
    Fetch all symbols in the USDC perpetual futures market.
    """
    try:
        exchange_info = client.futures_exchange_info()
        usdc_symbols = []
        for symbol_info in exchange_info['symbols']:
            if symbol_info['quoteAsset'] == 'USDC' and symbol_info['contractType'] == 'PERPETUAL':
                usdc_symbols.append(symbol_info['symbol'])
        return usdc_symbols
    except BinanceAPIException as err:
        print(f"Error fetching USDC perpetual symbols: {err}")
        return []


def fetch_price_data(symbol, interval='15m', lookback='3 days ago UTC'):
    """
    Fetch historical price data for a given symbol.
    """
    try:
        df = pd.DataFrame(client.futures_historical_klines(
            symbol, interval, lookback)).iloc[:-1, :7]
        df.columns = ["datetime_open", "open", "high",
                      "low", "close", "volume", "datetime_close"]
        df['datetime_open'] = pd.to_datetime(df['datetime_open'], unit='ms')
        df['datetime_close'] = pd.to_datetime(df['datetime_close'], unit='ms')
        df['close'] = df['close'].astype(float)
        now = datetime.utcnow()
        df = df[df['datetime_close'] <= now]
        df['returns'] = df['close'].pct_change()
        return df.set_index('datetime_open')[['returns', 'close']].dropna()
    except BinanceAPIException as err:
        print(f"Error fetching price data for {symbol}: {err}")
        return pd.DataFrame()


def calculate_correlation(series1, series2):
    """
    Calculate the Pearson correlation between two series.
    """
    return series1.corr(series2)


def test_cointegration(pair1, pair2, data, significance_level=0.01):
    """
    Test for cointegration using Engle-Granger and calculate metrics including correlation.
    Returns: (is_cointegrated, p_value, hedge_ratio, half_life, correlation)
    Enhanced with detailed data validation and error handling.
    """
    series1 = data[pair1]['close']
    series2 = data[pair2]['close']

    # Initial data checks before alignment
    if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
        print(
            f"Error: Input data is not Pandas Series for pair {pair1}-{pair2}. Check data fetching.")
        return (False, 1.0, 0, np.inf, np.nan)

    if series1.empty or series2.empty:
        print(
            f"Warning: Initial series is empty for pair {pair1}-{pair2} before alignment. Check data availability.")
        return (False, 1.0, 0, np.inf, np.nan)

    # Align indices before any calculation
    series1, series2 = series1.align(series2, join='inner')

    print(f"Checking cointegration for pair: {pair1}, {pair2}")  # Debug print
    print(f"Series 1 head:\n{series1.head()}")  # Debug print
    print(f"Series 2 head:\n{series2.head()}")  # Debug print
    # Debug print
    print(f"Series 1 shape: {series1.shape}, Series 2 shape: {series2.shape}")

    # --- Robust checks for NaNs, Infs, and empty series AFTER alignment ---
    if series1.isnull().any() or series2.isnull().any():
        nan_count1 = series1.isnull().sum()
        nan_count2 = series2.isnull().sum()
        print(
            f"Warning: NaN values found in series for pair {pair1}-{pair2} AFTER alignment. Series 1 NaNs: {nan_count1}, Series 2 NaNs: {nan_count2}. Not checking cointegration.")
        return (False, 1.0, 0, np.inf, np.nan)  # Return not cointegrated, NaN correlation

    if not np.isfinite(series1).all() or not np.isfinite(series2).all():  # Check for Inf or -Inf
        inf_count1 = (~np.isfinite(series1)).sum()
        inf_count2 = (~np.isfinite(series2)).sum()
        print(
            f"Warning: Infinite values found in series for pair {pair1}-{pair2} AFTER alignment. Series 1 Infs: {inf_count1}, Series 2 Infs: {inf_count2}. Not checking cointegration.")
        return (False, 1.0, 0, np.inf, np.nan)  # Return not cointegrated, NaN correlation

    if series1.empty or series2.empty:
        print(
            f"Warning: Empty series after alignment for pair {pair1}-{pair2}. Alignment removed all data points. Check for index overlap and data ranges.")
        return (False, 1.0, 0, np.inf, np.nan)  # Return not cointegrated, NaN correlation
    # --- End of robust data checks ---

    correlation = calculate_correlation(series1, series2)

    try:
        model = sm.OLS(series1, sm.add_constant(series2)
                       ).fit()  # Regress series1 on series2
        residuals = model.resid
        result = adfuller(residuals)
        p_value = result[1]

        # Calculate half-life only if cointegrated
        half_life = np.inf
        if p_value < significance_level:
            ylag = residuals.shift(1).dropna()
            deltay = (residuals - ylag).dropna()
            ylag = ylag.loc[deltay.index]
            res = sm.OLS(deltay, sm.add_constant(ylag)).fit()
            slope = res.params.values[1]
            # Avoid non-mean-reverting cases
            half_life = (-np.log(2) / slope) if slope < 0 else np.inf

        # Return correlation
        return (p_value < significance_level, p_value, model.params[1], half_life, correlation)

    except Exception as e:
        # More specific error message
        print(
            f"Error checking cointegration for {pair1}-{pair2}: Exception during OLS/ADF test.")
        print(f"Error Type: {type(e)}, Error String: {str(e)}")  # Print exception details
        # Return NaN for correlation on error as well
        return (False, 1.0, 0, np.inf, np.nan)


def hurst_function(time_series):
    """Returns Hurst Exponent."""
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag])))
           for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return round(poly[0] * 2.0, 3)


def calculate_halflife(spread):
    """Calculate half-life of mean reversion."""
    ylag = spread.shift(1).dropna()
    deltay = (spread - ylag).dropna()
    ylag = ylag.loc[deltay.index]
    res = sm.OLS(deltay, sm.add_constant(ylag)).fit()
    slope = res.params[1]
    halflife = (-np.log(2) / slope)
    return int(halflife) if halflife > 0 else np.inf


def calculate_halflife2(spread):
    """Calculate half-life of mean reversion."""
    halflife = np.inf  # Default to infinite half-life in case of errors
    try:
        ylag = spread.shift(1).dropna()
        deltay = (spread - ylag).dropna()
        ylag = ylag.loc[deltay.index]
        res = sm.OLS(deltay, sm.add_constant(ylag)).fit()
        if len(res.params) > 1:  # Check if params has at least two elements (constant and ylag)
            slope = res.params[1]
            halflife = (-np.log(2) / slope) if slope < 0 else np.inf
        else:
            print("Warning: OLS regression in halflife calculation did not produce slope parameter. Returning infinite halflife.")
            halflife = np.inf  # Handle cases where slope param is not available
    except Exception as e:
        print(f"Error calculating halflife: Exception: {e}")
        return np.inf  # Return infinite halflife on exception

    # Ensure valid halflife or inf
    return int(halflife) if halflife != np.inf and halflife > 0 and np.isfinite(halflife) else np.inf


def calculate_spread(series1, series2):
    """Calculate spread and hedge ratio."""
    # Align series to ensure matching indices
    series1, series2 = series1.align(series2, join='inner')
    model = sm.OLS(series1, sm.add_constant(series2)).fit()
    hedge_ratio = model.params[1]
    spread = series1 - hedge_ratio * series2
    return spread, round(hedge_ratio, 3)


def calculate_z_score(spread):
    """Calculate z-score of spread."""
    mean = spread.mean()
    std = spread.std()
    return (spread - mean) / std


def get_tradeable_pairs(data):
    """Identify tradeable pairs based on cointegration and calculate scoring metrics."""
    significant_pairs = []
    pairs = list(combinations(data.keys(), 2))
    for pair in pairs:
        is_cointegrated, p_value, hedge_ratio, half_life, correlation = test_cointegration(
            pair[0], pair[1], data)
        if is_cointegrated:
            significant_pairs.append({
                'pair': pair,
                'p_value': p_value,
                'hedge_ratio': hedge_ratio,
                'half_life': half_life,
                'correlation': correlation
            })
    return significant_pairs


def select_best_pair(significant_pairs):
    """
    Select the best trading pair from significant pairs based on a composite score.
    """
    if not significant_pairs:
        return None

    # Avoid inf half-life in max calculation, handling cases where no pairs have finite half-life
    max_half_life = max(p['half_life']
                        for p in significant_pairs if p['half_life'] != np.inf)
    # Handle the case where all half-lives are infinite or zero (to prevent division by zero)
    if max_half_life == 0 or max_half_life == np.inf:
        max_half_life = 1  # Or another default value, consider what makes sense in your logic

    for pair_data in significant_pairs:
        # Composite score considers:
        # - Statistical significance (p-value)
        # - Speed of mean reversion (half-life)
        # - Strength of relationship (correlation)
        # Normalize half-life, cap at 1 if infinite
        half_life_score = (
            pair_data['half_life'] / max_half_life) if pair_data['half_life'] != np.inf else 1
        pair_data['score'] = (pair_data['p_value'] * 0.4 +
                              half_life_score * 0.3 +
                              (1 - abs(pair_data['correlation'])) * 0.3)

    best_pair_data = min(significant_pairs, key=lambda x: x['score'])
    return best_pair_data


def get_current_price(symbol):
    """Get last price of a trading pair on binance futures market."""
    ticker = client.futures_symbol_ticker(symbol=symbol)
    return float(ticker['price'])


def get_time_until_next_candle():
    """Calculate time remaining until the next 5-minute candle."""
    current_time = datetime.utcnow()
    minutes = current_time.minute
    seconds = current_time.second
    remaining_minutes = 5 - (minutes % 5)
    remaining_seconds = (remaining_minutes * 60) - seconds
    return remaining_seconds


def wait_for_next_candle():
    """Sleep until the next candle."""
    seconds_to_sleep = get_time_until_next_candle()
    print(f"Waiting for {seconds_to_sleep} seconds until the next 15-minute candle...")
    time.sleep(seconds_to_sleep)


# --- Trading Script Parameters ---
INITIAL_BALANCE = 1000  # Initial balance - adjust as needed
TRADE_BALANCE_PERCENT = 0.25  # Use 25% of balance for each pair trade
LEVERAGE = 20          # Leverage - adjust carefully
ENTRY_THRESHOLD = 2.5   # Z-score entry threshold (adjust based on backtesting)
EXIT_THRESHOLD = 0.5    # Z-score exit threshold (adjust based on backtesting)
TIME_BETWEEN_PAIR_CHECKS_SECONDS = 60  # Time to wait between checks when no trade is active

# --- Trading Script ---
if __name__ == "__main__":
    usdc_symbols = get_usdc_perpetual_symbols()
    print(f"USDC Perpetual Futures Symbols: {usdc_symbols}")

    in_trade = False
    trades = []
    pnl = []
    position = 0  # 0: no position, 1: long spread, -1: short spread
    entry_price1 = 0
    entry_price2 = 0
    quantity1 = 0
    quantity2 = 0
    asset1_symbol = ""
    asset2_symbol = ""

    balance = INITIAL_BALANCE  # Start with initial balance

    while True:
        if not in_trade:
            print('--- No Trade Active ---')
            print(f'Current Balance: {balance:.2f} USDC')
            print(
                f'Waiting {TIME_BETWEEN_PAIR_CHECKS_SECONDS} seconds before next pair check...')
            time.sleep(TIME_BETWEEN_PAIR_CHECKS_SECONDS)  # Wait a bit before next check
            print('Fetching price data and looking for trade opportunities...')

            price_data = {symbol: fetch_price_data(
                symbol, '5m', '15 days ago UTC') for symbol in usdc_symbols}

            # Get list of significant pairs with metrics
            significant_pairs_data = get_tradeable_pairs(price_data)

            if not significant_pairs_data:
                print("No cointegrated pairs found. Retrying...")
                continue  # Skip to the next iteration if no pairs found

            # Select best pair based on score
            best_pair_data = select_best_pair(significant_pairs_data)

            # Handle case if no best pair is found after scoring (though unlikely if significant_pairs_data is not empty)
            if best_pair_data is None:
                print("No best pair found after scoring. Retrying...")
                continue

            asset1_symbol = best_pair_data['pair'][0]
            asset2_symbol = best_pair_data['pair'][1]
            hedge_ratio = best_pair_data['hedge_ratio']

            # Get latest close prices for spread calculation
            series1 = price_data[asset1_symbol]['close'].tail(300)
            series2 = price_data[asset2_symbol]['close'].tail(300)

            # Align the series to ensure matching indices
            series1, series2 = series1.align(series2, join='inner')

            if series1.empty or series2.empty:
                print(
                    f"Warning: No overlapping data for {asset1_symbol} and {asset2_symbol} after alignment.")
                print(f"Series1 index: {series1.index}")
                print(f"Series2 index: {series2.index}")
                print(f"Series1 length: {len(series1)}, Series2 length: {len(series2)}")
                continue

            spread, _ = calculate_spread(series1, series2)
            z_score_value = calculate_z_score(spread).iloc[-1]  # Current Z-score

            hurst = hurst_function(spread.values)
            half_life = calculate_halflife(spread)

            print(
                f'Best Pair: ({asset1_symbol, asset2_symbol}), Z_Score: {z_score_value:.3f}, Hurst: {hurst}, Halflife: {half_life}, Score: {best_pair_data["score"]:.4f}')

            price1 = get_current_price(asset1_symbol)
            price2 = get_current_price(asset2_symbol)

            if z_score_value > ENTRY_THRESHOLD:  # Short spread
                entry_time = datetime.now()
                print(
                    f"Entering SHORT spread trade at {entry_time} with Z-Score: {z_score_value:.3f}")

                total_trade_balance = balance * TRADE_BALANCE_PERCENT  # Use a fraction of total balance
                total_position_size = total_trade_balance * LEVERAGE  # Position size with leverage

                # Normalize Notionals - using total_position_size based on allocated balance fraction
                notional1 = total_position_size * \
                    (price2 * hedge_ratio) / (price1 + (price2 * hedge_ratio))
                notional2 = total_position_size - notional1

                quantity1 = round(notional1 / price1, 5)
                quantity2 = round(notional2 / price2, 5)

                print(
                    f'SHORT: {quantity1} {asset1_symbol} @ {price1:.4f} (Notional: {round(notional1, 2)} USDC)')
                print(
                    f'LONG: {quantity2} {asset2_symbol} @ {price2:.4f} (Notional: {round(notional2, 2)} USDC)')

                position = -1  # Short spread
                entry_price1, entry_price2 = price1, price2

                trades.append({
                    'type': 'short',
                    'entry_time': entry_time,
                    'pair': (asset1_symbol, asset2_symbol),
                    'entry_z': z_score_value,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'quantity1': quantity1,
                    'quantity2': quantity2,
                    'hedge_ratio': hedge_ratio
                })
                in_trade = True

            elif z_score_value < -ENTRY_THRESHOLD:  # Long spread
                entry_time = datetime.now()
                print(
                    f"Entering LONG spread trade at {entry_time} with Z-Score: {z_score_value:.3f}")

                total_trade_balance = balance * TRADE_BALANCE_PERCENT  # Use a fraction of total balance
                total_position_size = total_trade_balance * LEVERAGE  # Position size with leverage

                # Normalize Notionals - using total_position_size based on allocated balance fraction
                notional1 = total_position_size * \
                    (price1) / (price1 + (price2 * hedge_ratio))
                notional2 = total_position_size - notional1

                quantity1 = round(notional1 / price1, 5)
                quantity2 = round(notional2 / price2, 5)

                print(
                    f'LONG: {quantity1} {asset1_symbol} @ {price1:.4f} (Notional: {round(notional1, 2)} USDC)')
                print(
                    f'SHORT: {quantity2} {asset2_symbol} @ {price2:.4f} (Notional: {round(notional2, 2)} USDC)')

                position = 1  # Long spread
                entry_price1, entry_price2 = price1, price2
                trades.append({
                    'type': 'long',
                    'entry_time': entry_time,
                    'pair': (asset1_symbol, asset2_symbol),
                    'entry_z': z_score_value,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'quantity1': quantity1,
                    'quantity2': quantity2,
                    'hedge_ratio': hedge_ratio
                })
                in_trade = True
            else:
                print("No entry signal found.")

        if in_trade:
            print('\n--- Trade Active ---')
            print(
                f'Pair: ({asset1_symbol, asset2_symbol}), Position Type: {"Long" if position == 1 else "Short"} Spread')
            wait_for_next_candle()
            prices1 = fetch_price_data(asset1_symbol)['close']
            prices2 = fetch_price_data(asset2_symbol)['close']
            prices1, prices2 = prices1.align(prices2, join='inner')
            # hedge_ratio not needed here, already recorded
            spread, _ = calculate_spread(prices1, prices2)
            z_score_value = calculate_z_score(spread).iloc[-1]

            price1 = get_current_price(asset1_symbol)
            price2 = get_current_price(asset2_symbol)

            if (position == 1 and z_score_value > EXIT_THRESHOLD) or (position == -1 and z_score_value < -EXIT_THRESHOLD) or (abs(z_score_value) < EXIT_THRESHOLD):  # Exit trade

                exit_time = datetime.now()
                print(
                    f"Exiting trade at {exit_time}, Z-Score: {z_score_value:.3f}, Prices: {asset1_symbol} {price1:.4f}, {asset2_symbol} {price2:.4f}")

                if position == 1:  # Long spread (long asset1, short asset2)
                    pnl_asset1 = quantity1 * (price1 - entry_price1)  # Long asset1 PnL
                    pnl_asset2 = quantity2 * (entry_price2 - price2)  # Short asset2 PnL
                    trade_pnl = pnl_asset1 + pnl_asset2
                elif position == -1:  # Short spread (short asset1, long asset2)
                    pnl_asset1 = quantity1 * (entry_price1 - price1)  # Short asset1 PnL
                    pnl_asset2 = quantity2 * (price2 - entry_price2)  # Long asset2 PnL
                    trade_pnl = pnl_asset1 + pnl_asset2
                else:
                    trade_pnl = 0  # Should not happen, but just in case

                pnl.append(trade_pnl)
                balance += trade_pnl

                trades[-1].update({
                    'exit_time': exit_time,
                    'exit_z': z_score_value,
                    'exit_price1': price1,
                    'exit_price2': price2,
                    'pnl': trade_pnl
                })

                in_trade = False
                position = 0  # Reset position

                # Performance metrics on trade close
                total_pnl = sum(pnl)
                win_rate = len([p for p in pnl if p > 0]) / len(pnl) if pnl else 0
                returns = pd.Series(pnl)
                sharpe_ratio = returns.mean() / returns.std() if not returns.empty else 0
                max_drawdown = max(
                    (returns.cumsum().max() - returns.cumsum().min()), 0) if not returns.empty else 0

                print(f"\n--- Trade Closed ---")
                print(
                    f"Pair: ({asset1_symbol, asset2_symbol}), Type: {'Long' if trades[-1]['type'] == 'long' else 'Short'} Spread")
                print(
                    f"Entry Time: {trades[-1]['entry_time'].strftime('%Y-%m-%d %H:%M')}, Exit Time: {trades[-1]['exit_time'].strftime('%Y-%m-%d %H:%M')}")
                # Exit z-score might be slightly off due to time passing
                print(
                    f"Entry Z-Score: {trades[-1]['entry_z']:.3f}, Exit Z-Score: {trades[-1]['exit_z']:.3f}")
                print(
                    f"Entry Prices: {asset1_symbol} {trades[-1]['entry_price1']:.4f}, {asset2_symbol} {trades[-1]['entry_price2']:.4f}")
                # Exit prices might be slightly off due to time passing between signal and execution in real scenario
                print(
                    f"Exit Prices: {asset1_symbol} {trades[-1]['exit_price1']:.4f}, {asset2_symbol} {trades[-1]['exit_price2']:.4f}")
                print(f"Trade P&L: {trade_pnl:.2f} USDC")
                print(f"Current Balance: {balance:.2f} USDC")

                print(f"\n--- Performance Summary ---")
                print(f"Total Trades: {len(trades)}")
                print(f"Total PnL: {total_pnl:.2f} USDC")
                print(f"Win Rate: {win_rate:.2%}")
                print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
                print(f"Maximum Drawdown: {max_drawdown:.2f} USDC")
                print("-" * 30)

            else:
                if position == 1:  # Long spread (long asset1, short asset2)
                    pnl_asset1 = quantity1 * (price1 - entry_price1)  # Long asset1 PnL
                    pnl_asset2 = quantity2 * (entry_price2 - price2)  # Short asset2 PnL
                    trade_pnl = pnl_asset1 + pnl_asset2
                elif position == -1:  # Short spread (short asset1, long asset2)
                    pnl_asset1 = quantity1 * (entry_price1 - price1)  # Short asset1 PnL
                    pnl_asset2 = quantity2 * (price2 - entry_price2)  # Long asset2 PnL
                    trade_pnl = pnl_asset1 + pnl_asset2
                else:
                    trade_pnl = 0

                time_diff_minutes = (
                    datetime.now() - trades[-1]['entry_time']).total_seconds() / 60
                print(
                    f'Current PnL of the trade ({asset1_symbol, asset2_symbol}) is {trade_pnl:.2f} USDC after {time_diff_minutes:.2f} minutes, Z-score: {z_score_value:.3f}')
