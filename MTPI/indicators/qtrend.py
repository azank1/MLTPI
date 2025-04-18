import os
import json
from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

# === Settings Loader ===
def load_json_settings(settings_path):
    with open(settings_path, 'r') as f:
        return json.load(f)

# === Core Q-Trend Logic ===
def compute_qtrend(df, trend_period, atr_period, atr_mult, mode, use_ema, ema_period):
    """
    Compute Q-Trend signal core (without resampling).
    
    Returns:
        pd.Series of +1/-1 signals indexed to df
    """
    src = df["close"]
    if use_ema:
        src = src.ewm(span=ema_period, adjust=False).mean()

    h = src.rolling(trend_period).max()
    l = src.rolling(trend_period).min()
    d = h - l
    m = (h + l) / 2

    atr = df["high"].subtract(df["low"]).rolling(atr_period).mean()
    epsilon = atr_mult * atr

    if mode == "Type B":
        change_up = ((src.shift(1) < m + epsilon) & (src >= m + epsilon)) | (src > m + epsilon)
        change_down = ((src.shift(1) > m - epsilon) & (src <= m - epsilon)) | (src < m - epsilon)
    else:
        change_up = ((src.shift(1) < m + epsilon) & (src >= m + epsilon))
        change_down = ((src.shift(1) > m - epsilon) & (src <= m - epsilon))

    signal = np.where(change_up, 1, np.where(change_down, -1, np.nan))
    signal = pd.Series(signal, index=df.index).ffill().fillna(-1).astype(int)
    return signal

# === Final Signal Interface ===
def final_signal(df, timeframe="1D"):
    settings_path = os.path.join("settings", "qtrend_settings.json")
    settings = load_json_settings(settings_path)

    trend_period = int(settings.get("trend_period", 200))
    atr_period = int(settings.get("atr_period", 14))
    atr_mult = float(settings.get("atr_mult", 1.0))
    mode = settings.get("mode", "Type A")
    use_ema = settings.get("use_ema", False)
    ema_period = int(settings.get("ema_period", 3))

    if timeframe != "1D":
        df_resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'manual_signal': 'last'
        }).dropna()
    else:
        df_resampled = df.copy()

    signal_series = compute_qtrend(df_resampled, trend_period, atr_period, atr_mult, mode, use_ema, ema_period)
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

def train_indicator(df, output_path):
    """
    Trains Q-Trend using Bayesian Optimization.
    Optimizes all relevant inputs including source.
    Saves best config to output_path.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(trend_period, atr_period, atr_mult, mode_flag, use_ema_flag, ema_period, source_flag):
        try:
            trend_period = int(round(trend_period))
            atr_period = int(round(atr_period))
            ema_period = int(round(ema_period))
            mode = "Type B" if mode_flag > 0.5 else "Type A"
            use_ema = bool(use_ema_flag > 0.5)

            source_map = ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]
            source = source_map[int(round(source_flag))]

            signal_series = compute_qtrend(
                df.copy(), trend_period, atr_period, atr_mult, mode, use_ema, ema_period, source
            )
            isp = df["manual_signal"].astype(int).values

            mae = compute_mae(signal_series, isp)
            penalty = compute_transition_penalty(signal_series)
            return -(mae + penalty)
        except Exception:
            return -100

    pbounds = {
        "trend_period": (50, 250),
        "atr_period": (5, 30),
        "atr_mult": (0.5, 3.0),
        "mode_flag": (0, 1),        # Type A / B
        "use_ema_flag": (0, 1),     # No / Yes
        "ema_period": (2, 10),
        "source_flag": (0, 6)       # close to ohlc4
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=30)

    best = optimizer.max["params"]
    source_map = ["close", "open", "high", "low", "hl2", "hlc3", "ohlc4"]

    settings = {
        "trend_period": int(round(best["trend_period"])),
        "atr_period": int(round(best["atr_period"])),
        "atr_mult": float(best["atr_mult"]),
        "mode": "Type B" if best["mode_flag"] > 0.5 else "Type A",
        "use_ema": bool(best["use_ema_flag"] > 0.5),
        "ema_period": int(round(best["ema_period"])),
        "source": source_map[int(round(best["source_flag"]))]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(settings, f, indent=4)

    print(f"✅ Q-Trend training complete. Settings saved to {output_path}")
