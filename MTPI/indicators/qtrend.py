# indicators/qtrend.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "qtrend_settings.json"

# === Load Settings ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core Q-Trend Logic ===
def compute_qtrend(df, trend_period, atr_period, atr_mult, mode, use_ema, ema_period, source="close"):
    """Compute Q-Trend signal based on source price."""
    if source not in df.columns:
        if source == "hl2":
            src = (df["high"] + df["low"]) / 2
        elif source == "hlc3":
            src = (df["high"] + df["low"] + df["close"]) / 3
        elif source == "ohlc4":
            src = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        else:
            src = df["close"]
    else:
        src = df[source]

    if use_ema:
        src = src.ewm(span=ema_period, adjust=False).mean()

    h = src.rolling(trend_period).max()
    l = src.rolling(trend_period).min()
    d = h - l
    m = (h + l) / 2

    atr = (df["high"] - df["low"]).rolling(atr_period).mean()
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
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public entry for Q-Trend signal generation.
    Args:
        df (pd.DataFrame): Price data.
        timeframe (str): Timeframe like '1D'.
        settings (dict or None): If None, loads from disk.
    Returns:
        np.ndarray: Final signal array (-1, 1).
    """
    settings = load_settings(settings)

    trend_period = int(settings.get("trend_period", 200))
    atr_period = int(settings.get("atr_period", 14))
    atr_mult = float(settings.get("atr_mult", 1.0))
    mode = settings.get("mode", "Type A")
    use_ema = bool(settings.get("use_ema", False))
    ema_period = int(settings.get("ema_period", 3))
    source = settings.get("source", "close")

    # Resample if needed
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

    signal_series = compute_qtrend(
        df_resampled,
        trend_period,
        atr_period,
        atr_mult,
        mode,
        use_ema,
        ema_period,
        source
    )

    # Realign to original dataframe
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian Optimization of Q-Trend settings.
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
        "mode_flag": (0, 1),
        "use_ema_flag": (0, 1),
        "ema_period": (2, 10),
        "source_flag": (0, 6)
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
