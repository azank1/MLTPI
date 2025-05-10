# indicators/zscoreMA.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "zscoreMA_settings.json"

# === Settings Loader ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core zscoreMA Computation ===
def compute_zscoreMA(df, ma_period, z_thresh, use_ema):
    """
    zscoreMA: z-score of price from moving average threshold.
    """
    price = df["close"]

    if use_ema:
        ma = price.ewm(span=ma_period, adjust=False).mean()
    else:
        ma = price.rolling(window=ma_period).mean()

    std = price.rolling(window=ma_period).std().replace(0, 1e-8)
    zscore = (price - ma) / std

    signal = np.where(zscore > z_thresh, 1, np.where(zscore < -z_thresh, -1, 0))
    return pd.Series(signal, index=df.index)

# === Final Signal Interface ===
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public method to generate zscoreMA signals.
    """
    settings = load_settings(settings)

    ma_period = int(settings.get("ma_period", 20))
    z_thresh = float(settings.get("z_thresh", 1.0))
    use_ema = bool(settings.get("use_ema", False))

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

    signal_series = compute_zscoreMA(df_resampled, ma_period, z_thresh, use_ema)
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian optimization to find best zscoreMA parameters.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(ma_period, z_thresh, use_ema_flag):
        ma_period = int(round(ma_period))
        z_thresh = float(z_thresh)
        use_ema = bool(use_ema_flag > 0.5)

        signal = compute_zscoreMA(df, ma_period, z_thresh, use_ema)
        signal = signal.fillna(-1).astype(int)

        target_signal = df["manual_signal"].astype(int).values
        mae = compute_mae(signal.values, target_signal)
        penalty = compute_transition_penalty(signal.values)

        return -(mae + penalty)

    pbounds = {
        "ma_period": (5, 50),
        "z_thresh": (0.5, 3.0),
        "use_ema_flag": (0, 1)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=25)

    best = optimizer.max["params"]
    best_settings = {
        "ma_period": int(round(best["ma_period"])),
        "z_thresh": float(best["z_thresh"]),
        "use_ema": bool(best["use_ema_flag"] > 0.5)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_settings, f, indent=4)

    print(f"✅ zscoreMA training complete. Best settings saved to {output_path}")
