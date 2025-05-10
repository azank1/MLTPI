# indicators/momentumX.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "momentumX_settings.json"

# === Settings Loader ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core MomentumX Computation ===
def compute_momentumX(df, fast_window, slow_window, threshold):
    """
    MomentumX signal based on difference between fast and slow momentum.
    """
    momentum_fast = df["close"].diff(int(fast_window))
    momentum_slow = df["close"].diff(int(slow_window))
    momentum_diff = momentum_fast - momentum_slow

    signal = np.where(momentum_diff > threshold, 1, -1)
    return pd.Series(signal, index=df.index)

# === Final Signal Interface ===
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public interface for MomentumX signal generation.
    """
    settings = load_settings(settings)

    fast_window = int(settings.get("fast_window", 5))
    slow_window = int(settings.get("slow_window", 20))
    threshold = float(settings.get("threshold", 0.0))

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

    signal_series = compute_momentumX(df_resampled, fast_window, slow_window, threshold)
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian optimization to find best MomentumX parameters.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(fast_window, slow_window, threshold):
        fast_window = int(round(fast_window))
        slow_window = int(round(slow_window))
        threshold = float(threshold)

        if fast_window >= slow_window:
            return -1e6  # Penalize invalid setting

        signal = compute_momentumX(df, fast_window, slow_window, threshold)
        signal = signal.fillna(-1).astype(int)

        target_signal = df["manual_signal"].astype(int).values
        mae = compute_mae(signal.values, target_signal)
        penalty = compute_transition_penalty(signal.values)

        return -(mae + penalty)

    pbounds = {
        "fast_window": (2, 10),
        "slow_window": (5, 30),
        "threshold": (-1.0, 1.0)
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
        "fast_window": int(round(best["fast_window"])),
        "slow_window": int(round(best["slow_window"])),
        "threshold": float(best["threshold"])
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_settings, f, indent=4)

    print(f"✅ MomentumX training complete. Best settings saved to {output_path}")
