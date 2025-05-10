# indicators/gstX.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "gstX_settings.json"

# === Settings Loader ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core gstX Computation ===
def compute_gstX(df, window, smooth_factor, directional_bias):
    """
    gstX: smoothed directional gradient indicator.
    """
    raw_diff = df["close"].diff(periods=int(window))
    smoothed = raw_diff.ewm(alpha=smooth_factor).mean()
    signal = np.where(smoothed > directional_bias, 1, -1)
    return pd.Series(signal, index=df.index)

# === Final Signal Interface ===
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public method to generate gstX signals.
    """
    settings = load_settings(settings)

    window = int(settings.get("window", 5))
    smooth_factor = float(settings.get("smooth_factor", 0.1))
    directional_bias = float(settings.get("directional_bias", 0.0))

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

    signal_series = compute_gstX(df_resampled, window, smooth_factor, directional_bias)
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian optimization to find best gstX parameters.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(window, smooth_factor, directional_bias):
        window = int(round(window))
        smooth_factor = float(smooth_factor)
        directional_bias = float(directional_bias)

        signal = compute_gstX(df, window, smooth_factor, directional_bias)
        signal = signal.fillna(-1).astype(int)

        target_signal = df["manual_signal"].astype(int).values
        mae = compute_mae(signal.values, target_signal)
        penalty = compute_transition_penalty(signal.values)

        return -(mae + penalty)

    pbounds = {
        "window": (3, 30),
        "smooth_factor": (0.01, 0.5),
        "directional_bias": (-1.0, 1.0)
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
        "window": int(round(best["window"])),
        "smooth_factor": float(best["smooth_factor"]),
        "directional_bias": float(best["directional_bias"])
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_settings, f, indent=4)

    print(f"✅ gstX training complete. Best settings saved to {output_path}")
