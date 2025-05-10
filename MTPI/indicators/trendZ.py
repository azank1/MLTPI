# indicators/trendZ.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "trendZ_settings.json"

# === Settings Loader ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core trendZ Computation ===
def compute_trendZ(df, period, smoothing_factor, bias_thresh):
    """
    trendZ: slope of smoothed rolling mean vs bias threshold.
    """
    price = df["close"]
    trend = price.rolling(window=period).mean()
    smoothed = trend.ewm(alpha=smoothing_factor).mean()
    slope = smoothed.diff()

    signal = np.where(slope > bias_thresh, 1, -1)
    return pd.Series(signal, index=df.index)

# === Final Signal Interface ===
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public method to generate trendZ signals.
    """
    settings = load_settings(settings)

    period = int(settings.get("period", 14))
    smoothing_factor = float(settings.get("smoothing_factor", 0.1))
    bias_thresh = float(settings.get("bias_thresh", 0.0))

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

    signal_series = compute_trendZ(df_resampled, period, smoothing_factor, bias_thresh)
    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian optimization to find best trendZ parameters.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(period, smoothing_factor, bias_thresh):
        period = int(round(period))
        smoothing_factor = float(smoothing_factor)
        bias_thresh = float(bias_thresh)

        signal = compute_trendZ(df, period, smoothing_factor, bias_thresh)
        signal = signal.fillna(-1).astype(int)

        target_signal = df["manual_signal"].astype(int).values
        mae = compute_mae(signal.values, target_signal)
        penalty = compute_transition_penalty(signal.values)

        return -(mae + penalty)

    pbounds = {
        "period": (5, 50),
        "smoothing_factor": (0.01, 0.5),
        "bias_thresh": (-0.5, 0.5)
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
        "period": int(round(best["period"])),
        "smoothing_factor": float(best["smoothing_factor"]),
        "bias_thresh": float(best["bias_thresh"])
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_settings, f, indent=4)

    print(f"✅ trendZ training complete. Best settings saved to {output_path}")
