# indicators/agma.py

import os
import json
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
SETTINGS_DIR = "settings"
SETTINGS_FILE = "agma_settings.json"

# === Settings Loader ===
def load_settings(settings=None):
    if settings is None:
        with open(os.path.join(SETTINGS_DIR, SETTINGS_FILE), "r") as f:
            settings = json.load(f)
    return settings

# === Core AGMA Computation ===
def compute_agma(df, length, adaptive, volatilityPeriod, sigma_fixed):
    """
    Compute AGMA values based on close price and provided parameters.
    """
    close = df["close"]
    agma_values = [np.nan] * (length - 1)

    for i in range(length - 1, len(df)):
        window = close.iloc[i - length + 1: i + 1].values
        if adaptive:
            start = max(0, i - volatilityPeriod + 1)
            sigma = np.std(close.iloc[start: i + 1])
            if sigma == 0:
                sigma = sigma_fixed
        else:
            sigma = sigma_fixed

        sum_val = 0.0
        sum_w = 0.0
        for j in range(length):
            weight = np.exp(-((j - (length - 1)) / (2 * sigma))**2 / 2)
            val_high = np.max(window[:j + 1])
            val_low = np.min(window[:j + 1])
            value = val_high + val_low
            sum_val += value * weight
            sum_w += weight

        agma = (sum_val / sum_w) / 2
        agma_values.append(agma)

    return pd.Series(agma_values, index=df.index)

# === Final Signal Interface ===
def final_signal(df, timeframe="1D", settings=None):
    """
    Main public method to generate AGMA signals.
    """
    settings = load_settings(settings)

    length = int(settings.get("length", 20))
    volatilityPeriod = int(settings.get("volatilityPeriod", 10))
    sigma_fixed = float(settings.get("sigma_fixed", 1.0))
    adaptive = bool(settings.get("adaptive", True))
    thresh = float(settings.get("thresh", 0.0))

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

    agma_series = compute_agma(df_resampled, length, adaptive, volatilityPeriod, sigma_fixed)

    # Generate binary signal
    binary_signal = np.where(df_resampled["close"] >= (agma_series + thresh), 1, -1)
    signal_series = pd.Series(binary_signal, index=df_resampled.index)

    aligned_signal = signal_series.reindex(df.index, method='ffill').fillna(-1).astype(int)
    return aligned_signal.values

# === Training Interface ===
def train_indicator(df, output_path):
    """
    Bayesian optimization to find best AGMA parameters.
    Saves the settings to output_path.
    """
    def compute_mae(signal, target):
        return np.mean(np.abs(signal - target))

    def compute_transition_penalty(signal, penalty_coef=0.1):
        transitions = np.sum(np.diff(signal) != 0)
        return penalty_coef * transitions / (len(signal) - 1)

    def objective(length, volatilityPeriod, sigma_fixed, adaptive_flag, thresh):
        length = int(round(length))
        volatilityPeriod = int(round(volatilityPeriod))
        adaptive = adaptive_flag > 0.5
        sigma_fixed = float(sigma_fixed)
        thresh = float(thresh)

        agma_series = compute_agma(df, length, adaptive, volatilityPeriod, sigma_fixed)
        signal = np.where(df["close"] >= (agma_series + thresh), 1, -1)
        signal = pd.Series(signal, index=df.index).fillna(-1).astype(int)

        target_signal = df["manual_signal"].astype(int).values
        mae = compute_mae(signal.values, target_signal)
        penalty = compute_transition_penalty(signal.values)

        return -(mae + penalty)

    pbounds = {
        "length": (5, 50),
        "volatilityPeriod": (5, 30),
        "sigma_fixed": (0.1, 5.0),
        "adaptive_flag": (0, 1),
        "thresh": (-0.5, 0.5)
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=30)

    best = optimizer.max["params"]
    best_settings = {
        "length": int(round(best["length"])),
        "volatilityPeriod": int(round(best["volatilityPeriod"])),
        "sigma_fixed": float(best["sigma_fixed"]),
        "adaptive": bool(best["adaptive_flag"] > 0.5),
        "thresh": float(best["thresh"])
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_settings, f, indent=4)

    print(f"✅ AGMA training complete. Best settings saved to {output_path}")
