import os
import json
import importlib
import numpy as np
import pandas as pd

FEATURES_PATH = "features/indicator_profiles.json"
SETTINGS_DIR = "settings"
DATA_PATH = "CSVdata/target.csv"
INDICATOR_NAMES = ["agma", "qtrend","gstX"]  # Add others like "kalman", "firefly", etc.

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def compute_sharpe(signal, prices):
    rets = prices.pct_change().fillna(0)
    active_rets = signal[:-1] * rets[1:]  # shift signal
    return np.mean(active_rets) / (np.std(active_rets) + 1e-8)

def compute_omega(signal, prices):
    rets = prices.pct_change().fillna(0)
    active = signal[:-1] * rets[1:]
    gain = np.mean(active[active > 0]) if np.any(active > 0) else 0
    loss = np.abs(np.mean(active[active < 0])) if np.any(active < 0) else 1e-8
    return gain / loss

def compute_transition_freq(signal):
    transitions = np.sum(np.diff(signal) != 0)
    return transitions / (len(signal) - 1)

def compute_holding_period(signal):
    transitions = np.where(np.diff(signal) != 0)[0]
    if len(transitions) < 2:
        return len(signal)
    holding_periods = np.diff(transitions)
    return np.mean(holding_periods)

def extract_features_for_indicator(name, df):
    module = importlib.import_module(f"indicators.{name}")
    settings_path = os.path.join(SETTINGS_DIR, f"{name}_settings.json")
    if not os.path.exists(settings_path):
        print(f"❌ Skipping {name} - settings not found.")
        return None

    with open(settings_path, "r") as f:
        settings = json.load(f)

    signal = module.final_signal(df.copy(), timeframe="1D")
    isp = df["manual_signal"].astype(int).values
    prices = df["close"]

    features = {
        "mae_vs_isp": float(np.mean(np.abs(signal - isp))),
        "correlation_vs_isp": float(np.corrcoef(signal, isp)[0, 1]),
        "signal_count": int(np.sum(np.abs(np.diff(signal)) > 0)),
        "sharpe_ratio": float(compute_sharpe(signal, prices)),
        "omega_ratio": float(compute_omega(signal, prices)),
        "transition_frequency": float(compute_transition_freq(signal)),
        "avg_holding_period": float(compute_holding_period(signal))
    }

    # Keep only behavioral-relevant settings
    for k in ["length", "adaptive", "volatilityPeriod"]:
        if k in settings:
            features[k] = settings[k]

    return name, features

def main():
    df = load_data()
    os.makedirs("features", exist_ok=True)

    all_features = {}
    for name in INDICATOR_NAMES:
        print(f"📊 Extracting features for {name}...")
        result = extract_features_for_indicator(name, df)
        if result:
            name, features = result
            all_features[name] = features

    with open(FEATURES_PATH, "w") as f:
        json.dump(all_features, f, indent=4)

    print(f"\n✅ All features saved to {FEATURES_PATH}")

if __name__ == "__main__":
    main()
