import os
import json
import importlib
import numpy as np
import pandas as pd

DATA_PATH = "CSVdata/target.csv"
FEATURES_PATH = "features/indicator_profiles.json"
INDICATOR_NAMES = ["agma", "qtrend"]  # Add more indicators when ready
TIMEFRAMES = ["1D", "2D", "3D"]

# === Load Data ===
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

# === Scoring Metrics ===
def compute_sharpe(signal, prices):
    rets = prices.pct_change().fillna(0)
    active_rets = signal[:-1] * rets[1:]
    return np.mean(active_rets) / (np.std(active_rets) + 1e-8)

def compute_omega(signal, prices):
    rets = prices.pct_change().fillna(0)
    active = signal[:-1] * rets[1:]
    gain = np.mean(active[active > 0]) if np.any(active > 0) else 0
    loss = np.abs(np.mean(active[active < 0])) if np.any(active < 0) else 1e-8
    return gain / loss

def compute_transition_freq(signal):
    return np.sum(np.diff(signal) != 0) / (len(signal) - 1)

def compute_holding_period(signal):
    transitions = np.where(np.diff(signal) != 0)[0]
    if len(transitions) < 2:
        return len(signal)
    return float(np.mean(np.diff(transitions)))

# === Scoring Function (Crypto Efficient Frontier) ===
def compute_score(sharpe, omega, mae, correlation):
    return (
        -((sharpe - 2.1) ** 2)
        -((omega - 7) ** 2)
        - mae
        + 0.5 * correlation
    )

# === Evaluate One Indicator ===
def scale_timeframe_for_indicator(name, df, existing_features):
    module = importlib.import_module(f"indicators.{name}")
    best_score = -np.inf
    best_result = None

    for tf in TIMEFRAMES:
        signal = module.final_signal(df.copy(), timeframe=tf)
        isp = df["manual_signal"].astype(int).values
        prices = df["close"]

        sharpe = compute_sharpe(signal, prices)
        omega = compute_omega(signal, prices)
        mae = np.mean(np.abs(signal - isp))
        corr = np.corrcoef(signal, isp)[0, 1]
        transitions = compute_transition_freq(signal)
        holding = compute_holding_period(signal)
        score = compute_score(sharpe, omega, mae, corr)

        if score > best_score:
            best_score = score
            best_result = {
                "preferred_timeframe": tf,
                "score": round(score, 4),
                "sharpe_ratio": round(sharpe, 4),
                "omega_ratio": round(omega, 4),
                "mae_vs_isp": round(mae, 4),
                "correlation_vs_isp": round(corr, 4),
                "transition_frequency": round(transitions, 4),
                "avg_holding_period": round(holding, 2)
            }

    if best_result:
        existing_features[name].update(best_result)
        print(f"✅ {name} scaled to {best_result['preferred_timeframe']}")

# === Main Runner ===
def main():
    df = load_data()

    with open(FEATURES_PATH, "r") as f:
        all_features = json.load(f)

    for name in INDICATOR_NAMES:
        if name not in all_features:
            print(f"⚠️ Skipping {name} — no features found.")
            continue
        scale_timeframe_for_indicator(name, df, all_features)

    with open(FEATURES_PATH, "w") as f:
        json.dump(all_features, f, indent=4)

    print("\n✅ Timeframe scaling complete and saved.")

if __name__ == "__main__":
    main()
