# PATCHED: profile_isp_and_align_signals.py — keeps `score` field from strategy_behavior.json

import json
import numpy as np
import pandas as pd
import importlib
from scipy.stats import entropy

CSV_PATH = "CSVdata/target.csv"
BEHAVIOR_PATH = "features/strategy_behavior.json"
OUTPUT_PATH = "features/strategy_behavior_aligned_isp.json"
TIMEFRAMES = ["1D", "2D", "3D"]

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def apply_tf(df, tf):
    return df.resample(tf).last().dropna()

def extract_signal_features(signal):
    signal = pd.Series(signal).fillna(0).astype(int)
    transitions = np.sum(np.diff(signal) != 0)
    flip_rate = transitions / len(signal)
    signal_std = np.std(signal)

    durations = []
    count = 1
    for i in range(1, len(signal)):
        if signal[i] == signal[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    if count > 0:
        durations.append(count)
    avg_hold = np.mean(durations)
    ent = entropy(pd.Series(signal).value_counts(normalize=True))

    return {
        "flip_rate": flip_rate,
        "std": signal_std,
        "avg_hold": avg_hold,
        "entropy": ent
    }

def compute_distance(ind_feat, isp_feat):
    keys = isp_feat.keys()
    return np.mean([abs(ind_feat[k] - isp_feat[k]) for k in keys])

def main():
    df = load_data()
    isp_signal = df["manual_signal"]
    isp_features = extract_signal_features(isp_signal)

    with open(BEHAVIOR_PATH, "r") as f:
        behavior = json.load(f)

    aligned_behavior = {}

    for strat, indicators in behavior.items():
        aligned_behavior[strat] = {}

        for name, info in indicators.items():
            module = importlib.import_module(f"indicators.{name}")
            settings = info["settings"]
            orig_score = info.get("score", 1.0)
            best_score = float("inf")
            best_tf = info["tf"]

            for tf in TIMEFRAMES:
                tf_df = apply_tf(df.copy(), tf)
                signal = module.final_signal(tf_df.copy(), tf, settings=settings)
                feat = extract_signal_features(signal)
                dist = compute_distance(feat, isp_features)

                if dist < best_score:
                    best_score = dist
                    best_tf = tf

            aligned_behavior[strat][name] = {
                "tf": best_tf,
                "settings": settings,
                "score": orig_score  # Keep score for mtpi_form weighting
            }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(aligned_behavior, f, indent=4)

    print("\\n✅ ISP-profile-aligned strategy saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
