# mtpi_final.py

import os
import json
import numpy as np
import pandas as pd
import importlib

# === Constants ===
CSV_PATH = "CSVdata/target.csv"
BEHAVIOR_PATH = "features/strategy_behavior_aligned_isp.json"
WEIGHTS_PATH = "features/strategy_weights.json"
TIMEFRAME_OPTIONS = ["1D", "2D", "3D"]

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def apply_tf(df, tf):
    tf_map = {"1D": "1D", "2D": "2D", "3D": "3D"}
    return df.resample(tf_map[tf]).last().dropna()

def reconstruct_strategy_signal(df, indicators):
    signal_list = []
    for name, info in indicators.items():
        tf = info["tf"]
        settings = info["settings"]

        mod = importlib.import_module(f"indicators.{name}")
        tf_df = apply_tf(df.copy(), tf)
        signal = mod.final_signal(tf_df, tf, settings=settings)
        signal_series = pd.Series(signal, index=tf_df.index).dropna()
        signal_list.append(signal_series)

    if not signal_list:
        raise ValueError("No signals to aggregate.")

    aligned = pd.concat(signal_list, axis=1).dropna()
    averaged = aligned.mean(axis=1)
    return averaged

def main():
    df = load_data()

    with open(BEHAVIOR_PATH, "r") as f:
        behavior = json.load(f)

    with open(WEIGHTS_PATH, "r") as f:
        weights = json.load(f)

    final_isp = None

    for strat, indicators in behavior.items():
        strat_signal = reconstruct_strategy_signal(df.copy(), indicators)
        weighted_signal = strat_signal * weights.get(strat, 0)

        if final_isp is None:
            final_isp = weighted_signal
        else:
            final_isp = final_isp.add(weighted_signal, fill_value=0)

    unit_signal = final_isp / (np.max(np.abs(final_isp)) + 1e-8)
    os.makedirs("signals", exist_ok=True)
    np.save("signals/final_ISP.npy", unit_signal.values)
    unit_signal.to_csv("signals/final_ISP.csv", index=True)

    print("\\n✅ Final MTPI signal constructed.")
    print(f"📈 ISP Range: [{unit_signal.min():.4f}, {unit_signal.max():.4f}]")
    print(f"🧭 Last 10 values: {unit_signal.tail(10).tolist()}")

if __name__ == "__main__":
    main()