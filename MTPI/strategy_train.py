
import os
import json
import importlib
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# === Constants ===
DATA_PATH = "CSVdata/target.csv"
CLUSTERS_PATH = "features/strategy_clusters.json"
SETTINGS_DIR = "settings"
TIMEFRAME_OPTIONS = ["1D", "2D", "3D"]

# === Load Data ===
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

# === Reward Function for Medium-Term Learning ===
def compute_reward(signal, df, penalty_weight=0.5, target_hold_range=(5, 15)):
    signal = pd.Series(signal, index=df.index).fillna(0).astype(int)

    prices = df["close"].astype(float).values
    equity = [1.0]
    for i in range(1, len(prices)):
        if signal[i] == 1:
            equity.append(equity[-1] * (prices[i] / prices[i-1]))
        else:
            equity.append(equity[-1])
    equity = np.array(equity)

    rets = np.diff(np.log(equity + 1e-8))
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
    omega = np.mean(rets[rets > 0]) / (abs(np.mean(rets[rets < 0])) + 1e-8)

    transitions = np.sum(np.diff(signal) != 0)
    flip_rate = transitions / (len(signal) - 1)

    state = (signal != signal.shift(1)).astype(int)
    hold_lengths = state[state == 1].index.to_series().diff().dt.days.dropna()
    avg_hold = hold_lengths.mean() if not hold_lengths.empty else 0

    target_low, target_high = target_hold_range
    if avg_hold is None or np.isnan(avg_hold):
        hold_score = 0
    elif avg_hold < target_low:
        hold_score = - (target_low - avg_hold) / target_low
    elif avg_hold > target_high:
        hold_score = - (avg_hold - target_high) / target_high
    else:
        hold_score = 1 - abs(avg_hold - np.mean(target_hold_range)) / (target_high - target_low)

    reward = (sharpe * omega) / (1 + flip_rate + 1e-6)
    total_score = reward + hold_score - penalty_weight * flip_rate
    return total_score

# === Strategy Objective Function Builder ===
def objective_builder(df, strategy_name, indicators):
    def objective_fn(**kwargs):
        signals = []

        for ind in indicators:
            name = ind["name"]
            module = importlib.import_module(f"indicators.{name}")
            param_count = len(ind["settings_subset"])
            tf_idx = int(round(kwargs[f"{name}_tf"]))
            timeframe = TIMEFRAME_OPTIONS[tf_idx]

            tuned_settings = {}
            for j, key in enumerate(ind["settings_subset"].keys()):
                val = kwargs[f"{name}_p{j}"]
                tuned_settings[key] = int(round(val)) if isinstance(ind["settings_subset"][key], int) else float(val)

            signal = module.final_signal(df.copy(), timeframe, settings=tuned_settings)
            signals.append(signal)

        avg_signal = np.mean(signals, axis=0)
        combined_signal = np.where(avg_signal > 0, 1, -1)
        return compute_reward(combined_signal, df)

    return objective_fn

# === Strategy Optimizer ===
def optimize_strategy(name, indicators, df):
    pbounds = {}
    for ind in indicators:
        iname = ind["name"]
        pbounds[f"{iname}_tf"] = (0, len(TIMEFRAME_OPTIONS) - 1)
        for j, (param, val) in enumerate(ind["settings_subset"].items()):
            key = f"{iname}_p{j}"
            if isinstance(val, int):
                low = max(1, int(val * 0.5))
                high = int(val * 2) if val > 1 else int(val + 3)
            elif isinstance(val, float):
                span = abs(val) if val != 0 else 1.0
                low = val - span * 0.5
                high = val + span * 0.5
            if low > high:
                low, high = high, low
            pbounds[key] = (low, high)

    optimizer = BayesianOptimization(
        f=objective_builder(df, name, indicators),
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=25)

    print(f"\n✅ Optimized strategy: {name}")
    print(f"📈 Best score: {optimizer.max['target']:.4f}")
    return optimizer.max

# === Main Runner ===
def main():
    with open(CLUSTERS_PATH, "r") as f:
        strategy_map = json.load(f)

    df = load_data()
    result = {}

    for strategy_name, indicators in strategy_map.items():
        print(f"\n🚀 Optimizing strategy {strategy_name} with {len(indicators)} indicators...")
        opt_result = optimize_strategy(strategy_name, indicators, df)
        result[strategy_name] = {}

        for ind in indicators:
            name = ind["name"]
            tf_idx = int(round(opt_result[f"{name}_tf"]))
            tf = TIMEFRAME_OPTIONS[tf_idx]
            result[strategy_name][name] = {
                "tf": tf,
                "settings": {}
            }
            for j, key in enumerate(ind["settings_subset"].keys()):
                param = f"{name}_p{j}"
                val = opt_result[param]
                result[strategy_name][name]["settings"][key] = (
                    int(round(val)) if isinstance(ind["settings_subset"][key], int) else float(val)
                )

    os.makedirs("features", exist_ok=True)
    with open("features/strategy_behavior.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\n📁 Saved strategy_behavior.json")

if __name__ == "__main__":
    main()
