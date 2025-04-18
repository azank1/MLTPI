import os
import json
import importlib
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

DATA_PATH = "CSVdata/target.csv"
CLUSTERS_PATH = "features/strategy_clusters.json"
SETTINGS_DIR = "settings"
TIMEFRAME_OPTIONS = ["1D", "2D", "3D"]

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def backtest_equity(df, combined_signal):
    prices = df["close"].astype(float).values
    equity = [1.0]
    for i in range(1, len(prices)):
        if combined_signal[i] == 1:
            equity.append(equity[-1] * (prices[i] / prices[i-1]))
        else:
            equity.append(equity[-1])
    return np.array(equity)

def transition_penalty(signal):
    return np.sum(np.diff(signal) != 0) / len(signal)

def objective_builder(df, strategy_name, indicators):
    def objective_fn(**kwargs):
        signals = []

        idx = 0
        for ind in indicators:
            name = ind["name"]
            module = importlib.import_module(f"indicators.{name}")
            param_count = len(ind["settings_subset"])
            tf_idx = int(round(kwargs[f"{name}_tf"]))
            timeframe = TIMEFRAME_OPTIONS[tf_idx]

            # Reconstruct settings from search space
            tuned_settings = {}
            for j, key in enumerate(ind["settings_subset"].keys()):
                val = kwargs[f"{name}_p{j}"]
                tuned_settings[key] = int(round(val)) if isinstance(ind["settings_subset"][key], int) else float(val)

            # Write to settings file
            full_path = os.path.join(SETTINGS_DIR, f"{name}_settings.json")
            with open(full_path, "r") as f:
                all_settings = json.load(f)
            all_settings.update(tuned_settings)
            with open(full_path, "w") as f:
                json.dump(all_settings, f, indent=4)

            signal = module.final_signal(df.copy(), timeframe)
            signals.append(signal)

        avg_signal = np.mean(signals, axis=0)
        combined_signal = np.where(avg_signal > 0, 1, -1)

        # Evaluate strategy
        equity = backtest_equity(df, combined_signal)
        rets = np.diff(np.log(equity + 1e-8))
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
        omega = np.mean(rets[rets > 0]) / (abs(np.mean(rets[rets < 0])) + 1e-8)
        penalty = transition_penalty(combined_signal)

        return sharpe + omega - penalty * 0.5
    return objective_fn

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

def main():
    with open(CLUSTERS_PATH, "r") as f:
        strategy_map = json.load(f)

    df = load_data()

    for strategy_name, indicators in strategy_map.items():
        print(f"\n🚀 Optimizing strategy {strategy_name} with {len(indicators)} indicators...")
        optimize_strategy(strategy_name, indicators, df)

if __name__ == "__main__":
    main()
