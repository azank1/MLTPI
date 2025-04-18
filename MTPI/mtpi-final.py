import os
import json
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Paths ===
DATA_PATH = "CSVdata/target.csv"
CLUSTERS_PATH = "features/strategy_clusters.json"
OUTPUT_DIR = "outputs"

# === Load target data ===
df = pd.read_csv(DATA_PATH)
df["DateTime"] = pd.to_datetime(df["time"], unit="s")
df.set_index("DateTime", inplace=True)
close_price = df["close"]

# === Load strategy definitions ===
with open(CLUSTERS_PATH, "r") as f:
    all_strategies = json.load(f)

# === Aggregate all strategy indicator outputs ===
strategy_signals = []
for strategy_name, indicators in all_strategies.items():
    strategy_output = []
    print(f"\n🚀 Strategy: {strategy_name}")
    for ind in indicators:
        try:
            name = ind["name"]
            timeframe = ind.get("preferred_timeframe", "1D")
            print(f"  ↳ {name} @ {timeframe}")
            module = importlib.import_module(f"indicators.{name}")
            signal = module.final_signal(df.copy(), timeframe)
            strategy_output.append(signal)
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    if strategy_output:
        strat_signal = np.mean(strategy_output, axis=0)
        strategy_signals.append(strat_signal)

# === Final MTPI Signal with threshold-based transitions ===
if strategy_signals:
    mtpi_score = np.mean(strategy_signals, axis=0)
    mtpi_signal = np.zeros_like(mtpi_score)

    # State logic: 1 = long, -1 = exit to cash
    state = 0
    for i in range(len(mtpi_score)):
        if mtpi_score[i] > 0.1 and state != 1:
            mtpi_signal[i] = 1
            state = 1
        elif mtpi_score[i] < -0.1 and state != -1:
            mtpi_signal[i] = -1
            state = -1
        else:
            mtpi_signal[i] = state

    # === Correct Equity Calculation ===
    equity = [1.0]
    returns = close_price.pct_change().fillna(0).values
    for i in range(1, len(mtpi_signal)):
        if mtpi_signal[i - 1] == 1:
            equity.append(equity[-1] * (1 + returns[i]))
        else:
            equity.append(equity[-1])  # stay flat (cash)

    equity = np.array(equity)

    # === Output DataFrame ===
    df_out = pd.DataFrame(index=df.index)
    df_out["close"] = close_price
    df_out["MTPI_Score"] = mtpi_score
    df_out["MTPI_Signal"] = mtpi_signal
    df_out["Equity"] = equity
    df_out["signal_change"] = df_out["MTPI_Signal"].ne(df_out["MTPI_Signal"].shift(1))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(os.path.join(OUTPUT_DIR, "MTPI_signal.csv"))

    # === Plotting: Price + Signal Markers + Equity ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- Price + signal markers
    ax1.plot(df_out.index, df_out["close"], label="Price", color="black", linewidth=1.5)
    for i in range(1, len(df_out)):
        if df_out["signal_change"].iloc[i]:
            dt = df_out.index[i]
            sig = df_out["MTPI_Signal"].iloc[i]
            if sig == 1:
                ax1.axvline(x=dt, color="green", linestyle="--", alpha=0.6,
                            label="Enter" if "Enter" not in ax1.get_legend_handles_labels()[1] else "")
            elif sig == -1:
                ax1.axvline(x=dt, color="red", linestyle="--", alpha=0.6,
                            label="Exit" if "Exit" not in ax1.get_legend_handles_labels()[1] else "")
    ax1.set_ylabel("Price")
    ax1.set_title("MTPI Signal on Price")
    ax1.legend()
    ax1.grid(True)

    # --- Equity curve
    ax2.plot(df_out.index, df_out["Equity"], label="Equity", color="orange", linewidth=2)
    ax2.set_ylabel("Equity")
    ax2.set_title("Backtested Equity Curve (MTPI)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "MTPI_equity_plot.png"))
    plt.show()

else:
    print("❌ No valid signals generated from strategies.")
