import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load MTPI signal and BTC Price ===
mtpi = pd.read_csv("signals/final_ISP.csv", index_col=0, parse_dates=True).squeeze()
price = pd.read_csv("CSVdata/target.csv")
price["DateTime"] = pd.to_datetime(price["time"], unit="s")
price.set_index("DateTime", inplace=True)
btc_price = price["close"].astype(float)

# === Combine Data ===
df = pd.concat([btc_price, mtpi.rename("MTPI")], axis=1).dropna()
df.rename(columns={"close": "BTC"}, inplace=True)

# === Derive Equity from MTPI signal ===
signal = np.where(df["MTPI"] > 0, 1, 0)
df["Signal"] = signal
equity = [1.0]
for i in range(1, len(df)):
    if signal[i] == 1:
        equity.append(equity[-1] * (df["BTC"].iloc[i] / df["BTC"].iloc[i - 1]))
    else:
        equity.append(equity[-1])
df["Equity"] = equity

# === Entry/Exit detection from signal transitions ===
transitions = df["Signal"].diff()
entries = df[transitions == 1].index
exits = df[transitions == -1].index

# === Plotting ===
plt.figure(figsize=(14, 8))

# Price and transitions
plt.subplot(2, 1, 1)
plt.plot(df.index, df["BTC"], label="BTC Price", alpha=0.6)
for t in entries:
    plt.axvline(t, color="green", linestyle="--", alpha=0.6)
for t in exits:
    plt.axvline(t, color="red", linestyle="--", alpha=0.6)
plt.plot(df.index, df["MTPI"], label="Emergent MTPI Signal", alpha=0.8)
plt.title("BTC Price & MTPI Transitions Inferred from Equity Allocation")
plt.legend()
plt.grid()

# Equity curve
plt.subplot(2, 1, 2)
plt.plot(df.index, df["Equity"], label="Equity from MTPI Signal", color="purple")
plt.title("Equity Curve Based on MTPI Allocation Transitions")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
