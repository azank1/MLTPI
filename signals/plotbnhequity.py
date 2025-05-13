import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load BTC price data ===
df = pd.read_csv("CSVdata/target.csv")
df["DateTime"] = pd.to_datetime(df["time"], unit="s")
df.set_index("DateTime", inplace=True)
df["close"] = df["close"].astype(float)

# === Load MTPI signal ===
mtpi = pd.read_csv("signals/final_ISP.csv", index_col=0, parse_dates=True).squeeze()
df = df.join(mtpi.rename("MTPI"), how="left").dropna()

# === Slice from Jan 1, 2023 onward ===
df = df[df.index >= "2023-01-01"].copy()

# === Compute MTPI strategy equity ===
signal = np.where(df["MTPI"] > 0, 1, 0)
equity = [1.0]
for i in range(1, len(df)):
    if signal[i] == 1:
        equity.append(equity[-1] * (df["close"].iloc[i] / df["close"].iloc[i - 1]))
    else:
        equity.append(equity[-1])
df["MTPI_Equity"] = equity

# === Compute BTC buy & hold equity ===
df["BTC_Equity"] = df["close"] / df["close"].iloc[0]

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["BTC_Equity"], label="BTC Buy & Hold", alpha=0.5)
plt.plot(df.index, df["MTPI_Equity"], label="MTPI Strategy", color="purple")
plt.title("BTC Buy & Hold vs MTPI Strategy Equity (Since 2023)")
plt.ylabel("Equity Growth")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
