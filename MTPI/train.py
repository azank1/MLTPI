import os
import importlib
import pandas as pd

INDICATOR_NAMES = ["agma", "qtrend", "trendZ", "momentumX", "gstX", "zscoreMA"]  # Just add names here to include more indicators
SETTINGS_DIR = "settings"
DATA_PATH = "CSVdata/target.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def train_all_indicators():
    df = load_data()

    for name in INDICATOR_NAMES:
        print(f"🔧 Training {name}...")
        module = importlib.import_module(f"indicators.{name}")
        output_path = os.path.join(SETTINGS_DIR, f"{name}_settings.json")
        module.train_indicator(df.copy(), output_path)
        print(f"✅ {name} settings saved to {output_path}")

if __name__ == "__main__":
    train_all_indicators()
