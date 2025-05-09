import numpy as np
import pandas as pd
import json
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error

def alma(series, window, offset, sigma):
    m = offset * (window - 1)
    s = window / sigma
    weights = np.exp(-((np.arange(window) - m)**2) / (2 * s * s))
    weights /= weights.sum()
    return np.convolve(series, weights, mode='same')

def final_signal(df, alma_window=14, offset=0.85, sigma=6, z_threshold=1.5, timeframe="1D"):
    if timeframe != "1D":
        df = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'manual_signal': 'last'
        }).dropna()
    else:
        df = df.copy()

    price = df["close"]
    alma_series = pd.Series(alma(price.values, int(alma_window), offset, sigma), index=df.index)
    z = (price - alma_series) / (price.rolling(int(alma_window)).std() + 1e-9)

    df["signal"] = 0
    df.loc[z > z_threshold, "signal"] = 1
    df.loc[z < -z_threshold, "signal"] = -1
    return df["signal"].fillna(0)

def evaluate(alma_window, offset, sigma, z_threshold, df, manual_signal, timeframe):
    try:
        signal = final_signal(df.copy(),
                              alma_window=int(alma_window),
                              offset=offset,
                              sigma=sigma,
                              z_threshold=z_threshold,
                              timeframe=timeframe)
        score = -mean_absolute_error(manual_signal.reindex(signal.index).fillna(0), signal)
        return score
    except Exception:
        return -1e6

def train_indicator(df, output_path):
    manual_signal = df["manual_signal"]
    best_score = -1e6
    best_params = {}
    best_tf = "1D"

    for tf in ["1D", "2D", "3D"]:
        def bo_wrapper(alma_window, offset, sigma, z_threshold):
            return evaluate(alma_window, offset, sigma, z_threshold, df, manual_signal, tf)

        optimizer = BayesianOptimization(
            f=bo_wrapper,
            pbounds={
                "alma_window": (5, 40),
                "offset": (0.1, 1.0),
                "sigma": (2, 10),
                "z_threshold": (0.5, 3.0)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=3, n_iter=10)

        if optimizer.max["target"] > best_score:
            best_score = optimizer.max["target"]
            best_params = optimizer.max["params"]
            best_params["alma_window"] = int(best_params["alma_window"])
            best_tf = tf

    best_params["preferred_timeframe"] = best_tf
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=4)