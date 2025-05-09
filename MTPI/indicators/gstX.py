import numpy as np
import pandas as pd
import json
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error

def gaussian_smooth(series, length):
    weights = np.exp(-0.5 * (np.arange(length) - length // 2)**2)
    weights /= weights.sum()
    return np.convolve(series, weights, mode='same')

def final_signal(df, gauss_length=14, dema_length=20, threshold=1.5, timeframe="1D"):
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
    ema1 = price.ewm(span=dema_length, adjust=False).mean()
    ema2 = ema1.ewm(span=dema_length, adjust=False).mean()
    dema = 2 * ema1 - ema2

    gauss = pd.Series(gaussian_smooth(dema.values, int(gauss_length)), index=df.index)

    delta = gauss.diff()
    std_dev = gauss.rolling(window=int(gauss_length)).std()
    zscore = delta / (std_dev + 1e-9)

    df["signal"] = 0
    df.loc[zscore > threshold, "signal"] = 1
    df.loc[zscore < -threshold, "signal"] = -1
    return df["signal"].fillna(0)

def evaluate(gauss_length, dema_length, threshold, df, manual_signal, timeframe):
    try:
        signal = final_signal(df.copy(),
                              gauss_length=int(gauss_length),
                              dema_length=int(dema_length),
                              threshold=threshold,
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
        def bo_wrapper(gauss_length, dema_length, threshold):
            return evaluate(gauss_length, dema_length, threshold, df, manual_signal, tf)

        optimizer = BayesianOptimization(
            f=bo_wrapper,
            pbounds={
                "gauss_length": (5, 40),
                "dema_length": (5, 40),
                "threshold": (0.5, 3.0)
            },
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=3, n_iter=10)

        if optimizer.max["target"] > best_score:
            best_score = optimizer.max["target"]
            best_params = optimizer.max["params"]
            best_params["gauss_length"] = int(best_params["gauss_length"])
            best_params["dema_length"] = int(best_params["dema_length"])
            best_tf = tf

    best_params["preferred_timeframe"] = best_tf
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=4)