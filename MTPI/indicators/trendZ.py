import numpy as np
import pandas as pd
import json
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error

def final_signal(df, window=10, timeframe="1D"):
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

    df['ma'] = df['close'].rolling(window=window).mean()
    df['slope'] = df['ma'].diff()
    df['signal'] = 0
    df.loc[df['slope'] > 0, 'signal'] = 1
    df.loc[df['slope'] < 0, 'signal'] = -1
    return df['signal'].fillna(0)

def evaluate(window, df, manual_signal, timeframe):
    try:
        signal = final_signal(df.copy(), int(window), timeframe)
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
        def bo_wrapper(window):
            return evaluate(window, df, manual_signal, tf)

        optimizer = BayesianOptimization(
            f=bo_wrapper,
            pbounds={"window": (3, 30)},
            random_state=42,
            verbose=0
        )
        optimizer.maximize(init_points=3, n_iter=10)

        if optimizer.max['target'] > best_score:
            best_score = optimizer.max['target']
            best_params = optimizer.max['params']
            best_params['window'] = int(best_params['window'])
            best_tf = tf

    best_params["preferred_timeframe"] = best_tf
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=4)