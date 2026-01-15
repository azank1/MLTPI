# Rewritten cluster_indicator.py with integrated Optuna-based cluster optimization

import os
import json
import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import optuna

FEATURES_PATH = "features/indicator_profiles.json"
CLUSTER_OUTPUT_PATH = "features/strategy_clusters.json"
SETTINGS_DIR = "settings"
DATA_PATH = "CSVdata/target.csv"

def load_indicator_features():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)

def extract_feature_matrix(profiles):
    names = []
    features = []

    for name, props in profiles.items():
        if "preferred_timeframe" not in props:
            continue

        fvec = [
            props.get("mae_vs_isp", 0),
            props.get("correlation_vs_isp", 0),
            props.get("sharpe_ratio", 0),
            props.get("omega_ratio", 0),
            props.get("transition_frequency", 0),
            props.get("avg_holding_period", 0)
        ]
        names.append(name)
        features.append(fvec)

    return names, np.array(features)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("DateTime", inplace=True)
    df["manual_signal"] = df["manual_signal"].ffill().fillna(0)
    return df

def select_dynamic_settings(indicator_name, top_n=2, perturb_pct=0.1):
    # Try importing indicator module
    try:
        module = importlib.import_module(f"indicators.{indicator_name}")
    except ModuleNotFoundError:
        print(f"⚠️ Module not found for '{indicator_name}'. Skipping setting optimization.")
        return {}

    # Load time series data
    df = load_data()

    # Load current settings
    settings_path = os.path.join(SETTINGS_DIR, f"{indicator_name}_settings.json")
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Settings file missing for '{indicator_name}'. Skipping.")
        return {}

    # Get base signal
    base_signal = module.final_signal(df.copy(), timeframe="1D")
    signal_base = np.array(base_signal)

    impact_scores = {}

    for param, value in settings.items():
        if isinstance(value, (int, float)):
            delta = value * perturb_pct if value != 0 else 1.0
            variants = [
                {**settings, param: value + delta},
                {**settings, param: value - delta}
            ]

            variances = []
            for var in variants:
                temp_path = os.path.join(SETTINGS_DIR, f"temp_{indicator_name}.json")
                with open(temp_path, "w") as tf:
                    json.dump(var, tf)
                os.replace(temp_path, settings_path)

                pert_signal = module.final_signal(df.copy(), timeframe="1D")
                variances.append(np.mean(np.abs(signal_base - np.array(pert_signal))))

            avg_impact = np.mean(variances)
            impact_scores[param] = avg_impact

    # Restore original settings
    with open(settings_path, "w") as f:
        json.dump(settings, f)

    top_params = sorted(impact_scores.items(), key=lambda x: -x[1])[:top_n]
    return {k: settings[k] for k, _ in top_params}


def optimize_n_clusters(X, n_trials=20, k_min=2, k_max=8):
    n_samples = X.shape[0]
    if n_samples <= 2:
        print("⚠️ Not enough indicators for clustering. Using 1 cluster.")
        return 1

    effective_max = max(min(n_samples - 1, k_max), k_min)

    def objective(trial):
        k = trial.suggest_int("n_clusters", k_min, effective_max)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        return silhouette_score(X, labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params["n_clusters"]

def build_clusters(names, labels, profiles):
    cluster_map = {}
    for idx, cluster_id in enumerate(labels):
        strat_key = f"S{cluster_id + 1}"
        if strat_key not in cluster_map:
            cluster_map[strat_key] = []

        name = names[idx]
        indicator_profile = profiles[name]
        cluster_map[strat_key].append({
            "name": name,
            "preferred_timeframe": indicator_profile.get("preferred_timeframe"),
            "settings_subset": select_dynamic_settings(name),
            "tempo_label": None
        })

    return cluster_map

def main():
    profiles = load_indicator_features()
    names, X = extract_feature_matrix(profiles)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)

    best_k = optimize_n_clusters(X_pca)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    strategy_clusters = build_clusters(names, labels, profiles)

    with open(CLUSTER_OUTPUT_PATH, "w") as f:
        json.dump(strategy_clusters, f, indent=4)

    print(f"✅ Strategy clusters saved to {CLUSTER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
