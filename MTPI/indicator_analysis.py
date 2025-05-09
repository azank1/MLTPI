import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURES_PATH = "features/indicator_profiles.json"

def load_profiles(path=FEATURES_PATH):
    with open(path, "r") as f:
        return json.load(f)

def compute_risk_return_matrix(profiles):
    data = []
    for name, props in profiles.items():
        if all(k in props for k in ("sharpe_ratio", "omega_ratio", "transition_frequency", "mae_vs_isp")):
            sharpe = props["sharpe_ratio"]
            omega = props["omega_ratio"]
            trans = props["transition_frequency"]
            mae = props["mae_vs_isp"]
            risk = mae + 0.005 * trans
            expected_return = sharpe  # or combine with omega
            data.append((name, expected_return, risk, sharpe, omega, mae, trans))

    df = pd.DataFrame(data, columns=["name", "expected_return", "risk", "sharpe", "omega", "mae", "transitions"])
    return df

def plot_mpt_chart(df):
    plt.figure(figsize=(8, 6))
    for _, row in df.iterrows():
        plt.scatter(row["risk"], row["expected_return"])
        plt.text(row["risk"] + 0.0005, row["expected_return"], row["name"], fontsize=9)

    plt.title("MPT-style Indicator Scoring: Return vs Risk")
    plt.xlabel("Risk (MAE + Transitions)")
    plt.ylabel("Expected Return (Sharpe Ratio)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_pca_behavior_scores(df):
    features = df[["sharpe", "omega", "mae", "transitions"]]
    features_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=1)
    behavior_scores = pca.fit_transform(features_scaled).flatten()
    df["behavior_score"] = behavior_scores
    return df

def main():
    profiles = load_profiles()
    df = compute_risk_return_matrix(profiles)
    df = compute_pca_behavior_scores(df)
    plot_mpt_chart(df)
    df[["name", "expected_return", "risk", "behavior_score"]].to_csv("outputs/indicator_behavior_scores.csv", index=False)
    print("✅ Indicator behavior scores saved to outputs/indicator_behavior_scores.csv")

if __name__ == "__main__":
    main()
