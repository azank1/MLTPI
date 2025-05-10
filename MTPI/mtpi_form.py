# mtpi_form.py

import json
import numpy as np
import math

# === Constants ===
BEHAVIOR_PATH = "features/strategy_behavior_aligned_isp.json"
OUTPUT_WEIGHTS_PATH = "features/strategy_weights.json"

def compute_strategy_score(strategy_dict):
    scores = [entry["score"] for entry in strategy_dict.values() if isinstance(entry, dict) and "score" in entry]
    if not scores:
        return 0.0, 0
    avg_score = np.mean(scores)
    count = len(scores)
    return avg_score, count

def balance_scores(strategy_scores_and_counts):
    weighted = []
    for score, count in strategy_scores_and_counts:
        quality = 1 / (1 + abs(score))
        weight = quality * math.log(1 + count)
        weighted.append(weight)

    weighted = np.array(weighted)
    norm = weighted / (np.sum(weighted) + 1e-8)
    return norm.tolist()

def main():
    with open(BEHAVIOR_PATH, "r") as f:
        behavior_data = json.load(f)

    strategies = list(behavior_data.keys())
    strategy_scores_and_counts = [
        compute_strategy_score(behavior_data[strategy])
        for strategy in strategies
    ]

    normalized_weights = balance_scores(strategy_scores_and_counts)

    final_weights = dict(zip(strategies, normalized_weights))

    print("\\n📊 Strategy EF Scores & Counts:")
    for strat, (score, count) in zip(strategies, strategy_scores_and_counts):
        print(f" - {strat}: Score = {score:.6f}, Indicators = {count}")

    print("\\n⚖️ Final Normalized Weights:")
    for strat, weight in final_weights.items():
        print(f" - {strat}: {weight:.4f}")

    with open(OUTPUT_WEIGHTS_PATH, "w") as f:
        json.dump(final_weights, f, indent=4)

if __name__ == "__main__":
    main()