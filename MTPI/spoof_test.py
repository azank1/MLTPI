import numpy as np
import json

def generate_spoofer(name, kind="laggy"):
    if kind == "laggy":
        return {
            "mae_vs_isp": np.random.uniform(0.04, 0.08),
            "correlation_vs_isp": np.random.uniform(0.6, 0.8),
            "sharpe_ratio": np.random.uniform(0.5, 1.0),
            "omega_ratio": np.random.uniform(1.0, 1.5),
            "transition_frequency": np.random.randint(2, 6),
            "avg_holding_period": np.random.randint(10, 15),
            "preferred_timeframe": np.random.choice(["1d", "2d"])
        }
    elif kind == "noisy":
        return {
            "mae_vs_isp": np.random.uniform(0.02, 0.04),
            "correlation_vs_isp": np.random.uniform(0.3, 0.6),
            "sharpe_ratio": np.random.uniform(1.2, 2.0),
            "omega_ratio": np.random.uniform(1.5, 2.5),
            "transition_frequency": np.random.randint(15, 30),
            "avg_holding_period": np.random.randint(1, 4),
            "preferred_timeframe": np.random.choice(["15m", "30m", "1h"])
        }

# Generate spoofed profiles
spoof_profiles = {
    f"laggy_{i}": generate_spoofer(f"laggy_{i}", "laggy") for i in range(5)
}
spoof_profiles.update({
    f"noisy_{i}": generate_spoofer(f"noisy_{i}", "noisy") for i in range(5)
})

# Save to file
with open("features/indicator_profiles_spoof.json", "w") as f:
    json.dump(spoof_profiles, f, indent=4)

print("✅ Spoofed indicators written to features/indicator_profiles.json")
