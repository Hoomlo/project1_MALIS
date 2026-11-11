import numpy as np
import matplotlib.pyplot as plt
from fishing import FishingBay
from agents import RandomAgent, play

N_SIMULATIONS = 500
raw_totals = []
performance_ratios = []
theoretical_efficiency = []

agent = RandomAgent(seed=42)

for s in range(N_SIMULATIONS):
    bay = FishingBay(season=1, seed=s)
    total = play(agent, bay)

    # Ground truth values revealed after the season ends
    info = bay.reveal()
    true_means = info["means"]
    opt_mean = max(true_means)

    # 1. Raw total catch (contains both bay richness variance and daily noise)
    raw_totals.append(total)

    # 2. Empirical ratio (normalizes bay richness, but retains daily gaussian noise)
    performance_ratios.append(total / bay.best_possible())

    # 3. Expected theoretical ratio (eliminates both bay richness and daily noise)
    chosen_spots = [spot for spot, _ in bay.history]
    theoretical_catch = sum(true_means[spot] for spot in chosen_spots)
    best_theoretical = opt_mean * bay.days
    theoretical_efficiency.append(theoretical_catch / best_theoretical)

# Compute coefficients of variation (std / mean)
cv_raw = np.std(raw_totals) / np.mean(raw_totals)
cv_ratio = np.std(performance_ratios) / np.mean(performance_ratios)
cv_theo = np.std(theoretical_efficiency) / np.mean(theoretical_efficiency)

print(f"1. Raw Catch (kg)               : Mean = {np.mean(raw_totals):.1f}, CV = {cv_raw:.2%}")
print(f"2. Empirical Performance Ratio   : Mean = {np.mean(performance_ratios):.3f}, CV = {cv_ratio:.2%}")
print(f"3. Theoretical Expected Ratio    : Mean = {np.mean(theoretical_efficiency):.3f}, CV = {cv_theo:.2%}")

# Generate comparison plots for Task 3
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Plot 1: Raw Total Catch
axes[0].hist(raw_totals, bins=25, color="lightsteelblue", edgecolor="black")
axes[0].set_title(f"1. Raw Total Catch (kg)\nHeavily Luck-Dependent (CV = {cv_raw:.1%})")
axes[0].set_xlabel("Total catch (kg)")
axes[0].set_ylabel("Frequency")

# Plot 2: Empirical Performance Ratio
axes[1].hist(performance_ratios, bins=25, color="sandybrown", edgecolor="black")
axes[1].set_title(f"2. Performance Ratio (Catch / Best)\nNormalizes Bay Richness (CV = {cv_ratio:.1%})")
axes[1].set_xlabel("Performance ratio")

# Plot 3: Theoretical Expected Ratio
axes[2].hist(theoretical_efficiency, bins=25, color="mediumseagreen", edgecolor="black")
axes[2].set_title(f"3. Expected Efficiency Ratio\nRemoves Daily Noise (CV = {cv_theo:.1%})")
axes[2].set_xlabel("Theoretical ratio")

plt.tight_layout()
output_filename = "task3_metrics_comparison.png"
plt.savefig(output_filename, dpi=300)
print(f"Plot saved successfully as '{output_filename}'")