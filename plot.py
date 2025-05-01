import pandas as pd
import matplotlib.pyplot as plt

file_path = "experiment_results/quadruped_v3_copy_full_history_rnn_h16_gen80_pop60.csv"
df_full = pd.read_csv(file_path)

# Shift x_dist and fitness values by 310 (-310 is starting point)
df_full["x_dist"] += 300
df_full["fitness"] += 300

# group by generation and get the best values
best_fitness = df_full.groupby("generation")["fitness"].max()
best_x_dist = df_full.groupby("generation")["x_dist"].max()
best_y_dist = df_full.groupby("generation")["y_dist"].max()

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# fitness plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[0].scatter([gen] * len(gen_data), gen_data["fitness"], color="gray", s=20, alpha=0.5)
axes[0].plot(best_fitness.index, best_fitness.values, marker='o', label='Best Fitness', linewidth=2)
axes[0].set_title("Fitness per Generation (All, Best, and Average)")
# Calculate the average fitness per generation
avg_fitness = df_full.groupby("generation")["fitness"].mean()
# Add the average fitness line to the plot
axes[0].plot(avg_fitness.index, avg_fitness.values, marker='x', label='Average Fitness', linestyle='--', linewidth=2)
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Fitness")
axes[0].legend()

# x dist plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[1].scatter([gen] * len(gen_data), gen_data["x_dist"], color="gray", s=20, alpha=0.5)
axes[1].plot(best_x_dist.index, best_x_dist.values, marker='o', label='Best X Distance', linewidth=2)
axes[1].set_title("X Distance per Generation (All, Best, and Average)")
# Calculate the average x distance per generation
avg_x_dist = df_full.groupby("generation")["x_dist"].mean()
# Add the average x distance line to the plot
axes[1].plot(avg_x_dist.index, avg_x_dist.values, marker='x', label='Average X Distance', linestyle='--', linewidth=2)
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("X Distance")
axes[1].legend()

# y dist plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[2].scatter([gen] * len(gen_data), gen_data["y_dist"], color="gray", s=20, alpha=0.5)
axes[2].set_title("Y Distance per Generation (All and Average)")
# axes[2].plot(best_y_dist.index, best_y_dist.values, marker='o', label='Best Y Distance', linewidth=2)
# Calculate the average y distance per generation
avg_y_dist = df_full.groupby("generation")["y_dist"].mean()
# Add the average y distance line to the plot
axes[2].plot(avg_y_dist.index, avg_y_dist.values, marker='x', label='Average Y Distance', linestyle='--', linewidth=2)
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Y Distance")
axes[2].legend()

plt.tight_layout()
plt.show()
