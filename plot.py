import pandas as pd
import matplotlib.pyplot as plt

file_path = "vsr_models/quadruped_v3_copy/quadruped_v3_copy_full_history_mlp_plus_h24_16_gen4_pop8.csv"
df_full = pd.read_csv(file_path)

# group by generation and get the best values
best_fitness = df_full.groupby("generation")["fitness"].max()
best_x_dist = df_full.groupby("generation")["x_dist"].max()
best_y_dist = df_full.groupby("generation")["y_dist"].max()

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# fitness plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[0].scatter([gen] * len(gen_data), gen_data["fitness"], color="gray", s=20)
axes[0].plot(best_fitness.index, best_fitness.values, marker='o', label='Best Fitness', linewidth=2)
axes[0].set_title("Fitness per Generation (All and Best)")
axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Fitness")
axes[0].legend()

# x dist plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[1].scatter([gen] * len(gen_data), gen_data["x_dist"], color="gray", s=20)
axes[1].plot(best_x_dist.index, best_x_dist.values, marker='o', label='Best X Distance', linewidth=2)
axes[1].set_title("X Distance per Generation (All and Best)")
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("X Distance")
axes[1].legend()

# y dist plot
for gen in df_full["generation"].unique():
    gen_data = df_full[df_full["generation"] == gen]
    axes[2].scatter([gen] * len(gen_data), gen_data["y_dist"], color="gray", s=20)
axes[2].plot(best_y_dist.index, best_y_dist.values, marker='o', label='Best Y Distance', linewidth=2)
axes[2].set_title("Y Distance per Generation (All and Best)")
axes[2].set_xlabel("Generation")
axes[2].set_ylabel("Y Distance")
axes[2].legend()

plt.tight_layout()
plt.show()
