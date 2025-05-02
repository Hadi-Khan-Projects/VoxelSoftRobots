import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D # for 3d plotting

# global configuration
BASE_RESULTS_PATH = 'results_experiments' # adjust if needed
OUTPUT_DIR = 'results_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# alpha values for scatter plots
ALPHA_OPTIMISE_INDIVIDUAL = 0.3
ALPHA_EVOLVE_INDIVIDUAL_OVERALL = 0.15 # for per-method plots showing all points
ALPHA_EVOLVE_INDIVIDUAL_COMBINED = 0.1 # for combined plots showing all points (can be denser)
ALPHA_EVOLVE_TOP_N_INDIVIDUAL = 0.15

# colours
METHOD_COLORS = {
    'mlp': 'red',
    'mlp_2': 'orange', # for the second mlp evolution run
    'mlp_plus': 'green',
    'rnn': 'blue',
}
DEFAULT_COLOR = 'gray'

# morphology plotting config
N_MORPH_COLS = 6
N_MORPH_ROWS_PER_PLOT_GROUP = 4 # number of groups of 3 morphologies per column

# placeholder voxelrobot class
# important: replace this with an import of your actual voxelrobot class
# or copy the necessary methods from your vsr.py file.
# this placeholder is minimal and assumes voxel_grid is set after parsing.
class VoxelRobot:
    """
    a minimal representation of a voxel robot for plotting purposes.
    """
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, gear: float = 0.0) -> None:
        """
        initialise the voxelrobot with dimensions.

        args:
            x_dim (int): the x dimension of the voxel grid.
            y_dim (int): the y dimension of the voxel grid.
            z_dim (int): the z dimension of the voxel grid.
            gear (float, optional): gear ratio, not used in this placeholder's plotting. defaults to 0.0.

        returns:
            none
        """
        self.max_x = x_dim
        self.max_y = y_dim
        self.max_z = z_dim
        self.voxel_grid = np.zeros((x_dim, y_dim, z_dim), dtype=np.uint8)
        # other attributes like gear might be needed if your visualise_model uses them
        # for plotting purposes, only the grid and dimensions are strictly necessary.

    def set_val(self, x: int, y: int, z: int, value: int) -> None:
        """
        set the value of a voxel in the grid.

        args:
            x (int): x-coordinate of the voxel.
            y (int): y-coordinate of the voxel.
            z (int): z-coordinate of the voxel.
            value (int): value to set (typically 0 or 1).

        returns:
            none
        """
        if 0 <= x < self.max_x and 0 <= y < self.max_y and 0 <= z < self.max_z:
            self.voxel_grid[x, y, z] = value
        else:
            print(f"warning: voxel out of bounds ({x},{y},{z}) for grid {self.max_x}x{self.max_y}x{self.max_z}")


    def plot_on_ax(self, ax: Axes3D) -> None:
        """
        visualise the vsr's structure on a given matplotlib axes object.

        args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): the 3d axes to plot on.

        returns:
            none
        """
        ax.clear() # clear previous content if any
        x_coords, y_coords, z_coords = np.where(self.voxel_grid == 1)
        voxel_plot_size = 0.9

        if len(x_coords) > 0: # only plot if there are active voxels
            for xi, yi, zi in zip(x_coords, y_coords, z_coords):
                ax.bar3d(
                    xi - voxel_plot_size / 2,
                    yi - voxel_plot_size / 2,
                    zi - voxel_plot_size / 2,
                    voxel_plot_size,
                    voxel_plot_size,
                    voxel_plot_size,
                    color="red", # or derive from method colour if needed
                    alpha=0.6,
                )

        # determine dynamic limits based on max_x, max_y, max_z from the vsr instance itself
        ax.set_xlim(0, self.max_x)
        ax.set_ylim(0, self.max_y)
        ax.set_zlim(0, self.max_z)

        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.set_xticks(np.arange(0, self.max_x + 1, max(1, self.max_x // 2))) # dynamic ticks
        ax.set_yticks(np.arange(0, self.max_y + 1, max(1, self.max_y // 2)))
        ax.set_zticks(np.arange(0, self.max_z + 1, max(1, self.max_z // 2)))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.view_init(elev=20., azim=-60) # consistent view angle

# helper functions
def parse_voxel_coords(coord_str: str) -> list:
    """
    parses a string representation of voxel coordinates into a list of tuples.

    args:
        coord_str (str): the string representation of coordinates (e.g., "[(0,0,0), (1,0,0)]").

    returns:
        list: a list of (x,y,z) coordinate tuples, or an empty list if parsing fails or input is nan.
    """
    try:
        if pd.isna(coord_str):
            return []
        coords = ast.literal_eval(coord_str)
        if isinstance(coords, list):
            return coords
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

def count_voxels(coord_str: str) -> int:
    """
    counts the number of voxels from a string representation of coordinates.

    args:
        coord_str (str): the string representation of coordinates.

    returns:
        int: the number of voxels.
    """
    return len(parse_voxel_coords(coord_str))

def get_method_name_from_path(path_part: str) -> str:
    """
    extracts a standardised method name from a file path segment.

    args:
        path_part (str): a segment of a file path.

    returns:
        str: the standardised method name (e.g., 'mlp_plus', 'mlp', 'rnn') or 'unknown'.
    """
    if "mlp_plus" in path_part: # check for mlp_plus first
        return "mlp_plus"
    elif "mlp_2" in path_part:
        return "mlp_2"
    elif "mlp" in path_part:
        return "mlp"
    elif "rnn" in path_part:
        return "rnn"
    return "unknown"

# plotting functions for optimise.py logs

def plot_optimise_fitness_per_method(df: pd.DataFrame, method_name: str, color: str, output_prefix: str) -> None:
    """
    plots fitness vs. generation for a single optimisation method.
    includes individual fitness points, average fitness line, and best fitness line.

    args:
        df (pd.dataframe): dataframe containing optimisation log data for the method.
        method_name (str): name of the optimisation method (e.g., 'mlp').
        color (str): colour to use for plotting this method.
        output_prefix (str): prefix for the output plot filename.

    returns:
        none
    """
    plt.figure(figsize=(12, 7))
    generations = df['generation'].unique()
    all_fitness = []
    avg_fitness_per_gen = []
    best_fitness_per_gen = []

    for gen in sorted(generations):
        gen_df = df[df['generation'] == gen]
        fitness_values = gen_df['fitness'].dropna()
        if not fitness_values.empty:
            plt.scatter([gen] * len(fitness_values), fitness_values,
                        color=color, alpha=ALPHA_OPTIMISE_INDIVIDUAL, s=10,  edgecolor='none')
            all_fitness.extend(fitness_values)
            avg_fitness_per_gen.append(fitness_values.mean())
            best_fitness_per_gen.append(fitness_values.max())
        else:
            avg_fitness_per_gen.append(np.nan)
            best_fitness_per_gen.append(np.nan)

    gens_for_lines = sorted(generations)
    plt.plot(gens_for_lines, avg_fitness_per_gen, linestyle='--', color=color, label=f'{method_name} avg fitness')
    plt.plot(gens_for_lines, best_fitness_per_gen, linestyle='-', color=color, label=f'{method_name} best fitness')

    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title(f'optimisation fitness vs. generation for {method_name.upper()}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_fitness_vs_gen_{method_name}.png'))
    plt.close()

def plot_optimise_ydist_per_method(df: pd.DataFrame, method_name: str, color: str, output_prefix: str) -> None:
    """
    plots y-distance to target vs. generation for a single optimisation method.
    includes individual y-distance points and average y-distance line.

    args:
        df (pd.dataframe): dataframe containing optimisation log data for the method.
        method_name (str): name of the optimisation method.
        color (str): colour to use for plotting this method.
        output_prefix (str): prefix for the output plot filename.

    returns:
        none
    """
    plt.figure(figsize=(12, 7))
    generations = df['generation'].unique()
    avg_ydist_per_gen = []

    for gen in sorted(generations):
        gen_df = df[df['generation'] == gen]
        # filter out large/infinite y_dist values that skew plots
        ydist_values = gen_df['y_dist'].replace([np.inf, -np.inf], np.nan).dropna()
        ydist_values = ydist_values[ydist_values.abs() < 1000] # heuristic clip for plotting

        if not ydist_values.empty:
            plt.scatter([gen] * len(ydist_values), ydist_values,
                        color=color, alpha=ALPHA_OPTIMISE_INDIVIDUAL, s=10, edgecolor='none')
            avg_ydist_per_gen.append(ydist_values.mean())
        else:
            avg_ydist_per_gen.append(np.nan)

    gens_for_lines = sorted(generations)
    plt.plot(gens_for_lines, avg_ydist_per_gen, linestyle='--', color=color, label=f'{method_name} avg y-distance')

    plt.xlabel('generation')
    plt.ylabel('y-distance to target')
    plt.title(f'optimisation y-distance vs. generation for {method_name.upper()}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_ydist_vs_gen_{method_name}.png'))
    plt.close()

def plot_optimise_fitness_all_methods(dfs_map: dict, output_prefix: str) -> None:
    """
    plots fitness vs. generation for all optimisation methods on a single graph.

    args:
        dfs_map (dict): a dictionary mapping method names (str) to their dataframes (pd.dataframe).
        output_prefix (str): prefix for the output plot filename.

    returns:
        none
    """
    plt.figure(figsize=(14, 8))
    for method_name, df in dfs_map.items():
        color = METHOD_COLORS.get(method_name, DEFAULT_COLOR)
        generations = df['generation'].unique()
        avg_fitness_per_gen = []
        best_fitness_per_gen = []

        for gen in sorted(generations):
            gen_df = df[df['generation'] == gen]
            fitness_values = gen_df['fitness'].dropna()
            if not fitness_values.empty:
                plt.scatter([gen] * len(fitness_values), fitness_values,
                            color=color, alpha=ALPHA_OPTIMISE_INDIVIDUAL, s=10, edgecolor='none',
                            label=f'_{method_name} individuals' if gen == sorted(generations)[0] else None) # label once
                avg_fitness_per_gen.append(fitness_values.mean())
                best_fitness_per_gen.append(fitness_values.max())
            else:
                avg_fitness_per_gen.append(np.nan)
                best_fitness_per_gen.append(np.nan)

        gens_for_lines = sorted(generations)
        plt.plot(gens_for_lines, avg_fitness_per_gen, linestyle='--', color=color, label=f'{method_name} avg fitness')
        plt.plot(gens_for_lines, best_fitness_per_gen, linestyle='-', color=color, label=f'{method_name} best fitness')

    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('optimisation fitness vs. generation (all methods)')
    # create a clean legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # remove duplicate labels for individuals
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_fitness_vs_gen_all_methods.png'))
    plt.close()


# plotting functions for evolve.py logs

def load_evolution_run_data(evolution_run_path: str) -> pd.DataFrame:
    """
    loads all batch_x_history.csv files for a single evolution run and concatenates them.
    adds a 'plot_batch_num' column for consistent x-axis plotting.

    args:
        evolution_run_path (str): path to the directory containing batch history csv files.

    returns:
        pd.dataframe: a single dataframe containing data from all batches, or an empty dataframe if no files found.
    """
    all_batch_data = []
    batch_files = sorted(
        [f for f in os.listdir(evolution_run_path) if re.match(r'batch_\d+_history\.csv', f)],
        key=lambda x: int(re.search(r'batch_(\d+)_history\.csv', x).group(1))
    )
    if not batch_files:
        print(f"warning: no batch history files found in {evolution_run_path}")
        return pd.DataFrame()

    for i, batch_file in enumerate(batch_files):
        try:
            batch_df = pd.read_csv(os.path.join(evolution_run_path, batch_file))
            # the 'batch' column in evolve.py logs refers to the morphology evolution batch
            # for these plots, we use the file index as the batch number on x-axis
            batch_df['plot_batch_num'] = i + 1
            all_batch_data.append(batch_df)
        except pd.errors.EmptyDataError:
            print(f"warning: batch file {batch_file} in {evolution_run_path} is empty. skipping.")
        except Exception as e:
            print(f"error loading {batch_file} in {evolution_run_path}: {e}")


    if not all_batch_data:
        return pd.DataFrame()
    return pd.concat(all_batch_data, ignore_index=True)


def plot_evolve_metric_per_method(full_df: pd.DataFrame, metric_col: str, method_name: str, color: str, output_prefix: str,
                                  is_fitness: bool = True, top_n: int = None, alpha_val: float = ALPHA_EVOLVE_INDIVIDUAL_OVERALL) -> None:
    """
    plots a specific metric vs. morphology batch for a single evolution method.
    can optionally plot for top_n individuals per batch.

    args:
        full_df (pd.dataframe): dataframe containing concatenated evolution log data for the method.
        metric_col (str): the name of the column to plot (e.g., 'fitness', 'y_dist', 'voxel_count').
        method_name (str): name of the evolution method.
        color (str): colour to use for plotting.
        output_prefix (str): prefix for the output plot filename.
        is_fitness (bool, optional): if true, plot best metric line. defaults to true.
                                     for 'voxel_count', this means max and min lines are plotted.
        top_n (int, optional): if set, consider only the top n individuals per batch for the metric. defaults to none.
        alpha_val (float, optional): alpha transparency for individual scatter points. defaults to ALPHA_EVOLVE_INDIVIDUAL_OVERALL.

    returns:
        none
    """
    if full_df.empty:
        print(f"no data for {method_name} to plot {metric_col}.")
        return

    plt.figure(figsize=(12, 7))
    plot_batches = full_df['plot_batch_num'].unique()
    avg_metric_per_batch = []
    best_metric_per_batch = [] # only if is_fitness or top_n
    min_metric_per_batch = [] # only for voxel_count

    for batch_num in sorted(plot_batches):
        batch_df = full_df[full_df['plot_batch_num'] == batch_num]
        metric_values = batch_df[metric_col].dropna()
        if metric_col == 'y_dist': # filter y_dist for plotting
            metric_values = metric_values.replace([np.inf, -np.inf], np.nan).dropna()
            metric_values = metric_values[metric_values.abs() < 1000]

        if top_n and not metric_values.empty:
            metric_values = metric_values.nlargest(top_n)

        if not metric_values.empty:
            plt.scatter([batch_num] * len(metric_values), metric_values,
                        color=color, alpha=alpha_val, s=10, edgecolor='none')
            avg_metric_per_batch.append(metric_values.mean())
            if is_fitness or (top_n and metric_col == 'fitness'):
                best_metric_per_batch.append(metric_values.max())
            if metric_col == 'voxel_count': # for voxel_count, 'best' is max, 'min' is min
                min_metric_per_batch.append(metric_values.min())
                best_metric_per_batch.append(metric_values.max()) # 'best_metric_per_batch' stores max for voxel_count
        else:
            avg_metric_per_batch.append(np.nan)
            if is_fitness or (top_n and metric_col == 'fitness'):
                best_metric_per_batch.append(np.nan)
            if metric_col == 'voxel_count':
                min_metric_per_batch.append(np.nan)
                best_metric_per_batch.append(np.nan)

    batches_for_lines = sorted(plot_batches)
    plt.plot(batches_for_lines, avg_metric_per_batch, linestyle='--', color=color, label=f'{method_name} avg')

    if is_fitness or (top_n and metric_col == 'fitness'):
        plt.plot(batches_for_lines, best_metric_per_batch, linestyle='-', color=color, label=f'{method_name} best')
    elif metric_col == 'voxel_count' and min_metric_per_batch and best_metric_per_batch: # ensure lists are not empty
        plt.plot(batches_for_lines, min_metric_per_batch, linestyle='-', color=color, alpha=0.7, label=f'{method_name} min count')
        plt.plot(batches_for_lines, best_metric_per_batch, linestyle='-', color=color, label=f'{method_name} max count')


    ylabel = metric_col.replace('_', ' ').title()
    title_suffix = f" (top {top_n})" if top_n else ""
    plt.xlabel('morphology batch')
    plt.ylabel(ylabel)
    plt.title(f'evolution {ylabel} vs. batch for {method_name.upper()}{title_suffix}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    filename_suffix = f"_top{top_n}" if top_n else ""
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_{metric_col}_vs_batch_{method_name}{filename_suffix}.png'))
    plt.close()

def plot_evolve_fitness_all_methods(evol_runs_data_map: dict, output_prefix: str) -> None:
    """
    plots fitness vs. morphology batch for all evolution methods on a single graph.

    args:
        evol_runs_data_map (dict): a dictionary mapping method names (str) to their concatenated dataframes (pd.dataframe).
        output_prefix (str): prefix for the output plot filename.

    returns:
        none
    """
    plt.figure(figsize=(14, 8))
    for method_name, full_df in evol_runs_data_map.items():
        if full_df.empty:
            continue
        color = METHOD_COLORS.get(method_name, DEFAULT_COLOR)
        plot_batches = full_df['plot_batch_num'].unique()
        avg_fitness_per_batch = []
        best_fitness_per_batch = []

        for batch_num in sorted(plot_batches):
            batch_df = full_df[full_df['plot_batch_num'] == batch_num]
            fitness_values = batch_df['fitness'].dropna()
            if not fitness_values.empty:
                # plot individuals only once per method to avoid legend clutter
                if batch_num == sorted(plot_batches)[0]:
                     plt.scatter([batch_num] * len(fitness_values), fitness_values,
                                color=color, alpha=ALPHA_EVOLVE_INDIVIDUAL_COMBINED, s=10, edgecolor='none',
                                label=f'_{method_name} individuals') # label once
                else:
                     plt.scatter([batch_num] * len(fitness_values), fitness_values,
                                color=color, alpha=ALPHA_EVOLVE_INDIVIDUAL_COMBINED, s=10, edgecolor='none')
                avg_fitness_per_batch.append(fitness_values.mean())
                best_fitness_per_batch.append(fitness_values.max())
            else:
                avg_fitness_per_batch.append(np.nan)
                best_fitness_per_batch.append(np.nan)

        batches_for_lines = sorted(plot_batches)
        plt.plot(batches_for_lines, avg_fitness_per_batch, linestyle='--', color=color, label=f'{method_name} avg fitness')
        plt.plot(batches_for_lines, best_fitness_per_batch, linestyle='-', color=color, label=f'{method_name} best fitness')

    plt.xlabel('morphology batch')
    plt.ylabel('fitness')
    plt.title('evolution fitness vs. batch (all methods)')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_fitness_vs_batch_all_methods.png'))
    plt.close()


def visualise_evolve_top_morphologies_per_method(evolution_run_path: str, method_name: str, output_prefix: str) -> None:
    """
    visualises the top 3 best performing morphologies per batch for a given evolution method.
    morphologies are plotted in a grid.

    args:
        evolution_run_path (str): path to the directory containing batch_x_history.csv files for the method.
        method_name (str): name of the evolution method.
        output_prefix (str): prefix for the output plot filenames.

    returns:
        none
    """
    batch_files = sorted(
        [f for f in os.listdir(evolution_run_path) if re.match(r'batch_\d+_history\.csv', f)],
        key=lambda x: int(re.search(r'batch_(\d+)_history\.csv', x).group(1))
    )
    if not batch_files:
        return

    all_morph_plots_data = [] # stores (batch_num, rank, vsr_obj, best_fit, avg_fit)

    for batch_file_idx, batch_file in enumerate(batch_files):
        current_plot_batch_num = batch_file_idx + 1
        try:
            batch_df = pd.read_csv(os.path.join(evolution_run_path, batch_file))
        except pd.errors.EmptyDataError:
            print(f"skipping empty batch file: {batch_file}")
            continue


        # group by mutation_index to find best fitness for each evolved morphology in this batch
        # each mutation_index represents one unique morphology that was optimised.
        # the 'generation' column within the batch file is from the optimise.py run for that morphology.
        morphology_fitness_summary = []
        for mut_idx, mut_df in batch_df.groupby('mutation_index'):
            best_fitness_for_morph = mut_df['fitness'].max()
            avg_fitness_for_morph = mut_df['fitness'].mean()
            # take the voxel_coords_str from the first row of this mutation group
            voxel_coords_str = mut_df['voxel_coords_str'].iloc[0]
            coords = parse_voxel_coords(voxel_coords_str)
            if not coords:
                # if coords are empty, default to a 10x10x10 grid or skip
                max_dims = (10,10,10) # default
            else:
                # calculate max dimensions from actual coordinates to size the vsr object correctly
                max_dims = (
                    max(c[0] for c in coords) + 1 if coords else 10,
                    max(c[1] for c in coords) + 1 if coords else 10,
                    max(c[2] for c in coords) + 1 if coords else 10,
                )


            morphology_fitness_summary.append({
                'mutation_index': mut_idx,
                'best_fitness': best_fitness_for_morph,
                'avg_fitness': avg_fitness_for_morph,
                'voxel_coords_str': voxel_coords_str,
                'max_dims': max_dims
            })

        if not morphology_fitness_summary:
            continue

        # sort these morphologies by their best_fitness
        sorted_morphologies = sorted(morphology_fitness_summary, key=lambda x: x['best_fitness'], reverse=True)

        # get top 3
        for rank, morph_data in enumerate(sorted_morphologies[:3]):
            # use the calculated max_dims for this specific morphology
            vsr = VoxelRobot(morph_data['max_dims'][0], morph_data['max_dims'][1], morph_data['max_dims'][2])
            coords_list = parse_voxel_coords(morph_data['voxel_coords_str']) # re-parse for safety
            for x,y,z in coords_list:
                vsr.set_val(x,y,z,1)
            all_morph_plots_data.append((current_plot_batch_num, rank + 1, vsr, morph_data['best_fitness'], morph_data['avg_fitness']))

    # now plot all collected morphologies in grids
    num_total_morphs_to_plot = len(all_morph_plots_data)
    morphs_per_figure = N_MORPH_COLS * N_MORPH_ROWS_PER_PLOT_GROUP * 3 # 3 morphologies per group (col cell)

    for fig_idx in range(0, num_total_morphs_to_plot, morphs_per_figure):
        fig = plt.figure(figsize=(N_MORPH_COLS * 1.8, N_MORPH_ROWS_PER_PLOT_GROUP * 9.0)) # adjusted for titles
        
        current_morphs_on_fig = all_morph_plots_data[fig_idx : fig_idx + morphs_per_figure]
        
        plot_idx = 0
        for morph_plot_data_idx, (batch_num, rank, vsr, best_fit, avg_fit) in enumerate(current_morphs_on_fig):
            # calculate position in the grid
            # each "group" is a column cell that holds 3 morphologies stacked vertically
            group_col = (plot_idx // 3) % N_MORPH_COLS
            group_row = (plot_idx // 3) // N_MORPH_COLS
            sub_plot_in_group = plot_idx % 3 # 0, 1, or 2 for top, middle, bottom

            if group_row >= N_MORPH_ROWS_PER_PLOT_GROUP : # should not happen with current slicing
                break

            # create subplot for morphology
            ax = fig.add_subplot(N_MORPH_ROWS_PER_PLOT_GROUP * 3, N_MORPH_COLS,
                                 group_row * N_MORPH_COLS * 3 + group_col + sub_plot_in_group * N_MORPH_COLS + 1,
                                 projection='3d')
            vsr.plot_on_ax(ax)
            title_text = f"b{batch_num}-r{rank}\nbestf:{best_fit:.2f}\navgf:{avg_fit:.2f}" # more compact
            ax.set_title(title_text, fontsize=7)
            plot_idx +=1

        fig.suptitle(f'top morphologies for {method_name.upper()} (figure {fig_idx // morphs_per_figure + 1})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # adjust layout to make space for suptitle
        plt.savefig(os.path.join(OUTPUT_DIR, f'{output_prefix}_top_morphologies_{method_name}_fig{fig_idx // morphs_per_figure + 1}.png'), dpi=150)
        plt.close(fig)


# main execution
if __name__ == '__main__':
    print("starting plotting process...")

    # process optimisation runs
    print("\nprocessing optimisation logs...")
    optimise_dirs = {
        'mlp': os.path.join(BASE_RESULTS_PATH, 'optimisation_mlp'),
        'mlp_plus': os.path.join(BASE_RESULTS_PATH, 'optimisation_mlp_plus'),
        'rnn': os.path.join(BASE_RESULTS_PATH, 'optimisation_rnn'),
    }
    optimise_dfs_map = {}

    for method, path in optimise_dirs.items():
        print(f"  processing {method} from {path}...")
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv') and 'full_history' in f]
        if not csv_files:
            print(f"    no full_history csv found for {method}. skipping.")
            continue
        csv_path = os.path.join(path, csv_files[0]) # assume one relevant csv
        try:
            df = pd.read_csv(csv_path)
            optimise_dfs_map[method] = df
            plot_optimise_fitness_per_method(df, method, METHOD_COLORS.get(method, DEFAULT_COLOR), "optimise")
            plot_optimise_ydist_per_method(df, method, METHOD_COLORS.get(method, DEFAULT_COLOR), "optimise")
        except Exception as e:
            print(f"    error processing {csv_path}: {e}")

    if optimise_dfs_map:
        plot_optimise_fitness_all_methods(optimise_dfs_map, "optimise")

    # process evolution runs
    print("\nprocessing evolution logs...")
    evolution_base_dirs_map = {
        'mlp': os.path.join(BASE_RESULTS_PATH, 'evolution_mlp'),
        'mlp_2': os.path.join(BASE_RESULTS_PATH, 'evolution_mlp_2'),
        'mlp_plus': os.path.join(BASE_RESULTS_PATH, 'evolution_mlp_plus'),
        'rnn': os.path.join(BASE_RESULTS_PATH, 'evolution_rnn'),
    }
    evol_runs_data_map = {} # to store concatenated dfs for combined plot

    for method_key, base_dir_path in evolution_base_dirs_map.items():
        print(f"  processing {method_key} from {base_dir_path}...")
        # find the actual evolution run directory (e.g., co_evolve_...)
        sub_dirs = [d for d in os.listdir(base_dir_path) if os.path.isdir(os.path.join(base_dir_path, d)) and d.startswith('co_evolve')]
        if not sub_dirs:
            print(f"    no 'co_evolve_*' directory found in {base_dir_path}. skipping.")
            continue
        evolution_run_path = os.path.join(base_dir_path, sub_dirs[0]) # assume one run per method folder
        print(f"    found evolution run: {evolution_run_path}")

        full_run_df = load_evolution_run_data(evolution_run_path)
        if full_run_df.empty:
            print(f"    no data loaded for {method_key}. skipping plots for this method.")
            continue

        evol_runs_data_map[method_key] = full_run_df # store for combined plot
        color = METHOD_COLORS.get(method_key, DEFAULT_COLOR)

        # plot fitness (all individuals)
        plot_evolve_metric_per_method(full_run_df, 'fitness', method_key, color, "evolve", is_fitness=True)
        # plot y-distance (all individuals)
        plot_evolve_metric_per_method(full_run_df, 'y_dist', method_key, color, "evolve", is_fitness=False)
        # plot fitness (top 50)
        plot_evolve_metric_per_method(full_run_df, 'fitness', method_key, color, "evolve", is_fitness=True, top_n=50, alpha_val=ALPHA_EVOLVE_TOP_N_INDIVIDUAL)

        # plot voxel count
        full_run_df['voxel_count'] = full_run_df['voxel_coords_str'].apply(count_voxels)
        plot_evolve_metric_per_method(full_run_df, 'voxel_count', method_key, color, "evolve", is_fitness=False) # is_fitness=false for voxel_count specific lines

        # visualise top morphologies
        print(f"    generating morphology visualisations for {method_key}...")
        visualise_evolve_top_morphologies_per_method(evolution_run_path, method_key, "evolve")


    if evol_runs_data_map:
         plot_evolve_fitness_all_methods(evol_runs_data_map, "evolve")

    print("\nplotting process finished. check the '{}' directory.".format(OUTPUT_DIR))