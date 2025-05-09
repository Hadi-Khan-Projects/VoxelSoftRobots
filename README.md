![morph_vsr_banner](https://github.com/user-attachments/assets/6a9d8629-20eb-4a71-a47b-42f3e8b4a583)

# MorphVSR: A Voxel-based Soft Robot Simulation and Co-evolution Framework

This repository contains a Python-based framework for simulating, controlling, and evolving Voxel-based Soft Robots (VSRs) using the MuJoCo physics engine. The project allows for the optimisation of distributed neural network controllers for fixed VSR morphologies and the co-evolution of both morphology and control strategies.

This project was developed as a part of a Final Year Project submission for the MEng Computer Science program at University College London. In this README we provide:
*   A) Video Repository
*   B) Data Repository
*   C) Getting Started
*   D) Running Experiments

## A) Video Repository

https://github.com/user-attachments/assets/e95ab22f-9012-4627-a2df-72e49bf69ada

Experiments took ~89 hours to run. Some videos of experiment results, emergent behaviours, evolved morphologies available below:

### 1. Optimised Quadruped Locomotion with MLP Controller

**Best Performing Run (Gen 15, Ind 16):**

https://github.com/user-attachments/assets/c882f8b0-8eb2-4d4b-8c23-95437f56949d

https://github.com/user-attachments/assets/9e58390f-ee72-4e08-9d93-5acec5915d22

**2nd Best Performing Run (Gen 28, Ind 40):**

https://github.com/user-attachments/assets/cb773027-491a-4991-bb20-bd0e07153cbf

https://github.com/user-attachments/assets/1e8233ca-d983-4a40-a1e9-ffae786a4c36

### 2. Optimised Quadruped Locomotion with MLP+ Controller

**Best Performing Run (Gen 43, Ind 19):**

https://github.com/user-attachments/assets/8b4da943-00a6-4f1a-8f03-790d64a4b8d2

https://github.com/user-attachments/assets/f2579c0a-726f-47f0-b558-ada53fbf2d6d

**2nd Best Performing Run (Gen 79, Ind 11):**

https://github.com/user-attachments/assets/9761c9cc-57d7-4360-a229-3b6015a7e6ec

https://github.com/user-attachments/assets/91f9ce55-4b66-4138-953f-42d5d25635b9

### 3. Optimised Quadruped Locomotion with RNN Controller

**Best Performing Run (Gen 72, Ind 38):**

https://github.com/user-attachments/assets/53613480-9a1a-451f-b8cc-d533c673bef8

https://github.com/user-attachments/assets/6f83e841-0952-4140-92f8-15f7d8a8721a

**Run Exhibiting Trotting Gait (Gen 34, Ind 14):**

https://github.com/user-attachments/assets/ed01931e-89dd-4fd6-93f9-403c845a1622

https://github.com/user-attachments/assets/800bfd81-3dc9-4dbf-8236-3d53ce32a82a

### 4. Co-evolved Morphology and Locomotion with MLP Controller

**Best Performing Run (Batch 27, Mut 4, Gen 13, Ind 15):**

https://github.com/user-attachments/assets/330feb78-cc19-42c5-8fb2-ac0774264ddf

### 5. Co-evolved Morphology and Locomotion with MLP+ Controller

**Best Performing Run (Batch 9, Mut 5, Gen 14, Ind 18):**

https://github.com/user-attachments/assets/c921828e-c8c3-4be6-9e3a-f8be4552519a

### 6. Co-evolved Morphology and Locomotion with RNN Controller

**Best Performing Run (Batch 11, Mut 12, Gen 15, Ind 5):**

https://github.com/user-attachments/assets/3f40020e-8198-4ef0-ab83-abea926a5da7

## B) Data Repository

Data for all 7 experiment runs, are available below:

**[Experiments Data Download (5.78 GB)](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcabhk0_ucl_ac_uk/El0-7MIjyrxBmoKIpEhcnZ0BhGSSqqOCzxvJsSCyaI47Qg?e=evwzHb)**

The data contains the following folders/files:

```
results_experiments
├── evolution_mlp
│   ├── co_evolve_4x2x2_20250501-082218
│   │   ├── batch_1_history.csv
│   │   ├── batch_2_history.csv
│   │   ├── ...
│   │   ├── batch_29_history.csv
│   │   ├── initial_model/...
│   │   └── models/...
│   └── evolution_run_mlp.log
├── evolution_mlp_2
│   ├── co_evolve_4x2x2_20250501-154414
│   │   ├── batch_1_history.csv
│   │   ├── batch_2_history.csv
│   │   ├── ...
│   │   ├── batch_19_history.csv
│   │   ├── initial_model/...
│   │   └── models/...
│   └── evolution_run_mlp.log
├── evolution_mlp_plus
│   ├── co_evolve_4x2x2_20250501-112931
│   │   ├── batch_1_history.csv
│   │   ├── batch_2_history.csv
│   │   ├── ...
│   │   ├── batch_14_history.csv
│   │   ├── initial_model/...
│   │   └── models/...
│   └── evolution_run_mlp_plus.log
├── evolution_rnn
│   ├── co_evolve_4x2x2_20250501-113006
│   │   ├── batch_1_history.csv
│   │   ├── batch_2_history.csv
│   │   ├── ...
│   │   ├── batch_13_history.csv
│   │   ├── initial_model/...
│   │   └── models/...
│   └── evolution_run_rnn.log
├── optimisation_mlp
│   ├── optimisation_run_mlp.log
│   ├── quadruped_v3_best_params_mlp_gen80_pop60.npy
│   └── quadruped_v3_full_history_mlp_gen80_pop60.csv
├── optimisation_mlp_plus
│   ├── evolution_run_mlp_plus.log
│   ├── quadruped_v3_best_params_mlp_plus_h24_16_gen80_pop60.npy
│   └── quadruped_v3_full_history_mlp_plus_h24_16_gen80_pop60.csv
└── optimisation_rnn
    ├── optimisation_run_rnn.log
    ├── quadruped_v3_best_params_rnn_h16_gen80_pop60.npy
    └── quadruped_v3_full_history_rnn_h16_gen80_pop60.csv
```

## C) Getting Started

1.  **Clone the repository:**
    
    ```bash
    git clone https://github.com/Hadi-Khan-Projects/VoxelSoftRobots.git
    ```

2.  **Install Python dependencies:**

    We recommend creating a virtual environment, this project uses Python 3.12+ and MuJoCo 3.2.4+.
    
    For Linux:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    
    For Windows:
    ```bash
    python -m venv venv
    source venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **[Optional] Download Experiments Data:**

    If you wish to rerun simulations of experiments or plot the results of experiments, download the data and place it within your repository root `<repo_root>/results_experiments/`.

## D) Running Experiments

We provide command-line interfaces (CLIs) for running simulations, optimisations, and evolutions.

### 1. Running a Single Simulation (`simulate.py`)

Use `simulate.py` to execute a standalone simulation of a predefined VSR or visualise and validate specific individuals from optimisation/evolution history logs.

**Replaying individuals from Controller Optimisation logs (`optimise.py` output):**
```bash
python simulate.py \
    --csv_path <path_to_optimise_history.csv> \
    --generation <gen_number> \
    --individual_index <index_number> \
    --vsr_grid_dims <X> <Y> <Z> \
    [--duration <seconds>] \
    [--control_timestep <seconds>] \
    [--headless]
```

**For Example:**
```bash
python simulate.py \
    --csv_path results_experiments/optimisation_mlp_plus/quadruped_v3_full_history_mlp_plus_h24_16_gen80_pop60.csv \
    --generation 43 \
    --individual_index 19 \
    --vsr_grid_dims 10 10 10
```

**Replaying individuals from Morphology Evolution logs (`evolve.py` output):**
```bash
python simulate.py \
    --csv_path <path_to_evolve_history.csv> \
    --batch <batch_number> \
    --mutation_index <mutation_number> \
    --generation <gen_number_within_optimise> \
    --individual_index <index_number_within_gen> \
    --vsr_grid_dims <X> <Y> <Z> \
    [--duration <seconds>] \
    [--control_timestep <seconds>] \
    [--headless]
```

**For Example:**
```bash
python simulate.py \
    --csv_path results_experiments/evolution_mlp/co_evolve_4x2x2_20250501-082218/batch_29_history.csv \
    --batch 29 \
    --mutation_index 4 \
    --generation 2 \
    --individual_index 27 \
    --vsr_grid_dims 10 10 10
```

Note: The additional `--batch` and `--mutation_index` parameters. The `--generation` and `--individual_index` here refer to the nested controller optimisation run for that specific mutated morphology.

Note: For evolve.py logs, `--csv_path` should point to a specific `batch_X_history.csv` file as the `full_evolution_history.csv` is not necessarily available for runs that were terminated before completion.

**Parameters for `simulate.py`:**
*   `--csv_path`: Path to the history CSV file.
*   `--generation` / `--individual_index`: Identifiers for the specific individual within a controller optimisation run.
*   `--batch` / `--mutation_index`: Additional identifiers for individuals within a morphology evolution run.
*   `--vsr_grid_dims`: Dimensions (X Y Z) of the voxel grid (e.g., 10 10 10). Must match the grid used for the VSR morphology being simulated.
*   `--duration`: Simulation duration in seconds.
*   `--control_timestep`: Time interval between controller updates.
*   `--headless`: Run without GUI.

### 2. Optimising Controllers for a Fixed Morphology (`optimise.py`)

Use `optimise.py` to evolve the parameters of a neural controller for a VSR with a fixed physical structure.

**Running Opimisation for Fixed Morphology:**
```bash
python optimise.py \
    --model_name <name_of_your_model_csv_without_extension> \
    --controller_type <mlp|mlp_plus|rnn> \
    --gear <gear_ratio_float> \
    [--num_generations <int>] \
    [--population_size <int>] \
    [--tournament_size <int>] \
    [--crossover_probability <float_0_to_1>] \
    [--mutation_sigma <float>] \
    [--clip_range_min <float>] \
    [--clip_range_max <float>] \
    [--mlp_plus_hidden_sizes <size1> <size2> ...] \
    [--rnn_hidden_size <size>] \
    [--simulation_duration <seconds>] \
    [--control_timestep <seconds>] \
    [--num_workers <int>] \
    [--initial_param_vector_path <path_to_npy_file>] \
    [--output_dir <path>]
```

**For Example:**
```bash
python optimise.py \
    --model_name quadruped_v3 \
    --controller_type rnn \
    --gear 100.0 \
    --num_generations 50 \
    --population_size 120 \
    --tournament_size 6 \
    --mutation_sigma 0.2 \
    --clip_range_min -5.0 \
    --clip_range_max 5.0 \
    --rnn_hidden_size 16 \
    --simulation_duration 60 \
    --control_timestep 0.2 \
    --num_workers 60 \
    --output_dir results_optimise/my_rnn_run_fixed_quad
```

**Parameters for `optimise.py`:**
*   Model Configuration:
    *   `--model_name`: Name of the VSR model CSV file (e.g., quadruped_v3). The script expects vsr_models/<model_name>/<model_name>.csv.
    *   `--gear`: Actuation force scaling factor.
*   Controller Configuration:
    *   `--controller_type`: mlp, mlp_plus, or rnn.
    *   `--mlp_plus_hidden_sizes`: E.g., 24 16 for two hidden layers.
    *   `--rnn_hidden_size`: E.g., 16.
*   Evolutionary Algorithm Parameters:
    *   `--num_generations`, `--population_size`, `--tournament_size`, `--crossover_probability`, `--mutation_sigma`, `--clip_range_min`, `--clip_range_max`.
*   Simulation & Execution Parameters:
    *   `--simulation_duration`, `--control_timestep`, `--num_workers`, `--output_dir`, `--initial_param_vector_path`.

### 3. Co-evolving Morphology and Control (`evolve.py`)

Use `evolve.py` to simultaneously evolve the VSR's physical structure and its neural controller.

**Running Co-evolution:**
```bash
python evolve.py \
    --initial_morphology_csv <path_to_start_shape.csv> \
    --output_dir <path_for_all_evolution_results> \
    --controller_type <mlp|mlp_plus|rnn> \
    --gear <gear_ratio_float> \
    [--num_batches <int>] \
    [--num_mutations_per_batch <int>] \
    [--num_parents_select <int>] \
    [--optimise_generations <int>] \
    [--optimise_population_size <int>] \
    [--optimise_num_workers <int>] \
    [--tournament_size <int>] \
    [--crossover_probability <float_0_to_1>] \
    [--mutation_sigma <float>] \
    [--clip_range_min <float>] \
    [--clip_range_max <float>] \
    [--mlp_plus_hidden_sizes <size1> <size2> ...] \
    [--rnn_hidden_size <size>] \
    [--simulation_duration <seconds>] \
    [--control_timestep <seconds>] \
    [--vsr_grid_dims <X> <Y> <Z>] \
    [--mutation_verbose]
```

**For Example:**
```bash
python evolve.py \
    --initial_morphology_csv vsr_models/block_4x2x2/block_4x2x2.csv \
    --output_dir results_evolution/coevolve_mlp_run_1 \
    --controller_type mlp \
    --gear 80.0 \
    --num_batches 20 \
    --num_mutations_per_batch 20 \
    --num_parents_select 4 \
    --optimise_generations 20 \
    --optimise_population_size 120 \
    --optimise_num_workers 30 \
    --tournament_size 6 \
    --mutation_sigma 0.2 \
    --clip_range_min -5.0 \
    --clip_range_max 5.0 \
    --simulation_duration 60 \
    --control_timestep 0.2 \
    --vsr_grid_dims 10 10 10 \
    --mutation_verbose
```

**Parameters for `evolve.py`:**
*   Morphology Evolution Parameters:
    *   `--initial_morphology_csv`: Path to the CSV defining the starting VSR shape.
    *   `--num_batches`: Number of morphology evolution iterations.
    *   `--num_mutations_per_batch`: New morphologies generated/tested per batch.
    *   `--num_parents_select`: Top morphologies selected for the next batch.
    *   `--vsr_grid_dims`: Dimensions of the voxel grid (e.g., 10 10 10).
    *   `--mutation_verbose`: Enable detailed logging for morphology mutation.
*   Nested Controller Optimisation Parameters: (--optimise_generations, --optimise_population_size, --optimise_num_workers).
*   Other parameters are similar to optimise.py and apply to the nested controller optimization runs.

### 4. Plotting Results

You will need experiments data at `<repo_root>/results_experiments/`. Then run:

```bash
python results_plots.py
```

Plots will appear under `results_plots/`
