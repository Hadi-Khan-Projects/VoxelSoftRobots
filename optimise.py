# optimise.py
import os
import time
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count

import mujoco
import numpy as np
import pandas as pd

from controller import DistributedNeuralController  # Import updated controller

# Import necessary classes
from simulate import run_simulation
from vsr import VoxelRobot

# --- HELPER FUNCTIONS ---


def _get_vsr_details_and_base_dims(model):
    """
    Analyzes the VSR model to find active voxels and determine
    *base* controller input/output dimensions (independent of network type).
    """
    try:
        voxel_motor_map = {}
        voxel_tendon_map = {}

        # Map motors
        for i in range(model.nu):
            motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if motor_name and motor_name.startswith("voxel_"):
                parts = motor_name.split("_")
                voxel_coord = tuple(map(int, parts[1:4]))
                if voxel_coord not in voxel_motor_map:
                    voxel_motor_map[voxel_coord] = []
                voxel_motor_map[voxel_coord].append(i)

        # Map tendons
        for i in range(model.ntendon):
            tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
            if tendon_name and tendon_name.startswith("voxel_"):
                parts = tendon_name.split("_")
                voxel_coord = tuple(map(int, parts[1:4]))
                if voxel_coord not in voxel_tendon_map:
                    voxel_tendon_map[voxel_coord] = []
                voxel_tendon_map[voxel_coord].append(i)

        # Filter for valid voxels (must have 4 motors and 4 tendons)
        potential_active_coords = sorted(
            list(set(voxel_motor_map.keys()) | set(voxel_tendon_map.keys()))
        )
        active_voxel_coords = []
        for coord in potential_active_coords:
            # check if coord exists as a key AND has the correct length list
            has_motors = (
                coord in voxel_motor_map and len(voxel_motor_map.get(coord, [])) == 4
            )
            has_tendons = (
                coord in voxel_tendon_map and len(voxel_tendon_map.get(coord, [])) == 4
            )
            if has_motors and has_tendons:
                active_voxel_coords.append(coord)

        n_active_voxels = len(active_voxel_coords)
        if n_active_voxels == 0:
            raise ValueError("No valid active voxels found in the model.")
        print(f"Found {n_active_voxels} active voxels.")

        # Determine Controller BASE Dimensions (must match controller.py)
        N_SENSORS_PER_VOXEL = 8
        N_COMM_CHANNELS = 2
        N_COMM_DIRECTIONS = 6
        N_TIME_INPUTS = 2
        N_COM_VEL_INPUTS = 3
        N_TARGET_ORIENT_INPUTS = 2

        base_input_size = (
            N_SENSORS_PER_VOXEL
            + N_COMM_DIRECTIONS * N_COMM_CHANNELS
            + 1  # Driving signal
            + N_TIME_INPUTS
            + N_COM_VEL_INPUTS
            + N_TARGET_ORIENT_INPUTS
        )
        base_output_size = 1 + N_COMM_DIRECTIONS * N_COMM_CHANNELS

        print(f"Controller Base Input Size: {base_input_size}")
        print(f"Controller Base Output Size: {base_output_size}")

        # return base dimensions and voxel info needed by controller
        return n_active_voxels, active_voxel_coords, base_input_size, base_output_size

    except Exception as e:
        print(f"Error getting VSR details: {e}")
        print(traceback.format_exc())
        raise


def get_param_vector_size(
    controller_type,
    base_input_size,
    base_output_size,
    mlp_plus_hidden_sizes,
    rnn_hidden_size,
):
    """Calculates the total number of parameters for a given controller configuration."""
    # Instantiating a temporary controller just to use its setup logic
    # feed dummy voxel info, actual values don't matter for size calculation
    temp_controller = DistributedNeuralController(
        controller_type=controller_type,
        n_voxels=1,
        voxel_coords=[(0, 0, 0)],
        n_sensors_per_voxel=8,  # value doesn't impact structure size logic directly here
        n_comm_channels=2,  # value doesn't impact structure size logic directly here
        # Pass the actual configurations
        mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
    )
    # override the base sizes calculated internally with the correct ones from model analysis
    temp_controller.base_input_size = base_input_size
    temp_controller.base_output_size = base_output_size
    # re-run the setup logic with correct base sizes
    if controller_type == "mlp":
        temp_controller._setup_mlp()
    elif controller_type == "mlp_plus":
        temp_controller._setup_mlp_plus()
    elif controller_type == "rnn":
        temp_controller._setup_rnn()

    return temp_controller.get_total_parameter_count()


def evaluate_individual(
    param_vector,  # parameter vector for this individual
    # Arguments passed via partial
    xml_string,  # pass XML string instead of direct model
    duration,
    control_timestep,
    # Controller Config
    controller_type,
    mlp_plus_hidden_sizes,
    rnn_hidden_size,
    # VSR/Sim metadata for logging
    voxel_coords_list_str,
    simulation_timestep,
    gear_ratio,
):
    """
    Evaluates a single individual. Loads the model from XML within the worker.
    """
    model = None  # Initialize model to None
    try:
        # Load Model INSIDE the worker
        model = mujoco.MjModel.from_xml_string(xml_string)

        # --- Run Simulation ---
        fitness, x_dist, y_dist, reached = run_simulation(
            model=model,  # Pass locally loaded model
            duration=duration,
            control_timestep=control_timestep,
            param_vector=param_vector,
            # Controller configuration
            controller_type=controller_type,
            mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
            rnn_hidden_size=rnn_hidden_size,
            headless=True,
        )
        # ensure types are correct
        fitness = (
            float(fitness) if not isinstance(fitness, (float, np.float64)) else fitness
        )
        x_dist = (
            float(x_dist) if not isinstance(x_dist, (float, np.float64)) else x_dist
        )
        y_dist = (
            float(y_dist) if not isinstance(y_dist, (float, np.float64)) else y_dist
        )
        reached = bool(reached) if not isinstance(reached, bool) else reached

        # results and metadata
        return (
            fitness,
            x_dist,
            y_dist,
            reached,
            param_vector,
            controller_type,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
            voxel_coords_list_str,
            control_timestep,
            model.opt.timestep,
            gear_ratio,  # Log actual timestep used
        )

    except mujoco.FatalError as e:
        print(f"!!! MuJoCo Fatal Error evaluating individual: {e} !!!")
        sim_ts = (
            model.opt.timestep if model else simulation_timestep
        )  # log expected if load failed
        return (
            -np.inf,
            np.inf,
            np.inf,
            False,
            param_vector,
            controller_type,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
            voxel_coords_list_str,
            control_timestep,
            sim_ts,
            gear_ratio,
        )
    except Exception as e:
        print(f"!!! Error evaluating individual: {e} !!!")
        print(f"Traceback: {traceback.format_exc()}")
        sim_ts = model.opt.timestep if model else simulation_timestep
        return (
            -np.inf,
            np.inf,
            np.inf,
            False,
            param_vector,
            controller_type,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
            voxel_coords_list_str,
            control_timestep,
            sim_ts,
            gear_ratio,
        )
    # ensure model resources are freed if necessary, though worker process exit usually handles this.

    except mujoco.FatalError as e:
        print(f"!!! MuJoCo Fatal Error evaluating individual: {e} !!!")
        # return structure must match the success case for consistent processing
        return (
            -np.inf,
            np.inf,
            np.inf,
            False,
            param_vector,
            controller_type,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
            voxel_coords_list_str,
            control_timestep,
            simulation_timestep,
            gear_ratio,
        )
    except Exception as e:
        print(f"!!! Error evaluating individual: {e} !!!")
        print(f"Traceback: {traceback.format_exc()}")
        # return structure must match the success case
        return (
            -np.inf,
            np.inf,
            np.inf,
            False,
            param_vector,
            controller_type,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
            voxel_coords_list_str,
            control_timestep,
            simulation_timestep,
            gear_ratio,
        )


def tournament_selection(population, fitnesses, tournament_size):
    """Performs tournament selection. Handles potential empty lists."""
    num_individuals = len(population)
    if num_individuals == 0:
        raise ValueError("Empty population for tournament.")
    tournament_size = min(tournament_size, num_individuals)
    # ensure fitnesses align with population
    if len(fitnesses) != num_individuals:
        raise ValueError("Fitness length mismatch.")
    selected_indices = np.random.choice(
        num_individuals, size=tournament_size, replace=False
    )  # Sample without replacement if possible
    # get fitness values corresponding to the selected indices
    tournament_fitnesses = [fitnesses[i] for i in selected_indices]
    # find the index of the maximum fitness *within the tournament fitnesses list*
    winner_local_idx = np.argmax(tournament_fitnesses)
    # map this back to the index in the *original population list*
    winner_global_idx = selected_indices[winner_local_idx]
    return population[winner_global_idx]


def geometric_crossover(parent1, parent2):
    """Performs geometric crossover."""
    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)
    if parent1.shape != parent2.shape:
        raise ValueError("Parent shapes mismatch.")
    beta = np.random.uniform(-1.0, 2.0, size=parent1.shape)
    offspring = parent1 + beta * (parent2 - parent1)
    return offspring


def gaussian_mutation(individual, sigma, clip_range):
    """Performs Gaussian mutation and clips the result."""
    individual = np.asarray(individual)
    noise = np.random.normal(0, sigma, size=individual.shape)
    mutated = individual + noise
    mutated = np.clip(mutated, clip_range[0], clip_range[1])
    return mutated


# --- MAIN OPTIMISATION FUNCTION ---


def optimise(
    xml_string: str,  # pass XML string instead of direct model
    vsr_voxel_coords_list,  # The actual list of (x,y,z) tuples
    vsr_gear_ratio,
    # EA Params
    controller_type: str,
    num_generations: int,
    population_size: int,
    tournament_size: int,
    crossover_probability: float,
    mutation_sigma: float,
    clip_range: tuple,
    # Simulation Params
    simulation_duration: int,
    control_timestep: float,
    num_workers: int,
    initial_param_vector: np.ndarray = None,
    # Controller Config
    mlp_plus_hidden_sizes: list = [32, 32],
    rnn_hidden_size: int = 32,
):
    """
    Optimises the VSR controller parameters using a generational EA.
    Loads the model from xml_string within each worker process.
    """
    # --- Load model once in main process ONLY for setup analysis ---
    try:
        setup_model = mujoco.MjModel.from_xml_string(xml_string)
        print("Model loaded in main process for setup analysis.")
    except Exception as e:
        print(
            f"Fatal Error: Could not load model from XML string for setup. Aborting.\nError: {e}"
        )
        return None, pd.DataFrame()

    print("--- Starting Optimisation ---")
    print(f"Controller Type: {controller_type}")
    if controller_type == "mlp_plus":
        print(f"  MLP+ Hidden: {mlp_plus_hidden_sizes}")
    if controller_type == "rnn":
        print(f"  RNN Hidden: {rnn_hidden_size}")

    num_workers = min(num_workers, cpu_count())
    print(f"Using {num_workers} workers.")

    # 1. Get VSR details using the setup_model
    try:
        n_voxels, active_voxel_coords_from_model, base_input_size, base_output_size = (
            _get_vsr_details_and_base_dims(setup_model)  # use the locally loaded model
        )
        if sorted(active_voxel_coords_from_model) != sorted(vsr_voxel_coords_list):
            print(
                "Warning: Voxel coordinates from model analysis differ from provided list."
            )
            print("  Proceeding with the explicitly provided voxel coordinate list.")
    except Exception as e:
        print(f"Fatal Error: Could not initialise VSR details. Aborting. \nError: {e}")
        return None, pd.DataFrame()

    # 2. Calculate Parameter Vector Size
    try:
        param_vector_size = get_param_vector_size(
            controller_type,
            base_input_size,
            base_output_size,
            mlp_plus_hidden_sizes,
            rnn_hidden_size,
        )
        print(f"Total parameters per individual: {param_vector_size}")
    except Exception as e:
        print(
            f"Fatal Error: Could not calculate parameter vector size. Aborting. \nError: {e}"
        )
        return None, pd.DataFrame()

    # 3. Prepare Simulation Metadata
    simulation_timestep = (
        setup_model.opt.timestep
    )  # get expected timestep from setup model
    voxel_coords_list_str = str(vsr_voxel_coords_list)
    gear_ratio = vsr_gear_ratio
    del setup_model  # free the setup model, workers will load their own

    # 4. Initialise Population
    population = []
    print(f"Initializing population of size {population_size}...")
    if initial_param_vector is not None:
        if len(initial_param_vector) == param_vector_size:
            population.append(np.copy(initial_param_vector))
            print("Using provided initial parameter vector.")
        else:
            print("Warning: Initial vector size mismatch. Ignoring.")
    while len(population) < population_size:
        random_params = np.random.uniform(
            clip_range[0] / 2, clip_range[1] / 2, size=param_vector_size
        )
        population.append(random_params)
    print("Population initialized.")

    # 5. Evolutionary Loop Setup
    best_fitness_so_far = -np.inf
    best_param_vector_so_far = None
    all_history_records = []

    # --- Create partial function ---
    evaluate_partial = partial(
        evaluate_individual,
        xml_string=xml_string,  # pass XML string instead of direct model
        duration=simulation_duration,
        control_timestep=control_timestep,
        # Controller Config
        controller_type=controller_type,
        mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        # Metadata
        voxel_coords_list_str=voxel_coords_list_str,
        simulation_timestep=simulation_timestep,
        gear_ratio=gear_ratio,
    )
    print("\n")

    # --- 6. Evolutionary Loop ---
    for generation in range(num_generations):
        gen_start_time = time.time()
        print(f"\n--- Generation {generation + 1}/{num_generations} ---")

        # 7. Evaluate Population (Pool map call)
        print(
            f"Evaluating {len(population)} individuals using {num_workers} workers..."
        )
        eval_start_time = time.time()
        results = []
        current_population_to_eval = population

        try:
            actual_workers = min(num_workers, len(current_population_to_eval))
            if actual_workers <= 0:
                print("Warning: No individuals to evaluate.")
                continue

            with Pool(processes=actual_workers) as pool:
                results = pool.map(evaluate_partial, current_population_to_eval)

        except Exception as e:
            print(f"!!! Fatal Error during parallel evaluation: {e} !!!")
            print(f"Traceback: {traceback.format_exc()}")
            history_df = pd.DataFrame(all_history_records)
            return (
                best_param_vector_so_far,
                history_df,
            )  # Return current best and history

        eval_time = time.time() - eval_start_time
        print(f"Evaluation finished in {eval_time:.2f} seconds.")

        # 8. Process results
        fitnesses = []
        evaluated_population_params = []
        gen_best_fitness = -np.inf
        gen_best_params = None
        num_failed = 0

        for i, res in enumerate(results):
            # unpack results, simulation_timestep/res_sim_ts is now the one reported by worker ---
            (
                fitness,
                x_dist,
                y_dist,
                reached,
                params_vector,
                res_ctrl_type,
                res_mlp_hiddens,
                res_rnn_hidden,
                res_vox_str,
                res_ctrl_ts,
                res_sim_ts,
                res_gear,
            ) = res

            fitnesses.append(fitness)
            evaluated_population_params.append(params_vector)

            if fitness <= -np.inf:
                num_failed += 1
            elif fitness > gen_best_fitness:
                gen_best_fitness = fitness
                gen_best_params = params_vector

            # add record to history
            record = {
                "generation": generation + 1,
                "individual_index": i,
                "fitness": fitness,
                "x_dist": x_dist,
                "y_dist": y_dist,
                "reached": reached,
                "params_vector_str": str(list(params_vector))
                if params_vector is not None
                else None,
                "controller_type": res_ctrl_type,
                "mlp_plus_hidden_sizes_str": str(res_mlp_hiddens),
                "rnn_hidden_size": res_rnn_hidden,
                "voxel_coords_str": res_vox_str,
                "control_timestep": res_ctrl_ts,
                "simulation_timestep": res_sim_ts,
                "gear_ratio": res_gear,
            }
            all_history_records.append(record)

        if num_failed > 0:
            print(f"Warning: {num_failed}/{len(results)} evaluations failed.")

        valid_fitnesses = [f for f in fitnesses if f > -np.inf]
        avg_gen_fitness = np.mean(valid_fitnesses) if valid_fitnesses else -np.inf
        print(f"Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"Average fitness (valid):   {avg_gen_fitness:.4f}")

        # Update overall best
        if gen_best_fitness > best_fitness_so_far:
            if gen_best_params is not None:
                best_fitness_so_far = gen_best_fitness
                best_param_vector_so_far = np.copy(gen_best_params)
                print(
                    f"*** New overall best fitness found: {best_fitness_so_far:.4f} ***"
                )
            else:
                print("Warning: Gen best fitness improved, but params vector was None.")

        # 9. Selection
        valid_population_params = [
            p
            for p, f in zip(evaluated_population_params, fitnesses)
            if f > -np.inf and p is not None
        ]
        valid_pop_fitnesses = [f for f in fitnesses if f > -np.inf]
        if not valid_population_params:
            print("Warning: No valid individuals for selection! Re-initializing.")
            population = [
                np.random.uniform(clip_range[0], clip_range[1], size=param_vector_size)
                for _ in range(population_size)
            ]
            continue
        parents = [
            tournament_selection(
                valid_population_params, valid_pop_fitnesses, tournament_size
            )
            for _ in range(population_size)
        ]

        # 10. Variation
        offspring_population = []
        # crossover/mutation logic
        parent_indices = np.random.permutation(len(parents))
        for i in range(population_size):
            idx1 = parent_indices[i]
            idx2 = parent_indices[
                (i + np.random.randint(1, max(1, len(parents)))) % max(1, len(parents))
            ]
            parent1 = parents[idx1]
            parent2 = parents[
                idx2 if idx1 != idx2 or len(parents) == 1 else (idx1 + 1) % len(parents)
            ]
            if np.random.rand() < crossover_probability and len(parents) > 1:
                offspring = geometric_crossover(parent1, parent2)
            else:
                offspring = gaussian_mutation(parent1, mutation_sigma, clip_range)
            offspring_population.append(offspring)
        population = offspring_population[:population_size]

        # generation end
        gen_time = time.time() - gen_start_time
        print(f"Generation {generation + 1} finished in {gen_time:.2f} seconds.")

    # --- End of Optimisation ---
    print("\n--- Optimisation Finished ---")
    history_df = pd.DataFrame(all_history_records)
    if best_param_vector_so_far is None:
        print("Warning: No best individual found.")
    else:
        print(f"Overall best fitness achieved: {best_fitness_so_far:.4f}")
    return best_param_vector_so_far, history_df


# example on how to use actual in evolve.py
if __name__ == "__main__":
    # CONFIG: Evolutionary Algorithm
    CONTROLLER_TYPE = "rnn"  # 'mlp', 'mlp_plus', 'rnn'
    NUM_WORKERS = 30
    NUM_GENERATIONS = 200
    POPULATION_SIZE = 60  # paper used 250
    TOURNAMENT_SIZE = 6  # paper used 8

    CROSSOVER_PROBABILITY = 0.8
    MUTATION_SIGMA = 0.2  # paper used 0.15
    CLIP_RANGE = (-5.0, 5.0)  # clip parameters to this range

    # CONFIG: Controller
    MLP_PLUS_HIDDEN_SIZES = [24, 16]  # Example for mlp_plus
    RNN_HIDDEN_SIZE = 16  # Example for rnn

    # CONFIG: Simulation
    SIMULATION_DURATION = 60
    CONTROL_TIMESTEP = 0.2  # paper used 0.05
    GEAR = 100
    VSR_GRID_DIMS = (10, 10, 10)

    # CONFIG: Model
    MODEL_NAME = "quadruped_v3_copy"
    MODEL_BASE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}"
    MODEL_CSV_PATH = MODEL_BASE_PATH + ".csv"

    if not os.path.exists(MODEL_CSV_PATH):
        print(f"Error: Model CSV file not found at {MODEL_CSV_PATH}")
    else:
        print(
            f"Running optimization for model: {MODEL_NAME} using {CONTROLLER_TYPE} controller."
        )
        vsr_instance = None
        xml_string = None
        active_coords_list = []

        # --- Prepare VSR and GET XML STRING ---
        try:
            print("Generating/Loading MuJoCo model and VSR info...")
            vsr_instance = VoxelRobot(*VSR_GRID_DIMS, gear=GEAR)
            vsr_instance.load_model_csv(MODEL_CSV_PATH)
            xml_string = vsr_instance.generate_model(
                MODEL_BASE_PATH
            )  # generate_model returns the string

            # load model once here just to print info
            temp_model_for_info = mujoco.MjModel.from_xml_string(xml_string)

            active_coords_list = [
                tuple(c) for c in np.argwhere(vsr_instance.voxel_grid == 1)
            ]  # Get active coords

            print(
                f"MuJoCo XML generated. Sim timestep: {temp_model_for_info.opt.timestep}, Control timestep: {CONTROL_TIMESTEP}"
            )
            print(
                f"VSR Gear: {vsr_instance.gear}, Active Voxels: {len(active_coords_list)}"
            )
            del temp_model_for_info  # Don't need it anymore

        except Exception as e:
            print(f"Failed to load or generate MuJoCo model: {e}")
            print(traceback.format_exc())
            exit()

        # ensure xml_string was obtained
        if xml_string and vsr_instance and active_coords_list:
            # --- Run Optimisation ---
            start_opt_time = time.time()
            best_vector, history_data = optimise(
                xml_string=xml_string,  # pass XML string instead of direct model
                vsr_voxel_coords_list=active_coords_list,
                vsr_gear_ratio=vsr_instance.gear,
                # EA Params
                controller_type=CONTROLLER_TYPE,
                num_generations=NUM_GENERATIONS,
                population_size=POPULATION_SIZE,
                # other params
                tournament_size=TOURNAMENT_SIZE,
                crossover_probability=CROSSOVER_PROBABILITY,
                mutation_sigma=MUTATION_SIGMA,
                clip_range=CLIP_RANGE,
                mlp_plus_hidden_sizes=MLP_PLUS_HIDDEN_SIZES,
                rnn_hidden_size=RNN_HIDDEN_SIZE,
                simulation_duration=SIMULATION_DURATION,
                control_timestep=CONTROL_TIMESTEP,
                num_workers=NUM_WORKERS,
            )

            end_opt_time = time.time()
            print(
                f"\nTotal optimisation time: {end_opt_time - start_opt_time:.2f} seconds"
            )

            # --- Results ---
            if best_vector is not None:
                print("\n--- Best Found Parameter Vector ---")
                # saving logic using best_vector
                config_str = f"{CONTROLLER_TYPE}"
                if CONTROLLER_TYPE == "mlp_plus":
                    config_str += f"_h{'_'.join(map(str, MLP_PLUS_HIDDEN_SIZES))}"
                if CONTROLLER_TYPE == "rnn":
                    config_str += f"_h{RNN_HIDDEN_SIZE}"
                save_path_params = f"{MODEL_BASE_PATH}_best_params_{config_str}_gen{NUM_GENERATIONS}_pop{POPULATION_SIZE}.npy"
                np.save(save_path_params, best_vector)
                print(f"Best parameter vector saved to {save_path_params}")

                save_path_history = f"{MODEL_BASE_PATH}_full_history_{config_str}_gen{NUM_GENERATIONS}_pop{POPULATION_SIZE}.csv"
                try:
                    history_data.to_csv(save_path_history, index=False)
                    print(f"Full history saved to {save_path_history}")
                except Exception as e:
                    print(f"Error saving history CSV: {e}")

                # --- Run final simulation with viewer ---
                print("\nRunning final simulation with best parameters...")
                try:
                    # load the model again for the final run
                    final_run_model = mujoco.MjModel.from_xml_string(xml_string)
                    if final_run_model:
                        final_results = run_simulation(
                            model=final_run_model,  # Use the newly loaded model
                            duration=SIMULATION_DURATION,
                            control_timestep=CONTROL_TIMESTEP,
                            param_vector=best_vector,
                            controller_type=CONTROLLER_TYPE,
                            mlp_plus_hidden_sizes=MLP_PLUS_HIDDEN_SIZES,
                            rnn_hidden_size=RNN_HIDDEN_SIZE,
                            headless=False,
                        )
                        print(f"Final simulation results: {final_results}")
                    else:
                        print("Could not load model for final simulation.")
                except Exception as e:
                    print(f"Error running final simulation: {e}")
                    print(traceback.format_exc())
            else:
                print("\nOptimisation did not yield a valid best result.")
                if not history_data.empty:
                    # save history even if optimisation failed
                    config_str = f"{CONTROLLER_TYPE}"  # add config details
                    save_path_history = f"{MODEL_BASE_PATH}_failed_opt_history_{config_str}_gen{NUM_GENERATIONS}_pop{POPULATION_SIZE}.csv"
                    try:
                        history_data.to_csv(save_path_history, index=False)
                        print(f"Saved history from failed run to {save_path_history}")
                    except Exception as e:
                        print(f"Error saving failed history CSV: {e}")
