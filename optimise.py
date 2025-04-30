# optimization.py

import numpy as np
import pandas as pd
import mujoco
import os
import time
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback # For better error reporting in parallel processes

# Import your existing classes and the headless simulation function
from vsr import VoxelRobot
from controller import DistributedNeuralController
# Ensure this matches your headless simulation file name
from simulate_headless import run_simulation_headless
from simulate import run_simulation

# --- EA Configuration ---
DEFAULT_POPULATION_SIZE = 50 # Smaller for faster testing, paper used 250
DEFAULT_TOURNAMENT_SIZE = 5  # Paper used 8
DEFAULT_CROSSOVER_PROBABILITY = 0.8
DEFAULT_MUTATION_SIGMA = 0.15 # std deviation for mutation
DEFAULT_CLIP_RANGE = (-5.0, 5.0) # Clip parameters to this range
DEFAULT_SIMULATION_DURATION = 60 # seconds
DEFAULT_CONTROL_TIMESTEP = 0.05 # seconds

# --- Helper Functions --- (Keep these as they are)

def _get_vsr_details_and_controller_dims(model):
    """
    Analyzes the VSR model to find active voxels and determine
    controller input/output dimensions.
    """
    try:
        # data = mujoco.MjData(model) # Not strictly needed here if only analyzing model structure

        voxel_motor_map = {}
        voxel_tendon_map = {}

        # Map motors
        for i in range(model.nu):
            motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if motor_name and motor_name.startswith("voxel_"):
                parts = motor_name.split("_")
                try:
                    voxel_coord = tuple(map(int, parts[1:4]))
                    if voxel_coord not in voxel_motor_map:
                        voxel_motor_map[voxel_coord] = []
                    voxel_motor_map[voxel_coord].append(i)
                except (ValueError, IndexError):
                    # print(f"Warning: Could not parse motor name: {motor_name}")
                    continue # Skip malformed names silently during init

        # Map tendons
        for i in range(model.ntendon):
            tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
            if tendon_name and tendon_name.startswith("voxel_"):
                parts = tendon_name.split("_")
                try:
                    voxel_coord = tuple(map(int, parts[1:4]))
                    if voxel_coord not in voxel_tendon_map:
                        voxel_tendon_map[voxel_coord] = []
                    voxel_tendon_map[voxel_coord].append(i)
                except (ValueError, IndexError):
                     # print(f"Warning: Could not parse tendon name: {tendon_name}")
                     continue # Skip malformed names

        # Filter for valid voxels (must have 4 motors and 4 tendons)
        potential_active_coords = sorted(list(set(voxel_motor_map.keys()) | set(voxel_tendon_map.keys())))
        active_voxel_coords = []
        for coord in potential_active_coords:
             # Check if coord exists as a key AND has the correct length list
             has_motors = coord in voxel_motor_map and isinstance(voxel_motor_map[coord], list) and len(voxel_motor_map[coord]) == 4
             has_tendons = coord in voxel_tendon_map and isinstance(voxel_tendon_map[coord], list) and len(voxel_tendon_map[coord]) == 4
             if has_motors and has_tendons:
                 active_voxel_coords.append(coord)
             # else:
             #     # Debugging check (can be verbose)
             #     motor_len = len(voxel_motor_map.get(coord,[])) if isinstance(voxel_motor_map.get(coord), list) else -1
             #     tendon_len = len(voxel_tendon_map.get(coord,[])) if isinstance(voxel_tendon_map.get(coord), list) else -1
             #     print(f"Debug: Voxel {coord} skipped (Motors: {motor_len}, Tendons: {tendon_len})")


        n_active_voxels = len(active_voxel_coords)
        if n_active_voxels == 0:
            raise ValueError("No valid active voxels found in the model. Check MJCF generation and motor/tendon naming (expecting 'voxel_x_y_z_...')")

        print(f"Found {n_active_voxels} active voxels.") # Removed coords list for brevity: {active_voxel_coords}")

        # Determine Controller Dimensions based on the first active voxel (as they are shared)
        # These constants must match those used in DistributedNeuralController and run_simulation!
        N_SENSORS_PER_VOXEL = 8
        N_COMM_CHANNELS = 2
        N_COMM_DIRECTIONS = 6
        N_TIME_INPUTS = 2 # sin(t), cos(t)

        input_size = N_SENSORS_PER_VOXEL + N_COMM_DIRECTIONS * N_COMM_CHANNELS + 1 + N_TIME_INPUTS # +1 for driving signal
        output_size = 1 + N_COMM_DIRECTIONS * N_COMM_CHANNELS # +1 for actuation

        print(f"Controller MLP Input Size: {input_size}")
        print(f"Controller MLP Output Size: {output_size}")

        return n_active_voxels, active_voxel_coords, input_size, output_size

    except Exception as e:
        print(f"Error getting VSR details: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        raise

def flatten_params(weights, biases):
    """Flattens weights and biases into a single vector."""
    return np.concatenate([weights.flatten(), biases.flatten()])

def unflatten_params(param_vector, weight_shape, bias_shape):
    """Unflattens a vector into weights and biases."""
    weights_flat_len = np.prod(weight_shape)
    # Ensure param_vector is a numpy array for slicing
    param_vector = np.asarray(param_vector)
    weights = param_vector[:weights_flat_len].reshape(weight_shape)
    biases = param_vector[weights_flat_len:]
    if biases.shape != bias_shape:
         raise ValueError(f"Bias shape mismatch after unflattening. Expected {bias_shape}, got {biases.shape}. Vector len: {len(param_vector)}, Weight elements: {weights_flat_len}")
    return weights, biases

def evaluate_individual(param_vector, # The parameter vector for this individual
                        # Fixed arguments passed via partial:
                        model,
                        weight_shape,
                        bias_shape,
                        duration,
                        control_timestep):
    """
    Evaluates a single individual (parameter vector) by running the simulation.
    Handles parameter unflattening and simulation execution.
    Designed to be used with multiprocessing.Pool.map.
    """
    try:
        weights, biases = unflatten_params(param_vector, weight_shape, bias_shape)

        fitness, x_dist, y_dist, reached = run_simulation_headless(
            model=model,
            duration=duration,
            control_timestep=control_timestep,
            weights=weights,
            biases=biases
        )
        # Ensure fitness is a float
        if not isinstance(fitness, (float, np.float64)):
             print(f"Warning: Fitness is not float ({type(fitness)}), value: {fitness}. Converting.")
             fitness = float(fitness)
        # Ensure dists are floats
        if not isinstance(x_dist, (float, np.float64)): x_dist = float(x_dist)
        if not isinstance(y_dist, (float, np.float64)): y_dist = float(y_dist)
        # Ensure reached is bool
        if not isinstance(reached, bool): reached = bool(reached)


        # Return fitness, distances, reached status, AND the input parameters
        return fitness, x_dist, y_dist, reached, param_vector

    except mujoco.FatalError as e:
        # Handle MuJoCo crashes specifically if needed (already handled in sim)
        print(f"!!! MuJoCo Fatal Error evaluating individual: {e} !!!")
        return -np.inf, np.inf, np.inf, False, param_vector
    except Exception as e:
        print(f"!!! Error evaluating individual: {e} !!!")
        print(f"Traceback: {traceback.format_exc()}")
        # Return a very poor fitness score and default values
        return -np.inf, np.inf, np.inf, False, param_vector


def tournament_selection(population, fitnesses, tournament_size):
    """Performs tournament selection. Handles potential empty lists."""
    num_individuals = len(population)
    if num_individuals == 0:
        raise ValueError("Cannot perform tournament selection on an empty population.")
    if num_individuals < tournament_size:
        # Warning or adjust tournament size? Adjusting is safer.
        # print(f"Warning: Tournament size ({tournament_size}) > population size ({num_individuals}). Using population size.")
        tournament_size = num_individuals

    # Ensure fitnesses align with population
    if len(fitnesses) != num_individuals:
         raise ValueError("Population and fitnesses list must have the same length.")

    selected_indices = np.random.randint(0, num_individuals, size=tournament_size)
    # Get fitness values corresponding to the selected indices
    tournament_fitnesses = [fitnesses[i] for i in selected_indices]

    # Find the index of the maximum fitness *within the tournament fitnesses list*
    winner_index_in_tournament = np.argmax(tournament_fitnesses)
    # Map this back to the index in the *original population list*
    winner_index = selected_indices[winner_index_in_tournament]

    return population[winner_index]

def geometric_crossover(parent1, parent2):
    """Performs geometric crossover."""
    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)
    if parent1.shape != parent2.shape:
         raise ValueError("Parents must have the same shape for crossover.")
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

# --- Main Optimization Function ---

def optimize(model,
             num_generations: int,
             initial_weights: np.ndarray = None,
             initial_biases: np.ndarray = None,
             population_size: int = DEFAULT_POPULATION_SIZE,
             tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
             crossover_probability: float = DEFAULT_CROSSOVER_PROBABILITY,
             mutation_sigma: float = DEFAULT_MUTATION_SIGMA,
             clip_range: tuple = DEFAULT_CLIP_RANGE,
             simulation_duration: int = DEFAULT_SIMULATION_DURATION,
             control_timestep: float = DEFAULT_CONTROL_TIMESTEP,
             num_workers: int = None):
    """
    Optimizes the VSR controller parameters using a generational EA.

    Args:
        model (mujoco.MjModel): The MuJoCo model object.
        num_generations (int): Number of generations to run the EA.
        initial_weights (np.ndarray, optional): Starting weights for the first individual.
        initial_biases (np.ndarray, optional): Starting biases for the first individual.
        population_size (int): Number of individuals in the population.
        tournament_size (int): Number of individuals participating in each tournament selection.
        crossover_probability (float): Probability of performing crossover.
        mutation_sigma (float): Standard deviation for Gaussian mutation noise.
        clip_range (tuple): (min, max) values for clipping mutated parameters.
        simulation_duration (int): Duration of each simulation evaluation in seconds.
        control_timestep (float): Timestep for controller updates in seconds.
        num_workers (int, optional): Number of parallel workers for evaluation. Defaults to cpu_count().

    Returns:
        tuple: (best_weights, best_biases, history_df)
               - best_weights (np.ndarray): Weights of the best individual found overall.
               - best_biases (np.ndarray): Biases of the best individual found overall.
               - history_df (pd.DataFrame): DataFrame containing performance metrics and parameters
                                            for *every* individual in *each* generation.
    """

    print("--- Starting Optimization ---")
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Using default number of workers: {num_workers}")
    else:
        # Ensure not requesting more workers than available
        num_workers = min(num_workers, cpu_count())
        print(f"Using specified number of workers: {num_workers}")

    # 1. Get VSR details and controller dimensions
    try:
        n_voxels, voxel_coords, input_size, output_size = _get_vsr_details_and_controller_dims(model)
    except Exception as e:
        print(f"Fatal Error: Could not initialize VSR details. Aborting.")
        # No point continuing if we don't know the structure
        return None, None, pd.DataFrame() # Return empty DataFrame

    weight_shape = (output_size, input_size)
    bias_shape = (output_size,)
    param_vector_size = input_size * output_size + output_size
    print(f"Total parameters per individual: {param_vector_size}")

    # 2. Initialize Population
    population = []
    print(f"Initializing population of size {population_size}...")
    start_individual = None
    if initial_weights is not None and initial_biases is not None:
        if initial_weights.shape == weight_shape and initial_biases.shape == bias_shape:
            try:
                start_individual = flatten_params(initial_weights, initial_biases)
                # Check if flattened vector has expected size
                if len(start_individual) == param_vector_size:
                    population.append(start_individual)
                    print("Using provided initial weights/biases for the first individual.")
                else:
                    print(f"Warning: Flattened initial params size ({len(start_individual)}) != expected ({param_vector_size}). Ignoring.")
            except Exception as e:
                print(f"Warning: Error flattening initial params: {e}. Ignoring.")
        else:
            print(f"Warning: Provided initial weights ({initial_weights.shape} vs {weight_shape}) or biases ({initial_biases.shape} vs {bias_shape}) have incorrect shape. Ignoring.")

    # Fill the rest of the population randomly
    while len(population) < population_size:
        # Initialize parameters within a smaller range initially, e.g., U(-1, 1) or clip_range
        random_params = np.random.uniform(clip_range[0]/2, clip_range[1]/2, size=param_vector_size)
        population.append(random_params)

    print("Population initialized.")

    # 3. Evolutionary Loop
    best_fitness_so_far = -np.inf
    best_weights_so_far = None
    best_biases_so_far = None
    # MODIFICATION: Initialize history as a list to store individual dictionaries
    all_history_records = []

    # Create the partial function for evaluation to fix the constant arguments
    evaluate_partial = partial(evaluate_individual,
                               model=model,
                               weight_shape=weight_shape,
                               bias_shape=bias_shape,
                               duration=simulation_duration,
                               control_timestep=control_timestep)

    for generation in range(num_generations):
        gen_start_time = time.time()
        print(f"\n--- Generation {generation + 1}/{num_generations} ---")

        # 4. Evaluate Population
        print(f"Evaluating {len(population)} individuals using {num_workers} workers...")
        eval_start_time = time.time()

        results = []
        current_population_to_eval = population # Keep track of the population being evaluated

        # Use multiprocessing pool for parallel evaluation
        try:
            # Make sure the pool uses the desired number of workers
            actual_workers = min(num_workers, len(current_population_to_eval)) # Don't need more workers than individuals
            if actual_workers < num_workers:
                 # print(f"Note: Using {actual_workers} workers (population size limit).")
                 pass # Reduce verbosity
            if actual_workers <= 0:
                 print("Warning: No individuals to evaluate.")
                 continue # Skip to next generation or handle error

            with Pool(processes=actual_workers) as pool:
                 # results will be a list of tuples:
                 # [(fitness, x, y, reached, params), (fitness, x, y, reached, params), ...]
                 results = pool.map(evaluate_partial, current_population_to_eval)

        except Exception as e:
             print(f"!!! Fatal Error during parallel evaluation: {e} !!!")
             print(f"Traceback: {traceback.format_exc()}")
             # Attempt to salvage results if some processes finished
             if not results: # If results is empty, we can't continue
                 print("Aborting optimization due to evaluation error.")
                 history_df = pd.DataFrame(all_history_records) # Use records collected so far
                 return best_weights_so_far, best_biases_so_far, history_df


        eval_time = time.time() - eval_start_time
        print(f"Evaluation finished in {eval_time:.2f} seconds.")

        # Process results & *** MODIFIED HISTORY LOGGING ***
        fitnesses = []
        evaluated_population_params = [] # Store params corresponding to fitnesses

        gen_best_fitness = -np.inf
        gen_best_params = None

        num_failed = 0
        for i, res in enumerate(results):
            fitness, x_dist, y_dist, reached, params = res

            # Store results for selection/stats
            fitnesses.append(fitness)
            evaluated_population_params.append(params) # Store the params vector evaluated

            # Check for evaluation failure
            if fitness <= -np.inf:
                num_failed += 1
                # Skip logging failed individuals in history or log with bad values?
                # Let's log them so we know they failed.
                valid_eval = False
            else:
                valid_eval = True
                # Update generation's best
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_params = params # Store the params of the best in this gen

            # *** ADD RECORD FOR EACH INDIVIDUAL TO HISTORY ***
            record = {
                'generation': generation + 1,
                'individual_index': i, # Index within the generation's population
                'fitness': fitness,
                'x_dist': x_dist,
                'y_dist': y_dist,
                'reached': reached,
                # Store params as a list (more compatible with CSV/JSON than ndarray)
                'params_vector': list(params) if params is not None else None
            }
            all_history_records.append(record)

        if num_failed > 0:
            print(f"Warning: {num_failed}/{len(results)} evaluations failed in this generation.")

        # Calculate stats based on *valid* evaluations
        valid_fitnesses = [f for f in fitnesses if f > -np.inf]
        avg_gen_fitness = np.mean(valid_fitnesses) if valid_fitnesses else -np.inf

        print(f"Best fitness in generation: {gen_best_fitness:.4f}")
        print(f"Average fitness (valid):   {avg_gen_fitness:.4f}")

        # Update overall best (using the best found in this generation)
        if gen_best_fitness > best_fitness_so_far:
            best_fitness_so_far = gen_best_fitness
            # Unflatten the parameters of the generation's best individual
            if gen_best_params is not None:
                try:
                    best_weights_so_far, best_biases_so_far = unflatten_params(gen_best_params, weight_shape, bias_shape)
                    print(f"*** New overall best fitness found: {best_fitness_so_far:.4f} ***")
                except ValueError as e:
                    print(f"Error unflattening best params for overall best: {e}")
                    # Keep previous best if unflattening fails
            else:
                 print("Warning: Generation best fitness improved, but params were None.")


        # 5. Selection (Choose parents for the next generation)
        parents = []
        # Select from the parameters of the individuals that were *successfully* evaluated
        valid_population_params = [p for p, f in zip(evaluated_population_params, fitnesses) if f > -np.inf]
        valid_pop_fitnesses = [f for f in fitnesses if f > -np.inf]

        if not valid_population_params:
             print("Warning: No valid individuals left for selection! Re-initializing population randomly.")
             # Fallback: Create a completely new random population
             population = [np.random.uniform(clip_range[0], clip_range[1], size=param_vector_size) for _ in range(population_size)]
             continue # Skip variation and proceed to next generation's evaluation

        # Proceed with selection using valid individuals
        for _ in range(population_size): # Need population_size parents for replacement strategy
             selected_parent = tournament_selection(valid_population_params, valid_pop_fitnesses, tournament_size)
             parents.append(selected_parent)


        # 6. Variation (Create offspring using Crossover and Mutation)
        offspring_population = []
        # Shuffle parent indices to mix pairs for crossover
        parent_indices = np.random.permutation(len(parents))

        for i in range(0, population_size): # Create one offspring per parent slot
            # Select parent(s) - simple strategy: use parent[i] for mutation basis,
            # and pair parent[i] with parent[j] for crossover.
            idx1 = parent_indices[i]
            # For crossover, pick a second distinct parent randomly or sequentially
            idx2 = parent_indices[(i + np.random.randint(1, len(parents))) % len(parents)] # Ensure different index
            if idx1 == idx2 and len(parents) > 1: # Handle edge case if only 1 parent survived
                 idx2 = parent_indices[(idx1 + 1) % len(parents)]

            parent1 = parents[idx1]
            parent2 = parents[idx2]

            # Decide Crossover or Mutation (or both?) - Paper implies one OR the other.
            if np.random.rand() < crossover_probability and len(parents) > 1:
                 # Crossover
                 offspring = geometric_crossover(parent1, parent2)
                 # Optionally add a small mutation after crossover
                 # if np.random.rand() < 0.1: # Small chance
                 #    offspring = gaussian_mutation(offspring, mutation_sigma / 5, clip_range) # Smaller mutation
            else:
                 # Mutation - mutate parent1
                 offspring = gaussian_mutation(parent1, mutation_sigma, clip_range)

            offspring_population.append(offspring)

        # Ensure the new population has the correct size (should match population_size)
        population = offspring_population[:population_size]


        # --- Generation End ---
        gen_time = time.time() - gen_start_time
        print(f"Generation {generation + 1} finished in {gen_time:.2f} seconds.")
        # Optional: Flush print buffer if running in certain environments
        # sys.stdout.flush()

    # --- End of Optimization ---
    print("\n--- Optimization Finished ---")

    # Create DataFrame from the collected records
    history_df = pd.DataFrame(all_history_records)

    if best_weights_so_far is None:
        print("Warning: No best individual found (optimization might have failed early or found no valid solutions).")
        # Return None and the history collected so far
        return None, None, history_df
    else:
         print(f"Overall best fitness achieved: {best_fitness_so_far:.4f}")
         print("Detailed history DataFrame created.")
         # best_weights_so_far and best_biases_so_far hold the parameters
         return best_weights_so_far, best_biases_so_far, history_df

# --- Example Usage --- (Mostly unchanged, but uses the modified optimize)
if __name__ == "__main__":
    # Ensure the model files exist at this path
    MODEL_NAME = "quadruped_v3" # Or your specific model name
    MODEL_BASE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}"
    MODEL_CSV_PATH = MODEL_BASE_PATH + ".csv"
    MODEL_XML_PATH = MODEL_BASE_PATH + "_modded.xml" # Assuming vsr.py saves the final XML here


    if not os.path.exists(MODEL_CSV_PATH):
         print(f"Error: Model CSV file not found at {MODEL_CSV_PATH}")
    # elif not os.path.exists(MODEL_XML_PATH):
    #      print(f"Error: Model XML file not found at {MODEL_XML_PATH}. Generate it first.")
    else:
        print(f"Running optimization for model: {MODEL_NAME}")

        # --- Prepare Model (Run this once before optimize) ---
        try:
            print("Generating/Loading MuJoCo model...")
            vsr = VoxelRobot(10, 10, 10) # Adjust size if needed
            vsr.load_model_csv(MODEL_CSV_PATH)
            xml_string = vsr.generate_model(MODEL_BASE_PATH) # Generates _modded.xml
            model = mujoco.MjModel.from_xml_string(xml_string)
            print("MuJoCo model loaded successfully.")
        except Exception as e:
            print(f"Failed to load or generate MuJoCo model: {e}")
            print(traceback.format_exc())
            model = None # Ensure model is None if loading failed

        if model:
            # --- Run Optimization ---
            num_generations_to_run = 10 # Increase for real runs (e.g., 100, 500)
            population_s = 14 # Keep low for testing

            start_opt_time = time.time()
            best_w, best_b, history_data = optimize(
                model=model, # Pass the loaded model object
                num_generations=num_generations_to_run,
                population_size=population_s,
                tournament_size=4, # Keep small for small pop
                mutation_sigma=0.2, # Adjust if needed
                control_timestep=0.2, # Match simulation timestep if reasonable
                num_workers=7
            )
            end_opt_time = time.time()
            print(f"\nTotal optimization time: {end_opt_time - start_opt_time:.2f} seconds")

            # --- Results ---
            if best_w is not None and best_b is not None:
                print("\n--- Best Found Parameters ---")
                print("Weights shape:", best_w.shape)
                print("Biases shape:", best_b.shape)

                # Save the best parameters
                save_path_params = f"{MODEL_BASE_PATH}_best_params_gen{num_generations_to_run}_pop{population_s}.npz"
                np.savez(save_path_params, weights=best_w, biases=best_b)
                print(f"Best parameters saved to {save_path_params}")

                print("\n--- Optimization History (Last 5 records) ---")
                print(history_data.tail().to_string()) # Show tail for large history
                # Save full history
                save_path_history = f"{MODEL_BASE_PATH}_full_history_gen{num_generations_to_run}_pop{population_s}.csv"
                try:
                    history_data.to_csv(save_path_history, index=False)
                    print(f"Full optimization history saved to {save_path_history}")
                except Exception as e:
                    print(f"Error saving history CSV: {e}")

                # Optional: Run final simulation with viewer
                # print("\nRunning final simulation with best parameters (requires vsr_simulate.py)...")
                try:
                    # from vsr_simulate import run_simulation # Import the viewer version
                    # # Need to pass model_path, not model object to run_simulation
                    run_simulation(model, duration=60, control_timestep=0.2, weights=best_w, biases=best_b)
                except ImportError:
                    print("Could not import run_simulation from vsr_simulate.py to show final result.")
                except Exception as e:
                     print(f"Error running final visualization: {e}")


            else:
                print("\nOptimization did not yield a valid best result.")
                if not history_data.empty:
                     save_path_history = f"{MODEL_BASE_PATH}_failed_opt_history_gen{num_generations_to_run}_pop{population_s}.csv"
                     history_data.to_csv(save_path_history, index=False)
                     print(f"Saved history from failed optimization run to {save_path_history}")