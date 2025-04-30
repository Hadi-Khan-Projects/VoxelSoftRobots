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
from simulate_headless import run_simulation_headless # Assuming this file exists
from simulate import run_simulation

# --- EA Configuration ---
DEFAULT_POPULATION_SIZE = 50 # Smaller for faster testing, paper used 250
DEFAULT_TOURNAMENT_SIZE = 5  # Paper used 8
DEFAULT_CROSSOVER_PROBABILITY = 0.8
DEFAULT_MUTATION_SIGMA = 0.15 # std deviation for mutation
DEFAULT_CLIP_RANGE = (-5.0, 5.0) # Clip parameters to this range
DEFAULT_SIMULATION_DURATION = 60 # seconds
DEFAULT_CONTROL_TIMESTEP = 0.05 # seconds

# --- Helper Functions ---

def _get_vsr_details_and_controller_dims(model):
    """
    Analyzes the VSR model to find active voxels and determine
    controller input/output dimensions.
    """
    try:
        data = mujoco.MjData(model) # Needed for some MuJoCo functions if used later

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
                    print(f"Warning: Could not parse motor name: {motor_name}")
                    continue

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
                     print(f"Warning: Could not parse tendon name: {tendon_name}")
                     continue

        # Filter for valid voxels (must have 4 motors and 4 tendons)
        potential_active_coords = sorted(list(set(voxel_motor_map.keys()) | set(voxel_tendon_map.keys())))
        active_voxel_coords = []
        for coord in potential_active_coords:
             has_motors = coord in voxel_motor_map and len(voxel_motor_map[coord]) == 4
             has_tendons = coord in voxel_tendon_map and len(voxel_tendon_map[coord]) == 4
             if has_motors and has_tendons:
                 active_voxel_coords.append(coord)
             # else:
             #     print(f"Debug: Voxel {coord} skipped (Motors: {len(voxel_motor_map.get(coord,[]))}, Tendons: {len(voxel_tendon_map.get(coord,[]))})")


        n_active_voxels = len(active_voxel_coords)
        if n_active_voxels == 0:
            raise ValueError("No valid active voxels found in the model. Check MJCF generation.")

        print(f"Found {n_active_voxels} active voxels: {active_voxel_coords}")

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
        raise

def flatten_params(weights, biases):
    """Flattens weights and biases into a single vector."""
    return np.concatenate([weights.flatten(), biases.flatten()])

def unflatten_params(param_vector, weight_shape, bias_shape):
    """Unflattens a vector into weights and biases."""
    weights_flat_len = np.prod(weight_shape)
    weights = param_vector[:weights_flat_len].reshape(weight_shape)
    biases = param_vector[weights_flat_len:]
    if biases.shape != bias_shape:
         # This check is important after operations like crossover/mutation
         # Might need padding or truncation if lengths get messed up,
         # but ideally operators preserve length.
         raise ValueError(f"Bias shape mismatch after unflattening. Expected {bias_shape}, got {biases.shape}")
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

        # Assuming run_simulation_headless takes weights and biases directly
        fitness, x_dist, y_dist, reached = run_simulation_headless(
            model=model,
            duration=duration,
            control_timestep=control_timestep,
            weights=weights,
            biases=biases
            # NOTE: run_simulation_headless needs to internally get n_voxels, coords etc.
            # OR you modify it to accept these + controller dims if that's cleaner
        )
        # Ensure fitness is a float
        if not isinstance(fitness, (float, np.float64)):
             print(f"Warning: Fitness is not float ({type(fitness)}), value: {fitness}. Converting.")
             fitness = float(fitness)

        return fitness, x_dist, y_dist, reached, param_vector # Return params for tracking

    except Exception as e:
        print(f"!!! Error evaluating individual: {e} !!!")
        print(f"Traceback: {traceback.format_exc()}")
        # Return a very poor fitness score and default values
        # Ensure the number of return values matches the expected tuple
        return -np.inf, np.inf, np.inf, False, param_vector


def tournament_selection(population, fitnesses, tournament_size):
    """Performs tournament selection."""
    num_individuals = len(population)
    selected_indices = np.random.randint(0, num_individuals, size=tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in selected_indices]
    winner_index_in_tournament = np.argmax(tournament_fitnesses)
    winner_index = selected_indices[winner_index_in_tournament]
    return population[winner_index]

def geometric_crossover(parent1, parent2):
    """Performs geometric crossover."""
    if len(parent1) != len(parent2):
         raise ValueError("Parents must have the same length for crossover.")
    beta = np.random.uniform(-1.0, 2.0, size=len(parent1))
    offspring = parent1 + beta * (parent2 - parent1)
    return offspring

def gaussian_mutation(individual, sigma, clip_range):
    """Performs Gaussian mutation and clips the result."""
    noise = np.random.normal(0, sigma, size=len(individual))
    mutated = individual + noise
    # Clip the parameters to the defined range
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
        model ():
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
        num_workers (int, optional): Number of parallel workers for evaluation.
                                     Defaults to cpu_count().

    Returns:
        tuple: (best_weights, best_biases, history_df)
               - best_weights (np.ndarray): Weights of the best individual found.
               - best_biases (np.ndarray): Biases of the best individual found.
               - history_df (pd.DataFrame): DataFrame containing performance metrics
                                            for the best individual of each generation.
    """

    print("--- Starting Optimization ---")
    if num_workers is None:
        num_workers = cpu_count()
        print(f"Using default number of workers: {num_workers}")
    else:
        num_workers = min(num_workers, cpu_count())
        print(f"Using specified number of workers: {num_workers}")


    # 1. Get VSR details and controller dimensions
    try:
        n_voxels, voxel_coords, input_size, output_size = _get_vsr_details_and_controller_dims(model)
    except Exception as e:
        print(f"Fatal Error: Could not initialize VSR details. Aborting.")
        print(e)
        return None, None, None

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
            start_individual = flatten_params(initial_weights, initial_biases)
            population.append(start_individual)
            print("Using provided initial weights/biases for the first individual.")
        else:
            print("Warning: Provided initial weights/biases have incorrect shape. Ignoring.")

    # Fill the rest of the population randomly
    while len(population) < population_size:
        # Initialize parameters within a smaller range initially, e.g., U(-1, 1)
        random_params = np.random.uniform(-1.0, 1.0, size=param_vector_size)
        population.append(random_params)

    print("Population initialized.")

    # 3. Evolutionary Loop
    best_fitness_so_far = -np.inf
    best_weights_so_far = None
    best_biases_so_far = None
    history = []

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
        # Use multiprocessing pool for parallel evaluation
        try:
            with Pool(processes=num_workers) as pool:
                 # results will be a list of tuples:
                 # [(fitness, x, y, reached, params), (fitness, x, y, reached, params), ...]
                 results = pool.map(evaluate_partial, population)

        except Exception as e:
             print(f"!!! Fatal Error during parallel evaluation: {e} !!!")
             print(f"Traceback: {traceback.format_exc()}")
             # Attempt to salvage results if some processes finished
             if not results: # If results is empty, we can't continue
                 print("Aborting optimization due to evaluation error.")
                 # Return the best found so far, even if it's from a previous gen
                 history_df = pd.DataFrame(history)
                 return best_weights_so_far, best_biases_so_far, history_df


        eval_time = time.time() - eval_start_time
        print(f"Evaluation finished in {eval_time:.2f} seconds.")

        # Process results
        fitnesses = [r[0] for r in results]
        x_dists = [r[1] for r in results]
        y_dists = [r[2] for r in results]
        reached_flags = [r[3] for r in results]
        evaluated_population = [r[4] for r in results] # Get params back in case order changed

        # Check for evaluation failures indicated by -inf fitness
        valid_evals = [f > -np.inf for f in fitnesses]
        num_failed = population_size - sum(valid_evals)
        if num_failed > 0:
            print(f"Warning: {num_failed}/{population_size} evaluations failed in this generation.")
            # Consider strategies: retry failed, replace with random, or just proceed with valid ones.
            # For simplicity, we proceed, but this might bias selection.

        # Find best in current generation
        best_gen_idx = np.argmax(fitnesses)
        best_gen_fitness = fitnesses[best_gen_idx]
        best_gen_params = evaluated_population[best_gen_idx]
        best_gen_x_dist = x_dists[best_gen_idx]
        best_gen_y_dist = y_dists[best_gen_idx]
        best_gen_reached = reached_flags[best_gen_idx]
        avg_gen_fitness = np.mean([f for f, v in zip(fitnesses, valid_evals) if v]) if any(valid_evals) else -np.inf

        print(f"Best fitness in generation: {best_gen_fitness:.4f}")
        print(f"Average fitness (valid):   {avg_gen_fitness:.4f}")

        # Update overall best
        if best_gen_fitness > best_fitness_so_far:
            best_fitness_so_far = best_gen_fitness
            best_weights_so_far, best_biases_so_far = unflatten_params(best_gen_params, weight_shape, bias_shape)
            print(f"*** New overall best fitness found: {best_fitness_so_far:.4f} ***")

        # Log history for this generation (using the generation's best)
        history.append({
            'generation': generation + 1,
            'best_fitness': best_gen_fitness,
            'average_fitness': avg_gen_fitness,
            'best_x_dist': best_gen_x_dist,
            'best_y_dist': best_gen_y_dist,
            'best_reached': best_gen_reached,
        })

        # 5. Selection (Choose parents for the next generation)
        parents = []
        for _ in range(population_size): # Need enough parents to create population_size offspring
             # Ensure selection only happens among successfully evaluated individuals
             valid_population = [p for p, v in zip(evaluated_population, valid_evals) if v]
             valid_fitnesses = [f for f, v in zip(fitnesses, valid_evals) if v]
             if not valid_population:
                  print("Warning: No valid individuals left for selection. Re-initializing random parents.")
                  # As a fallback, generate random parents, otherwise selection fails
                  parents = [np.random.uniform(clip_range[0], clip_range[1], size=param_vector_size) for _ in range(population_size)]
                  break # Exit the parent selection loop

             selected_parent = tournament_selection(valid_population, valid_fitnesses, tournament_size)
             parents.append(selected_parent)


        # 6. Variation (Create offspring using Crossover and Mutation)
        offspring_population = []
        parent_indices = np.random.permutation(population_size) # Shuffle indices

        for i in range(0, population_size, 2):
            # Ensure we have pairs of parents
            idx1 = parent_indices[i]
            # Wrap around if odd number of parents needed (or handle explicitly)
            idx2 = parent_indices[(i + 1) % population_size]
            parent1 = parents[idx1]
            parent2 = parents[idx2]

            # Crossover
            if np.random.rand() < crossover_probability:
                offspring1 = geometric_crossover(parent1, parent2)
                offspring2 = geometric_crossover(parent2, parent1) # Can generate two distinct offspring
            else:
                offspring1 = parent1.copy()
                offspring2 = parent2.copy()

            # Mutation
            offspring1 = gaussian_mutation(offspring1, mutation_sigma, clip_range)
            offspring2 = gaussian_mutation(offspring2, mutation_sigma, clip_range)

            offspring_population.append(offspring1)
            if len(offspring_population) < population_size: # Avoid adding extra if pop_size is odd
                offspring_population.append(offspring2)

        # Ensure the new population has the correct size
        population = offspring_population[:population_size]


        # --- Generation End ---
        gen_time = time.time() - gen_start_time
        print(f"Generation {generation + 1} finished in {gen_time:.2f} seconds.")

    # --- End of Optimization ---
    print("\n--- Optimization Finished ---")
    if best_weights_so_far is None:
        print("Warning: No best individual found (optimization might have failed early).")
        # Return random params or None? Returning None seems safer.
        return None, None, pd.DataFrame(history)
    else:
         print(f"Overall best fitness achieved: {best_fitness_so_far:.4f}")
         history_df = pd.DataFrame(history)
         print("History DataFrame created.")
         return best_weights_so_far, best_biases_so_far, history_df

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure the model files exist at this path
    MODEL_NAME = "quadruped_v2" # Or your specific model name
    MODEL_BASE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}"

    if not os.path.exists(MODEL_BASE_PATH + ".csv"):
         print(f"Error: Model CSV file not found at {MODEL_BASE_PATH}.csv")
    else:
        print(f"Running optimization for model: {MODEL_NAME}")

        # --- Run Optimization ---
        num_generations_to_run = 10 # Increase for real runs (e.g., 100, 500)
        population_s = 12 # Keep low for testing

        vsr = VoxelRobot(10, 10, 10)  # Adjust size if needed for your model
        vsr.load_model_csv(MODEL_BASE_PATH + ".csv")
        # vsr.visualise_model()
        xml_string = vsr.generate_model(MODEL_BASE_PATH)

        model = mujoco.MjModel.from_xml_string(xml_string)

        best_w, best_b, history_data = optimize(
            model=model,
            num_generations=num_generations_to_run,
            population_size=population_s, # Use smaller pop size for quick test
            num_workers=12 # Limit workers for testing if needed
            # Pass initial_weights/biases here if you have them
        )

        # --- Results ---
        if best_w is not None:
            print("\n--- Best Found Parameters ---")
            print("Weights shape:", best_w.shape)
            print("Biases shape:", best_b.shape)
            # Optionally save the best parameters
            # np.savez(f"{MODEL_BASE_PATH}_best_params.npz", weights=best_w, biases=best_b)
            # print(f"Best parameters saved to {MODEL_BASE_PATH}_best_params.npz")

            print("\n--- Optimization History ---")
            print(history_data.to_string())
            # Optionally save history
            # history_data.to_csv(f"{MODEL_BASE_PATH}_optimization_history.csv", index=False)
            # print(f"Optimization history saved to {MODEL_BASE_PATH}_optimization_history.csv")

            # You could now run the simulation *with* the viewer using these best params:
            # from vsr_simulate import run_simulation # Import the viewer version
            print("\nRunning final simulation with best parameters...")
            run_simulation(MODEL_BASE_PATH, duration=60, control_timestep=0.05, weights=best_w, biases=best_b)

        else:
            print("Optimization did not yield a best result.")