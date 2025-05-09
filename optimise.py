import argparse
import os
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count

import mujoco
import numpy as np
import pandas as pd

from controller import DistributedNeuralController
from simulate import run_simulation, voxel_motor_mapping
from vsr import VoxelRobot


def _get_vsr_details_and_base_dims(
    model: mujoco.MjModel,
) -> tuple[int, list[tuple[int, int, int]], int, int]:
    """
    Analyzes the VSR model to find active voxels and determine
    *base* controller input/output dimensions (independent of network type).

    Args:
        model: mujoco.MjModel instance of the VSR model.

    Returns:
        n_active_voxels: Number of active voxels in the model.
        active_voxel_coords: List of tuples representing the coordinates of active voxels.
        base_input_size: Size of the base input vector for the controller.
        base_output_size: Size of the base output vector for the controller.
    """
    try:
        active_voxel_coords, _, _ = voxel_motor_mapping(model)
        n_active_voxels = len(active_voxel_coords)

        # determine controller BASE Dimensions (must match controller.py)
        N_SENSORS_PER_VOXEL = 8
        N_COMM_CHANNELS = 2
        N_COMM_DIRECTIONS = 6
        N_TIME_INPUTS = 2
        N_COM_VEL_INPUTS = 3
        N_TARGET_ORIENT_INPUTS = 2

        base_input_size = (
            N_SENSORS_PER_VOXEL
            + N_COMM_DIRECTIONS * N_COMM_CHANNELS
            + 1  # driving signal
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
    """
    Calculates the total number of parameters for a given controller configuration.

    Args:
        controller_type: Type of controller ('mlp', 'mlp_plus', 'rnn').
        base_input_size: Size of the base input vector.
        base_output_size: Size of the base output vector.
        mlp_plus_hidden_sizes: List of hidden layer sizes for MLP+.
        rnn_hidden_size: Hidden size for RNN.

    Returns:
        total_param_count: Total number of parameters in the controller.
    """
    # instantiating a temporary controller just to use its setup logic
    # feed dummy voxel info, actual values don't matter for size calculation
    temp_controller = DistributedNeuralController(
        controller_type=controller_type,
        n_voxels=1,
        voxel_coords=[(0, 0, 0)],
        n_sensors_per_voxel=8,  # value doesn't impact structure size logic directly here
        n_comm_channels=2,  # value doesn't impact structure size logic directly here
        # pass the actual configurations
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
    # arguments passed via partial
    xml_string,  # pass XML string, not direct model
    duration,
    control_timestep,
    # controller Config
    controller_type,
    mlp_plus_hidden_sizes,
    rnn_hidden_size,
    # VSR/sim metadata for logging
    voxel_coords_list_str,
    simulation_timestep,
    gear_ratio,
):
    """
    Evaluates a single individual. Loads the model from XML within the worker.

    Args:
        param_vector: Parameter vector for the individual.
        xml_string: XML string of the model.
        duration: Duration of the simulation.
        control_timestep: Control timestep for the simulation.
        controller_type: Type of controller to use.
        mlp_plus_hidden_sizes: Hidden sizes for MLP+ controller.
        rnn_hidden_size: Hidden size for RNN controller.
        voxel_coords_list_str: String representation of voxel coordinates list.
        simulation_timestep: Expected simulation timestep.
        gear_ratio: Gear ratio for the simulation.

    Returns:
        fitness: Fitness score of the individual.
        x_dist: X distance travelled.
        y_dist: Y distance travelled.
        reached: Boolean indicating if the target was reached.
        param_vector: Parameter vector for the individual.
        controller_type: Type of controller used.
        mlp_plus_hidden_sizes: Hidden sizes for MLP+ controller.
        rnn_hidden_size: Hidden size for RNN controller.
        voxel_coords_list_str: String representation of voxel coordinates list.
        control_timestep: Control timestep used.
        simulation_timestep: Actual simulation timestep used.
        gear_ratio: Gear ratio used.
    """
    model = None  # initialize model to None
    try:
        # load Model INSIDE the worker
        model = mujoco.MjModel.from_xml_string(xml_string)

        # run simulation
        fitness, x_dist, y_dist, reached = run_simulation(
            model=model,  # pass locally loaded model
            duration=duration,
            control_timestep=control_timestep,
            param_vector=param_vector,
            # controller configuration
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
            gear_ratio,  # log actual timestep used
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
    """
    Performs tournament selection. Handles potential empty lists.

    Args:
        population: List of individuals.
        fitnesses: List of fitness scores corresponding to the individuals.
        tournament_size: Size of the tournament.

    Returns:
        Selected individual from the population.
    """
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
    """
    Performs geometric crossover.

    Args:
        parent1: First parent individual.
        parent2: Second parent individual.

    Returns:
        Offspring individual resulting from the crossover.
    """
    parent1 = np.asarray(parent1)
    parent2 = np.asarray(parent2)
    if parent1.shape != parent2.shape:
        raise ValueError("Parent shapes mismatch.")
    beta = np.random.uniform(-1.0, 2.0, size=parent1.shape)
    offspring = parent1 + beta * (parent2 - parent1)
    return offspring


def gaussian_mutation(individual, sigma, clip_range):
    """
    Performs Gaussian mutation and clips the result.

    Args:
        individual: Individual to mutate.
        sigma: Standard deviation for the Gaussian noise.
        clip_range: Tuple specifying the clipping range.

    Returns:
        Mutated individual.
    """
    individual = np.asarray(individual)
    noise = np.random.normal(0, sigma, size=individual.shape)
    mutated = individual + noise
    mutated = np.clip(mutated, clip_range[0], clip_range[1])
    return mutated


def optimise(
    xml_string: str,  # pass XML string, not direct model
    vsr_voxel_coords_list,  # the actual list of (x,y,z) tuples
    vsr_gear_ratio,
    # EA Params
    controller_type: str,
    num_generations: int,
    population_size: int,
    tournament_size: int,
    crossover_probability: float,
    mutation_sigma: float,
    clip_range: tuple,
    # controller Config
    mlp_plus_hidden_sizes: list,
    rnn_hidden_size: int,
    # simulation Params
    simulation_duration: int,
    control_timestep: float,
    num_workers: int,
    initial_param_vector: np.ndarray = None,
):
    """
    Optimises the VSR controller parameters using a generational EA.
    Loads the model from xml_string within each worker process.
    """
    # load model once in main process ONLY for setup analysis
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

    # STEP 1: Get VSR details using the setup_model
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

    # STEP 2: Calculate Parameter Vector Size
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

    # STEP 3: Prepare simulation Metadata
    simulation_timestep = (
        setup_model.opt.timestep
    )  # get expected timestep from setup model
    voxel_coords_list_str = str(vsr_voxel_coords_list)
    gear_ratio = vsr_gear_ratio
    del setup_model  # free the setup model, workers will load their own

    # STEP 4: Initialise population
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

    # STEP 5: Evolutionary loop setup
    best_fitness_so_far = -np.inf
    best_param_vector_so_far = None
    all_history_records = []

    # create partial function
    evaluate_partial = partial(
        evaluate_individual,
        xml_string=xml_string,  # pass XML string, not direct model
        duration=simulation_duration,
        control_timestep=control_timestep,
        # controller Config
        controller_type=controller_type,
        mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        # metadata
        voxel_coords_list_str=voxel_coords_list_str,
        simulation_timestep=simulation_timestep,
        gear_ratio=gear_ratio,
    )
    print("\n")

    # STEP 6: Evolutionary loop
    for generation in range(num_generations):
        gen_start_time = time.time()
        print(f"\n--- Generation {generation + 1}/{num_generations} ---")

        # STEP 7: Evaluate population (Pool map call)
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
            )  # return current best and history

        eval_time = time.time() - eval_start_time
        print(f"Evaluation finished in {eval_time:.2f} seconds.")

        # STEP 8: process results
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

        # update overall best
        if gen_best_fitness > best_fitness_so_far:
            if gen_best_params is not None:
                best_fitness_so_far = gen_best_fitness
                best_param_vector_so_far = np.copy(gen_best_params)
                print(
                    f"*** New overall best fitness found: {best_fitness_so_far:.4f} ***"
                )
            else:
                print("Warning: Gen best fitness improved, but params vector was None.")

        # STEP 9. selection
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

        # STEP 10. variation
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

    # end of optimisation
    print("\n--- Optimisation Finished ---")
    history_df = pd.DataFrame(all_history_records)
    if best_param_vector_so_far is None:
        print("Warning: No best individual found.")
    else:
        print(f"Overall best fitness achieved: {best_fitness_so_far:.4f}")
    return best_param_vector_so_far, history_df


# cli params:

# python optimise.py \
#     --model_name <name_of_your_model> \
#     --controller_type <mlp|mlp_plus|rnn> \
#     --gear <gear_ratio_float> \
#     [--num_generations <int>] \
#     [--population_size <int>] \
#     [--tournament_size <int>] \
#     [--crossover_probability <float_0_to_1>] \
#     [--mutation_sigma <float>] \
#     [--clip_range_min <float>] [--clip_range_max <float>] \
#     [--mlp_plus_hidden_sizes <size1_int> <size2_int> ...] \
#     [--rnn_hidden_size <size_int>] \
#     [--simulation_duration <seconds_int>] \
#     [--control_timestep <seconds_float>] \
#     [--num_workers <int>] \
#     [--initial_param_vector_path <path_to_seed_params.npy>] \
#     [--output_dir <path_to_save_results>]

# cli usage example:

# python optimise.py \                                                                                                                                                                                              --model_name quadruped_v3 \
#     --controller_type rnn \
#     --gear 100.0 \
#     --num_generations 2 \
#     --population_size 8 \
#     --tournament_size 2 \
#     --mutation_sigma 0.2 \
#     --clip_range_min -4.0 --clip_range_max 4.0 \
#     --rnn_hidden_size 16 \
#     --simulation_duration 60 \
#     --control_timestep 0.2 \
#     --num_workers 8 \
#     --output_dir results_optimise/my_rnn_run_1

# can use via cli, actual usage in evolve.py
if __name__ == "__main__":
    # STEP 1: Get arguments/paremeters
    parser = argparse.ArgumentParser(
        description="Optimise VSR controller parameters using EA."
    )

    # required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the VSR model (e.g., 'quadruped_v3'). Assumes model files are in 'vsr_models/<model_name>/'.",
    )
    parser.add_argument(
        "--controller_type",
        type=str,
        required=True,
        choices=["mlp", "mlp_plus", "rnn"],
        help="Type of distributed controller.",
    )
    parser.add_argument(
        "--gear", type=float, required=True, help="Gear ratio for the VSR motors."
    )

    # EA parameters
    parser.add_argument(
        "--num_generations",
        type=int,
        default=20,
        help="Number of generations for the EA.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=60,
        help="Population size for the EA.",
    )
    parser.add_argument(
        "--tournament_size",
        type=int,
        default=6,
        help="Tournament size for parent selection.",
    )
    parser.add_argument(
        "--crossover_probability",
        type=float,
        default=0.8,
        help="Probability of performing crossover.",
    )
    parser.add_argument(
        "--mutation_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for Gaussian mutation.",
    )
    parser.add_argument(
        "--clip_range_min",
        type=float,
        default=-5.0,
        help="Minimum value for parameter clipping.",
    )
    parser.add_argument(
        "--clip_range_max",
        type=float,
        default=5.0,
        help="Maximum value for parameter clipping.",
    )

    # controller configuration (optional)
    parser.add_argument(
        "--mlp_plus_hidden_sizes",
        type=int,
        nargs="+",
        default=[],
        help="List of hidden layer sizes for mlp_plus controller.",
    )
    parser.add_argument(
        "--rnn_hidden_size",
        type=int,
        default=0,
        help="Hidden state size for rnn controller.",
    )

    # simulation params
    parser.add_argument(
        "--simulation_duration",
        type=int,
        default=60,
        help="Duration of each simulation evaluation in seconds.",
    )
    parser.add_argument(
        "--control_timestep",
        type=float,
        default=0.2,
        help="Time step for control updates in seconds.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),  # Default to max available
        help="Number of parallel workers for evaluation.",
    )

    # optional arguments
    parser.add_argument(
        "--initial_param_vector_path",
        type=str,
        default=None,
        help="Path to a .npy file to seed the initial population with one individual.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (defaults to 'vsr_models/<model_name>/results_optimise/<timestamp>').",
    )

    args = parser.parse_args()

    # validate args
    if args.controller_type == "mlp_plus" and not args.mlp_plus_hidden_sizes:
        print(
            "Error: --mlp_plus_hidden_sizes must be provided for controller_type 'mlp_plus'"
        )
        sys.exit(1)
    if args.controller_type == "rnn" and args.rnn_hidden_size <= 0:
        print(
            "Error: --rnn_hidden_size must be provided and positive for controller_type 'rnn'"
        )
        sys.exit(1)
    if args.clip_range_min >= args.clip_range_max:
        print("Error: --clip_range_min must be less than --clip_range_max")
        sys.exit(1)
    if args.initial_param_vector_path and not os.path.exists(
        args.initial_param_vector_path
    ):
        print(
            f"Error: Initial parameter vector file not found: {args.initial_param_vector_path}"
        )
        sys.exit(1)

    # STEP 2: Setup simulation, load params
    MODEL_NAME = args.model_name
    MODEL_BASE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}"
    MODEL_CSV_PATH = MODEL_BASE_PATH + ".csv"
    CLIP_RANGE = (args.clip_range_min, args.clip_range_max)

    if not os.path.exists(MODEL_CSV_PATH):
        print(f"Error: Model CSV file not found at {MODEL_CSV_PATH}")
        sys.exit(1)

    # setup output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(
            f"vsr_models/{MODEL_NAME}/results_optimise", timestamp
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # prepare VSR and GET XML STRING
    vsr_instance = None
    xml_string = None
    active_coords_list = []
    try:
        print("Generating/Loading MuJoCo model and VSR info...")
        # infer grid size
        temp_vsr_for_dims = VoxelRobot(10, 10, 10, args.gear)
        temp_vsr_for_dims.load_model_csv(MODEL_CSV_PATH)
        vsr_dims = temp_vsr_for_dims.voxel_grid.shape
        del temp_vsr_for_dims

        vsr_instance = VoxelRobot(*vsr_dims, gear=args.gear)
        vsr_instance.load_model_csv(MODEL_CSV_PATH)
        xml_string = vsr_instance.generate_model(
            MODEL_BASE_PATH
        )  # generate_model returns the string

        temp_model_for_info = mujoco.MjModel.from_xml_string(xml_string)
        active_coords_list = [
            tuple(c) for c in np.argwhere(vsr_instance.voxel_grid == 1)
        ]

        print(
            f"MuJoCo XML generated. Sim timestep: {temp_model_for_info.opt.timestep}, Control timestep: {args.control_timestep}"
        )
        print(
            f"VSR Gear: {vsr_instance.gear}, Active Voxels: {len(active_coords_list)}"
        )
        del temp_model_for_info

    except Exception as e:
        print(f"Failed to load or generate MuJoCo model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # load initial parameters if provided
    initial_param_vector = None
    if args.initial_param_vector_path:
        try:
            initial_param_vector = np.load(args.initial_param_vector_path)
            print(f"Loaded initial parameters from: {args.initial_param_vector_path}")
        except Exception as e:
            print(
                f"Warning: Failed to load initial parameters: {e}. Proceeding with random initialization."
            )
            initial_param_vector = None  # Ensure it's None if loading failed

    # STEP 3: Run simulation
    if xml_string and vsr_instance and active_coords_list:
        # Run Optimisation
        start_opt_time = time.time()
        best_vector, history_data = optimise(
            xml_string=xml_string,
            vsr_voxel_coords_list=active_coords_list,
            vsr_gear_ratio=vsr_instance.gear,
            # EA params
            controller_type=args.controller_type,
            num_generations=args.num_generations,
            population_size=args.population_size,
            tournament_size=args.tournament_size,
            crossover_probability=args.crossover_probability,
            mutation_sigma=args.mutation_sigma,
            clip_range=CLIP_RANGE,
            # controller config
            mlp_plus_hidden_sizes=args.mlp_plus_hidden_sizes,
            rnn_hidden_size=args.rnn_hidden_size,
            # simulation params
            simulation_duration=args.simulation_duration,
            control_timestep=args.control_timestep,
            num_workers=args.num_workers,
            initial_param_vector=initial_param_vector,  # pass loaded or None
        )

        end_opt_time = time.time()
        print(f"\nTotal optimisation time: {end_opt_time - start_opt_time:.2f} seconds")

        # results
        if best_vector is not None:
            print("\n--- Best Found Parameter Vector ---")
            config_str = f"{args.controller_type}"
            if args.controller_type == "mlp_plus":
                config_str += f"_h{'_'.join(map(str, args.mlp_plus_hidden_sizes))}"
            if args.controller_type == "rnn":
                config_str += f"_h{args.rnn_hidden_size}"
            save_path_params = os.path.join(
                args.output_dir,
                f"best_params_{config_str}_gen{args.num_generations}_pop{args.population_size}.npy",
            )
            np.save(save_path_params, best_vector)
            print(f"Best parameter vector saved to {save_path_params}")

            save_path_history = os.path.join(
                args.output_dir,
                f"full_history_{config_str}_gen{args.num_generations}_pop{args.population_size}.csv",
            )
            try:
                history_data.to_csv(save_path_history, index=False)
                print(f"Full history saved to {save_path_history}")
            except Exception as e:
                print(f"Error saving history CSV: {e}")

            # run final simulation with viewer
            print(
                "\nRunning final simulation with best parameters (press Ctrl+C to skip)..."
            )
            try:
                # load the model again for the final run
                final_run_model = mujoco.MjModel.from_xml_string(xml_string)
                if final_run_model:
                    final_results = run_simulation(
                        model=final_run_model,
                        duration=args.simulation_duration,
                        control_timestep=args.control_timestep,
                        param_vector=best_vector,
                        controller_type=args.controller_type,
                        mlp_plus_hidden_sizes=args.mlp_plus_hidden_sizes,
                        rnn_hidden_size=args.rnn_hidden_size,
                        headless=False,  # show viewer
                    )
                    print(f"Final simulation results: {final_results}")
                else:
                    print("Could not load model for final simulation.")
            except KeyboardInterrupt:
                print("\nSkipping final simulation.")
            except Exception as e:
                print(f"Error running final simulation: {e}")
                traceback.print_exc()
        else:
            print("\nOptimisation did not yield a valid best result.")
            if not history_data.empty:
                # save history even if optimisation failed
                config_str = f"{args.controller_type}"
                if args.controller_type == "mlp_plus":
                    config_str += f"_h{'_'.join(map(str, args.mlp_plus_hidden_sizes))}"
                if args.controller_type == "rnn":
                    config_str += f"_h{args.rnn_hidden_size}"
                save_path_history = os.path.join(
                    args.output_dir,
                    f"failed_opt_history_{config_str}_gen{args.num_generations}_pop{args.population_size}.csv",
                )
                try:
                    history_data.to_csv(save_path_history, index=False)
                    print(f"Saved history from failed run to {save_path_history}")
                except Exception as e:
                    print(f"Error saving failed history CSV: {e}")
    else:
        print("Optimisation could not start due to setup errors.")
        sys.exit(1)
