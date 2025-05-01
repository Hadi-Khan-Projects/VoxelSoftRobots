# evolve.py
import collections
import os
import random
import time
import traceback

import numpy as np
import pandas as pd

import optimise  # Import the optimisation script
from vsr import VoxelRobot

# --- Constants ---
GRID_SIZE = 10  # VSR operates within a 10x10x10 grid
MUTATION_RETRY_LIMIT = 10  # Max attempts to get a valid mutation

# --- Morphology Helper Functions ---


def _is_within_bounds(coord):
    """
    Check if a coordinate is within the defined grid.

    Args:
        coord (tuple): A tuple of (x, y, z) coordinates.

    Returns:
        bool: True if within bounds, False otherwise.
    """
    return all(0 <= c < GRID_SIZE for c in coord)


def _get_neighbors(coord):
    """
    Get the 6 neighboring coordinates.

    Args:
        coord (tuple): A tuple of (x, y, z) coordinates.

    Returns:
        list: A list of neighboring coordinates.
    """
    x, y, z = coord
    neighbors = [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    ]
    return [n for n in neighbors if _is_within_bounds(n)]


def _find_surface_voxels(vsr: VoxelRobot):
    """
    Find all voxels on the surface of the VSR.

    Args:
        vsr (VoxelRobot): The VSR instance.

    Returns:
        list: A list of surface voxel coordinates.
    """
    surface_voxels = []
    active_voxels = list(zip(*np.where(vsr.voxel_grid == 1)))
    active_voxel_set = set(active_voxels)

    for x, y, z in active_voxels:
        is_surface = False
        for nx, ny, nz in _get_neighbors((x, y, z)):
            if (nx, ny, nz) not in active_voxel_set:
                is_surface = True
                break
        if is_surface:
            surface_voxels.append((x, y, z))
    return surface_voxels


def _count_connections(coord, active_voxel_set):
    """
    Count how many active neighbors a coordinate has.

    Args:
        coord (tuple): A tuple of (x, y, z) coordinates.
        active_voxel_set (set): Set of active voxel coordinates.

    Returns:
        int: Number of active neighbors.
    """
    count = 0
    for n_coord in _get_neighbors(coord):
        if n_coord in active_voxel_set:
            count += 1
    return count


def _find_candidate_neighbors(coord, vsr: VoxelRobot, find_empty: bool):
    """
    Find neighbors suitable for addition (empty) or removal (active).
    Returns a list of (neighbor_coord, connectivity_score) tuples, sorted.
    For add (find_empty=True): higher score is better.
    For remove (find_empty=False): lower score is better.

    Args:
        coord (tuple): The coordinate to check neighbors for.
        vsr (VoxelRobot): The VSR instance.
        find_empty (bool): True to find empty neighbors, False for active.

    Returns:
        list: A list of candidate neighbors with their scores.
    """
    active_voxel_set = set(list(zip(*np.where(vsr.voxel_grid == 1))))
    candidates = []
    for n_coord in _get_neighbors(coord):
        is_empty = n_coord not in active_voxel_set
        is_active = not is_empty

        if find_empty and is_empty:  # candidate for ADDITION
            # score based on how many connections it *would* have
            score = _count_connections(n_coord, active_voxel_set)
            candidates.append((n_coord, score))
        elif not find_empty and is_active and n_coord != coord:  # candidate for REMOVAL
            # score based on current connections (lower is less critical)
            score = _count_connections(n_coord, active_voxel_set)
            candidates.append((n_coord, score))

    # sort based on score (desc for add, asc for remove)
    candidates.sort(key=lambda item: item[1], reverse=find_empty)
    return candidates


def _bfs_find_candidates(
    start_coord, vsr: VoxelRobot, find_empty: bool, num_needed: int
):
    """
    Use BFS starting from start_coord over *surface* voxels to find
    enough candidate neighbors for addition or removal.

    Args:
        start_coord (tuple): Starting coordinate for BFS.
        vsr (VoxelRobot): The VSR instance.
        find_empty (bool): True to find empty neighbors, False for active.
        num_needed (int): Number of unique candidates needed.

    Returns:
        list: A list of candidate coordinates.
    """
    surface_voxels = _find_surface_voxels(vsr)
    if not surface_voxels:
        return []  # should not happen if called correctly

    queue = collections.deque([(start_coord, 0)])  # (coord, distance)
    visited = {start_coord}
    all_candidates = []  # list of (coord, score, distance)

    while queue:
        current_coord, dist = queue.popleft()

        # find candidates around the current surface voxel
        local_candidates = _find_candidate_neighbors(current_coord, vsr, find_empty)
        for cand_coord, score in local_candidates:
            # add distance to allow prioritisation
            all_candidates.append((cand_coord, score, dist))

        # stop if we potentially have enough unique candidates
        if len(set(c[0] for c in all_candidates)) >= num_needed:
            break  # Optimisation, might collect more than needed

        # explore neighbors on the surface
        active_voxel_set = set(list(zip(*np.where(vsr.voxel_grid == 1))))
        for neighbor in _get_neighbors(current_coord):
            if (
                neighbor in active_voxel_set
                and neighbor in surface_voxels
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    # post-process candidates: remove duplicates, sort primarily by score, then distance
    unique_candidates = {}
    for cand_coord, score, dist in all_candidates:
        if (
            cand_coord not in unique_candidates
            or unique_candidates[cand_coord][1] > dist
        ):
            # keep the one found with the shortest BFS distance
            unique_candidates[cand_coord] = (score, dist)

    # convert back to list and sort
    sorted_candidates = sorted(
        [(coord, score, dist) for coord, (score, dist) in unique_candidates.items()],
        key=lambda item: (
            item[1],
            item[2],
        ),  # sort by score (primary), then distance (secondary)
        reverse=find_empty,  # high score first for add, Low score first for remove
    )

    # return only the coordinates of the top candidates needed
    return [c[0] for c in sorted_candidates[:num_needed]]


def mutate_morphology(vsr: VoxelRobot, verbose=False):  # verbose flag
    """
    Mutates the VSR morphology by adding 3 or removing 2 voxels from the surface.
    Ensures contiguity and bounds. Biased towards growth.
    Returns a *new* mutated VoxelRobot instance, or None if mutation fails.

    Args:
        vsr (VoxelRobot): The VSR instance to mutate.
        verbose (bool): If True, print detailed mutation process.

    Returns:
        VoxelRobot: A new mutated VSR instance, or None if mutation fails.
    """
    for attempt in range(MUTATION_RETRY_LIMIT):
        if verbose:
            print(f"\nMutation Attempt {attempt + 1}/{MUTATION_RETRY_LIMIT}")
        new_vsr = None  # ensure new_vsr is reset each attempt
        try:
            new_vsr = VoxelRobot(vsr.max_x, vsr.max_y, vsr.max_z, vsr.gear)
            new_vsr.voxel_grid = np.copy(vsr.voxel_grid)  # Work on a copy

            surface_voxels = _find_surface_voxels(new_vsr)
            if not surface_voxels:
                if verbose:
                    print("  Failure: No surface voxels found.")
                # cannot mutate if no surface exists (e.g., single voxel)
                return None  # fail.

            start_coord = random.choice(surface_voxels)
            if verbose:
                print(f"  Starting BFS from surface voxel: {start_coord}")

            # decide operation: 50% add, 50% remove
            add_operation = random.random() < 0.5
            operation_type = "ADD" if add_operation else "REMOVE"
            if verbose:
                print(f"  Operation Type: {operation_type}")

            if add_operation:
                # add 3 voxels
                num_to_operate = 3
                candidates = _bfs_find_candidates(
                    start_coord, new_vsr, find_empty=True, num_needed=num_to_operate
                )
                if verbose:
                    print(
                        f"  Found {len(candidates)} candidates for addition: {candidates}"
                    )

                if len(candidates) < num_to_operate:
                    if verbose:
                        print(
                            "  Failure: Not enough valid locations found for addition."
                        )
                    continue  # try mutation again

                voxels_to_change = candidates[:num_to_operate]
                if verbose:
                    print(f"  Selected voxels to add: {voxels_to_change}")
                for x, y, z in voxels_to_change:
                    if _is_within_bounds((x, y, z)):  # Double check bounds
                        new_vsr.voxel_grid[x, y, z] = 1
                    else:
                        if verbose:
                            print(f"  Warning: Candidate {x, y, z} out of bounds.")
                        # this shouldnt happen if _get_neighbors checks bounds correctly
                        # but as a safeguard, maybe fail this attempt
                        raise ValueError("Candidate out of bounds")

                # check contiguity (catch ValueError)
                is_contiguous = False
                try:
                    new_vsr._check_contiguous()  # this raises ValueError on failure
                    is_contiguous = True  # If no error, it's contiguous
                    if verbose:
                        print("  Contiguity Check: PASSED")
                except ValueError:
                    if verbose:
                        print("  Contiguity Check: FAILED")
                    is_contiguous = False

                if is_contiguous:
                    if verbose:
                        print("  Mutation Successful (ADD)")
                    return new_vsr
                else:
                    if verbose:
                        print("  Failure: Resulting VSR (ADD) is not contiguous.")
                    continue  # try mutation again

            else:
                # remove 2 voxels
                num_to_operate = 2
                # use BFS to find the least connected *active* neighbors
                candidates = _bfs_find_candidates(
                    start_coord, new_vsr, find_empty=False, num_needed=num_to_operate
                )
                if verbose:
                    print(
                        f"  Found {len(candidates)} candidates for removal: {candidates}"
                    )

                if len(candidates) < num_to_operate:
                    if verbose:
                        print("  Failure: Not enough removable voxels found.")
                    continue  # try mutation again

                voxels_to_change = candidates[:num_to_operate]
                if verbose:
                    print(f"  Selected voxels to remove: {voxels_to_change}")

                for x, y, z in voxels_to_change:
                    if new_vsr.voxel_grid[x, y, z] == 1:  # make sure it's active
                        new_vsr.voxel_grid[x, y, z] = 0
                    else:
                        # this indicates a logic error in candidate selection
                        if verbose:
                            print(
                                f"  ERROR: Tried to remove non-active voxel {x, y, z}"
                            )
                        raise RuntimeError("Tried to remove non-active voxel")

                # check if VSR became empty
                is_empty = np.sum(new_vsr.voxel_grid) == 0
                if is_empty:
                    if verbose:
                        print("  Failure: Resulting VSR is empty.")
                    continue  # cannot have an empty VSR

                # check contiguity (catch ValueError)
                is_contiguous = False
                try:
                    new_vsr._check_contiguous()  # this raises ValueError on failure
                    is_contiguous = True  # if no error, it's contiguous
                    if verbose:
                        print("  Contiguity Check: PASSED")
                except ValueError:
                    if verbose:
                        print("  Contiguity Check: FAILED")
                    is_contiguous = False

                if is_contiguous:
                    if verbose:
                        print("  Mutation Successful (REMOVE)")
                    return new_vsr
                else:
                    if verbose:
                        print("  Failure: Resulting VSR (REMOVE) is not contiguous.")
                    continue  # try mutation again

        except Exception as e:
            print(f"  Error during mutation attempt {attempt + 1}: {e}")
            # print(traceback.format_exc())
            continue  # try again

    # if loop finishes without returning, all attempts failed
    print(f"Mutation failed after {MUTATION_RETRY_LIMIT} attempts.")
    return None


def evolve(
    initial_morphology_csv: str,
    output_dir: str,
    # evolution Params
    num_batches: int,
    num_mutations_per_batch: int,  # e.g., 12
    num_parents_select: int,  # e.g., 3
    # optimisation Params (passed to optimise.optimise)
    optimise_generations: int,  # generations per morphology
    optimise_population_size: int,
    optimise_args: dict,  # other args for optimise.optimise
    # VSR Params
    vsr_grid_dims: tuple = (GRID_SIZE, GRID_SIZE, GRID_SIZE),
    vsr_gear_ratio: float = 100.0,
    # debugging
    mutation_verbose: bool = False,  # add flag to control mutation logging
):
    """
    Co-evolves VSR morphology and control strategy.

    Args:
        initial_morphology_csv (str): Path to the initial morphology CSV file.
        output_dir (str): Directory to save results.
        num_batches (int): Number of evolution batches.
        num_mutations_per_batch (int): Number of mutations per batch.
        num_parents_select (int): Number of parents to select for next batch.
        optimise_generations (int): Generations for controller optimisation.
        optimise_population_size (int): Population size for optimisation.
        optimise_args (dict): Additional arguments for optimisation.
        vsr_grid_dims (tuple): Dimensions of the VSR grid.
        vsr_gear_ratio (float): Gear ratio for the VSR.
        mutation_verbose (bool): If True, print detailed mutation process.

    Returns:
        tuple: Best overall morphology VSR and a list of all batch histories.
    """
    print("--- Starting VSR Morphology & Control Co-Evolution ---")
    os.makedirs(output_dir, exist_ok=True)

    # Load initial morphology
    try:
        initial_vsr = VoxelRobot(*vsr_grid_dims, gear=vsr_gear_ratio)
        initial_vsr.load_model_csv(initial_morphology_csv)
        print(f"Loaded initial morphology from: {initial_morphology_csv}")
        initial_vsr._check_contiguous()  # Validate initial morphology
    except Exception as e:
        print(f"Fatal Error: Could not load or validate initial morphology: {e}")
        print(traceback.format_exc())
        return None, []

    current_morphologies = [initial_vsr]  # list of VoxelRobot instances
    all_batch_histories = []
    best_overall_fitness = -np.inf
    best_overall_morphology_vsr = initial_vsr

    # evolution loop (batches)
    for batch_num in range(num_batches):
        batch_start_time = time.time()
        print(f"\n--- Starting Batch {batch_num + 1}/{num_batches} ---")
        print(f"Parent morphologies for this batch: {len(current_morphologies)}")

        mutations_to_process = []
        # ensure we don't get stuck if parent list is empty (shouldn't happen with current logic but just to be safe)
        if not current_morphologies:
            print("Error: No parent morphologies available. Stopping.")
            break
        parents_cycle = current_morphologies * (
            num_mutations_per_batch // len(current_morphologies) + 1
        )

        # STEP 1: Generate mutations
        print(f"Generating {num_mutations_per_batch} mutations...")
        attempts = 0
        max_total_attempts = (
            num_mutations_per_batch * MUTATION_RETRY_LIMIT * 3
        )  # increase total attempts limit slightly

        while (
            len(mutations_to_process) < num_mutations_per_batch
            and attempts < max_total_attempts
        ):
            parent_index = len(mutations_to_process) % len(parents_cycle)
            parent_vsr = parents_cycle[parent_index]

            mutated_vsr = mutate_morphology(
                parent_vsr, verbose=mutation_verbose
            )  # verbose flag

            if mutated_vsr:
                mutations_to_process.append(mutated_vsr)
                if not mutation_verbose:
                    print(
                        f"  Generated mutation {len(mutations_to_process)}/{num_mutations_per_batch}"
                    )
            # else: # mutate_morphology now prints its own failure message
            #     pass
            attempts += 1
        # end of generation loop

        print(
            f"Generated {len(mutations_to_process)} mutations successfully for batch {batch_num + 1}."
        )

        if not mutations_to_process:
            print(
                "Error: Could not generate any valid mutations for this batch. Stopping."
            )
            break  # exit evolve loop if no mutations could be made

        # STEP 2: Optimise Controller for Each Mutation
        batch_results = []  # list of (VoxelRobot, best_fitness, best_params_vector)
        batch_history_records = []  # aggregated history from optimise runs

        for mut_idx, mutated_vsr in enumerate(mutations_to_process):
            print(
                f"\nOptimizing Mutation {mut_idx + 1}/{len(mutations_to_process)} (Batch {batch_num + 1})..."
            )
            # define paths for this specific mutation
            mutation_id = f"batch{batch_num + 1}_mut{mut_idx + 1}"
            mutation_model_dir = os.path.join(output_dir, "models", mutation_id)
            os.makedirs(mutation_model_dir, exist_ok=True)
            mutation_csv_path = os.path.join(mutation_model_dir, f"{mutation_id}.csv")
            mutation_base_path = os.path.join(
                mutation_model_dir, mutation_id
            )  # base for vsr.generate_model output

            try:
                # save morphology and generate XML string
                mutated_vsr.save_model_csv(mutation_csv_path)
                xml_string = mutated_vsr.generate_model(mutation_base_path)
                active_coords = [
                    tuple(c) for c in np.argwhere(mutated_vsr.voxel_grid == 1)
                ]
                if not active_coords:
                    print(
                        "Warning: Mutation resulted in an empty VSR. Skipping optimisation."
                    )
                    batch_results.append((mutated_vsr, -np.inf, None))
                    continue

                print(f"  Saved mutation CSV to {mutation_csv_path}")
                # print(f"  Generated MuJoCo model files at {mutation_base_path}*.xml") # Less verbose
                print(f"  Number of active voxels: {len(active_coords)}")

                # run controller optimisation
                optimise_start_time = time.time()
                best_params_vector, history_df = optimise.optimise(
                    xml_string=xml_string,
                    vsr_voxel_coords_list=active_coords,
                    vsr_gear_ratio=mutated_vsr.gear,
                    num_generations=optimise_generations,
                    population_size=optimise_population_size,
                    **optimise_args,  # Pass the rest of the args
                )
                optimise_duration = time.time() - optimise_start_time
                print(f"  Optimisation finished in {optimise_duration:.2f} seconds.")

                # process results
                best_fitness_for_mutation = -np.inf
                if not history_df.empty:
                    # add batch and mutation index to the history
                    history_df.insert(0, "mutation_index", mut_idx + 1)
                    history_df.insert(0, "batch", batch_num + 1)
                    batch_history_records.append(history_df)

                    # find the best fitness achieved during this optimisation run
                    if "fitness" in history_df.columns:
                        # ensure we get a finite number, default to -inf
                        try:
                            max_fit = history_df["fitness"].max()
                            best_fitness_for_mutation = (
                                float(max_fit) if np.isfinite(max_fit) else -np.inf
                            )
                        except Exception:
                            best_fitness_for_mutation = -np.inf

                print(
                    f"  Best fitness for this mutation: {best_fitness_for_mutation:.4f}"
                )
                batch_results.append(
                    (mutated_vsr, best_fitness_for_mutation, best_params_vector)
                )

                # update overall best if needed
                if best_fitness_for_mutation > best_overall_fitness:
                    best_overall_fitness = best_fitness_for_mutation
                    best_overall_morphology_vsr = mutated_vsr  # keep the VSR object
                    print(
                        f"  *** New overall best morphology found (Fitness: {best_overall_fitness:.4f}) ***"
                    )

            except Exception as e:
                print(f"Error processing mutation {mut_idx + 1}: {e}")
                # print(traceback.format_exc()) # Uncomment for debug
                batch_results.append((mutated_vsr, -np.inf, None))  # record failure

        # STEP 3: Save combined batch history
        if batch_history_records:
            combined_batch_history_df = pd.concat(
                batch_history_records, ignore_index=True
            )
            batch_history_path = os.path.join(
                output_dir, f"batch_{batch_num + 1}_history.csv"
            )
            try:
                combined_batch_history_df.to_csv(batch_history_path, index=False)
                print(f"\nSaved batch {batch_num + 1} history to: {batch_history_path}")
                all_batch_histories.append(combined_batch_history_df)
            except Exception as e:
                print(f"Error saving batch history CSV: {e}")
        else:
            print("\nNo history records generated for this batch.")

        # STEP 4: Select top parents for next batch
        # filter out results where fitness is -inf before sorting
        valid_results = [
            res for res in batch_results if np.isfinite(res[1]) and res[1] > -np.inf
        ]

        if not valid_results:
            print("Warning: No successful candidates in this batch to select from.")
            # re-use parents from the previous batch (current_morphologies)
            # BUT COULD TRY IN FUTURE:
            # a) Stop evolution
            # b) Generate new random mutations from parents
            print("Re-using parents from the previous batch.")
            if not current_morphologies:  # If even the initial parents failed somehow
                print("Error: No previous parents to re-use. Stopping.")
                break
        else:
            # sort valid results by fitness (descending)
            valid_results.sort(key=lambda x: x[1], reverse=True)
            # select top N parents
            top_candidates = valid_results[:num_parents_select]
            current_morphologies = [
                res[0] for res in top_candidates
            ]  # update parents for next loop
            print(f"\nSelected {len(current_morphologies)} parents for the next batch.")
            print(f"Top fitness in batch: {top_candidates[0][1]:.4f}")

        batch_duration = time.time() - batch_start_time
        print(f"Batch {batch_num + 1} finished in {batch_duration:.2f} seconds.")

    # end of evolution
    print("\n--- Evolution Finished ---")
    print(f"Best overall fitness achieved: {best_overall_fitness:.4f}")

    if best_overall_morphology_vsr:
        final_best_path = os.path.join(output_dir, "best_overall_morphology.csv")
        try:
            best_overall_morphology_vsr.save_model_csv(final_best_path)
            print(f"Saved best overall morphology structure to: {final_best_path}")
            # best *parameters* for this morphology are not explicitly saved here,
            # but can be found in the corresponding batch history file.
        except Exception as e:
            print(f"Error saving final best morphology: {e}")
    else:
        final_best_path = None
        print("No best morphology was recorded.")

    # combine all batch histories
    if all_batch_histories:
        try:
            full_history_df = pd.concat(all_batch_histories, ignore_index=True)
            full_history_path = os.path.join(output_dir, "full_evolution_history.csv")
            full_history_df.to_csv(full_history_path, index=False)
            print(f"Saved full aggregated history to: {full_history_path}")
        except Exception as e:
            print(f"Error saving full aggregated history: {e}")

    return final_best_path, all_batch_histories


# example usage
if __name__ == "__main__":
    # VSR / Model
    MODEL_NAME = "co_evolve_4x2x2"
    INITIAL_MORPHOLOGY_COORDS = [  # 4x2x2 block starting at (4,4,4)
        (4, 4, 4),
        (5, 4, 4),
        (6, 4, 4),
        (7, 4, 4),
        (4, 5, 4),
        (5, 5, 4),
        (6, 5, 4),
        (7, 5, 4),
        (4, 4, 5),
        (5, 4, 5),
        (6, 4, 5),
        (7, 4, 5),
        (4, 5, 5),
        (5, 5, 5),
        (6, 5, 5),
        (7, 5, 5),
    ]
    VSR_GRID_DIMS = (10, 10, 10)
    VSR_GEAR = 100.0  # Potentially needs tuning

    # Evolution Control
    NUM_BATCHES = 50  # Total morphology evolution batches
    NUM_MUTATIONS_PER_BATCH = 12  # Morphologies generated/tested per batch
    NUM_PARENTS_SELECT = 3  # Top morphologies selected as parents

    # Controller Optimisation (per morphology)
    OPTIMISE_GENERATIONS = 16  # Generations for *controller* opt
    OPTIMISE_POPULATION_SIZE = 30  # Population size for *controller* opt
    OPTIMISE_NUM_WORKERS = 30  # Workers for *controller* opt

    # Simulation Parameters (used within optimise)
    SIMULATION_DURATION = 60  # Seconds
    CONTROL_TIMESTEP = 0.2  # Control decision frequency (lower is faster control)

    # Controller Configuration (used within optimise)
    CONTROLLER_TYPE = "mlp"  # 'mlp', 'mlp_plus', 'rnn'
    MLP_PLUS_HIDDEN_SIZES = [24, 16]  # Only used if controller_type='mlp_plus'
    RNN_HIDDEN_SIZE = 16  # Only used if controller_type='rnn'

    # EA Parameters (for optimise.optimise)
    TOURNAMENT_SIZE = 4
    CROSSOVER_PROBABILITY = 0.8
    MUTATION_SIGMA = 0.2
    CLIP_RANGE = (-5.0, 5.0)

    # Output Directory
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIR = f"evolution_results/{MODEL_NAME}_{TIMESTAMP}"

    MUTATION_VERBOSE_LOGGING = True  # Set to True to see detailed mutation logs

    initial_csv_dir = os.path.join(OUTPUT_DIR, "initial_model")
    os.makedirs(initial_csv_dir, exist_ok=True)
    initial_csv_path = os.path.join(initial_csv_dir, f"{MODEL_NAME}_initial.csv")
    try:
        print(f"Creating initial morphology file at: {initial_csv_path}")
        initial_df = pd.DataFrame(INITIAL_MORPHOLOGY_COORDS, columns=["x", "y", "z"])
        initial_df.to_csv(initial_csv_path, index=False)
        # Validate creation
        temp_vsr = VoxelRobot(*VSR_GRID_DIMS, gear=VSR_GEAR)
        temp_vsr.load_model_csv(initial_csv_path)
        temp_vsr._check_contiguous()
        print("Initial morphology created and validated.")
        del temp_vsr
    except Exception as e:
        print(f"Error creating or validating initial morphology CSV: {e}")
        exit()

    # --- Prepare arguments for optimise.optimise ---
    # ... (remains the same) ...
    optimise_args = {
        "controller_type": CONTROLLER_TYPE,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_probability": CROSSOVER_PROBABILITY,
        "mutation_sigma": MUTATION_SIGMA,
        "clip_range": CLIP_RANGE,
        "simulation_duration": SIMULATION_DURATION,
        "control_timestep": CONTROL_TIMESTEP,
        "num_workers": OPTIMISE_NUM_WORKERS,
        "mlp_plus_hidden_sizes": MLP_PLUS_HIDDEN_SIZES,
        "rnn_hidden_size": RNN_HIDDEN_SIZE,
    }

    # --- Run Evolution ---
    evo_start_time = time.time()

    final_best_morph_path, all_history = evolve(
        initial_morphology_csv=initial_csv_path,
        output_dir=OUTPUT_DIR,
        num_batches=NUM_BATCHES,
        num_mutations_per_batch=NUM_MUTATIONS_PER_BATCH,
        num_parents_select=NUM_PARENTS_SELECT,
        optimise_generations=OPTIMISE_GENERATIONS,
        optimise_population_size=OPTIMISE_POPULATION_SIZE,
        optimise_args=optimise_args,
        vsr_grid_dims=VSR_GRID_DIMS,
        vsr_gear_ratio=VSR_GEAR,
        mutation_verbose=MUTATION_VERBOSE_LOGGING,  # Pass the flag
    )

    # ... (rest of the __main__ block remains the same) ...
    evo_end_time = time.time()
    print(f"\nTotal Co-Evolution duration: {evo_end_time - evo_start_time:.2f} seconds")

    if final_best_morph_path:
        print(f"Evolution complete. Best morphology saved to: {final_best_morph_path}")
    else:
        print("Evolution finished, but no best morphology was determined.")
