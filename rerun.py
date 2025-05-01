import ast  # for safe evaluation of string representations of lists
import os
import traceback

import mujoco
import numpy as np
import pandas as pd

from optimise import _get_vsr_details_and_controller_dims, unflatten_params
from simulate import run_simulation
from vsr import VoxelRobot


def rerun_simulation_from_log(
    log_csv_path: str,
    generation: int,
    individual_index: int,
    vsr_grid_dims: tuple,  # e.g., (10, 10, 10)
    duration: int,
    headless: bool = True,
    temp_model_path: str = "vsr_rerun_cache",  # for temp model file
):
    """
    Reruns a simulation for a specific individual recorded in the optimization log,
    reconstructing the VSR model from logged data.

    Args:
        log_csv_path (str): Path to the optimization history CSV file.
        generation (int): The generation number (1-based) to rerun.
        individual_index (int): The index (0-based) of the individual within the generation.
        vsr_grid_dims (tuple): The (x, y, z) dimensions of the grid used for the
                               original VSR generation (needed for VoxelRobot init).
        duration (int): Duration of the simulation rerun in seconds.
        headless (bool): If True, runs headless. If False, uses viewer.
        temp_model_path (str): Base path for saving the reconstructed model XML temporarily.

    Returns:
        tuple: The results from the simulation run (e.g., fitness, x_dist, y_dist, reached)
               Returns None if an error occurs during setup or execution.
    """
    print("\n--- Rerunning Simulation from Log ---")
    print(f"Log file: {log_csv_path}")
    print(f"Generation: {generation}, Individual Index: {individual_index}")
    print(f"Mode: {'Headless' if headless else 'Viewer'}")
    print(f"VSR Grid Dimensions: {vsr_grid_dims}")

    # 1. Load Log Data
    try:
        history_df = pd.read_csv(log_csv_path)
        print(f"Loaded log file with {len(history_df)} records.")
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_csv_path}")
        return None
    except Exception as e:
        print(f"Error loading log file: {e}")
        return None

    # 2. Find the Specific Record
    try:
        # filter on generation and individual index
        target_row = history_df[
            (history_df["generation"] == generation)
            & (history_df["individual_index"] == individual_index)
        ]

        if target_row.empty:
            print(
                f"Error: No record found for Generation {generation}, Individual {individual_index}."
            )
            return None
        elif len(target_row) > 1:
            print(
                f"Warning: Multiple records found for Gen {generation}, Idx {individual_index}. Using the first one."
            )
        record = target_row.iloc[0]  # get the series for the target row
        print(f"Record found. Logged fitness: {record.get('fitness', 'N/A')}")

    except KeyError as e:
        print(f"Error: Column missing in CSV: {e}. Check log file structure.")
        return None
    except Exception as e:
        print(f"Error finding record in DataFrame: {e}")
        return None

    # 3. Extract Simulation Parameters and Voxel Coordinates from Record
    try:
        print("Extracting parameters from log record...")
        voxel_coords_str = record["voxel_coords_str"]
        control_timestep = record["control_timestep"]
        simulation_timestep_logged = record["simulation_timestep"]
        gear_ratio = record["gear_ratio"]
        param_vector_str = record["params_vector"]

        # convert voxel coords string back to list of tuples
        voxel_coords_list = ast.literal_eval(voxel_coords_str)
        if not isinstance(voxel_coords_list, list):
            raise TypeError("Parsed voxel coordinates are not a list.")
        if not all(
            isinstance(item, tuple) and len(item) == 3 for item in voxel_coords_list
        ):
            raise TypeError(
                "Parsed voxel coordinates list does not contain (x, y, z) tuples."
            )

        # convert parameter vector string to numpy array
        param_vector_list = ast.literal_eval(param_vector_str)
        param_vector = np.array(param_vector_list, dtype=np.float64)

        print(f"  Control Timestep: {control_timestep}")
        print(f"  Sim Timestep (Logged): {simulation_timestep_logged}")
        print(f"  Gear Ratio: {gear_ratio}")
        print(f"  Num Active Voxels: {len(voxel_coords_list)}")
        print(f"  Parameter Vector Length: {len(param_vector)}")

    except KeyError as e:
        print(f"Error: Missing required column in log file: {e}")
        return None
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing data from log record: {e}")
        print("Check 'voxel_coords_str' and 'params_vector' format in the CSV.")
        return None
    except Exception as e:
        print(f"Unexpected error processing log record data: {e}")
        return None

    # 4. Reconstruct VSR Model
    model = None
    vsr_instance = None
    try:
        print("Reconstructing VSR model...")
        # ensure temp path directory exists
        os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)

        # create VoxelRobot instance with logged gear and provided grid dims
        vsr_instance = VoxelRobot(*vsr_grid_dims, gear=gear_ratio)

        # set active voxels based on the loaded list
        for x, y, z in voxel_coords_list:
            if (
                0 <= x < vsr_instance.max_x
                and 0 <= y < vsr_instance.max_y
                and 0 <= z < vsr_instance.max_z
            ):
                vsr_instance.set_val(x, y, z, 1)
            else:
                print(
                    f"Warning: Voxel coord ({x},{y},{z}) from log is outside grid dims {vsr_grid_dims}. Skipping."
                )

        # generate the XML model string and save the temporary XML
        xml_string = vsr_instance.generate_model(
            temp_model_path
        )  # Generates _modded.xml

        # load the model into MuJoCo
        model = mujoco.MjModel.from_xml_string(xml_string)
        print(
            f"MuJoCo model reconstructed successfully. Sim Timestep (Actual): {model.opt.timestep}"
        )

        # --- CRITICAL CHECK ---
        # Use np.isclose for float comparison
        if not np.isclose(model.opt.timestep, simulation_timestep_logged):
            print(
                f"FATAL MISMATCH: Reconstructed model timestep ({model.opt.timestep}) "
                f"does not match logged timestep ({simulation_timestep_logged})."
            )
            print("Rerun would be inaccurate. Ensure VSR generation is deterministic.")
            # Clean up temporary file if desired
            # try: os.remove(temp_model_path + "_modded.xml") except OSError: pass
            return None  # Abort rerun
        # ----------------------

    except ImportError:
        print("Error: Could not import VoxelRobot class.")
        return None
    except FileNotFoundError:  # Should not happen with temp path unless perms issue
        print(f"Error generating model file at {temp_model_path}.")
        return None
    except Exception as e:
        print(f"Error during VSR model reconstruction: {e}")
        print(traceback.format_exc())
        return None

    # 5. Get Parameter Shapes for the *Reconstructed* Model
    try:
        print("Determining controller dimensions from reconstructed model...")
        n_voxels, _, input_size, output_size = _get_vsr_details_and_controller_dims(
            model
        )
        weight_shape = (output_size, input_size)
        bias_shape = (output_size,)
        print(f"Determined Shapes: Weights={weight_shape}, Biases={bias_shape}")

        # just to be safe compare n_voxels with len(voxel_coords_list)
        if n_voxels != len(voxel_coords_list):
            print(
                f"Warning: Number of active voxels in reconstructed model ({n_voxels}) "
                f"differs from number in log ({len(voxel_coords_list)}). Check model generation."
            )

    except Exception as e:
        print(f"Error determining parameter shapes from reconstructed model: {e}")
        print(traceback.format_exc())
        return None

    # 6. Unflatten Parameters
    try:
        weights, biases = unflatten_params(param_vector, weight_shape, bias_shape)
        print("Controller parameters unflattened successfully.")
    except ValueError as e:
        print(f"Error unflattening parameters: {e}")
        print(
            "Check if parameter vector length matches the reconstructed model's expected size."
        )
        return None
    except Exception as e:
        print(f"Unexpected error during unflattening: {e}")
        return None

    # 7. Run Simulation
    results = None
    try:
        print(f"Starting simulation rerun (Duration: {duration}s)...")
        results = run_simulation(
            model=model,  # Use the reconstructed model
            duration=duration,  # Use the duration specified for the rerun
            control_timestep=control_timestep,  # Use the control timestep from the log
            weights=weights,  # Use the unflattened weights
            biases=biases,  # Use the unflattened biases
            headless=headless,  # Use the requested mode
        )
        print("Simulation rerun finished.")

    except Exception as e:
        print(f"Error during simulation execution: {e}")
        print(traceback.format_exc())
        return None
    finally:
        # clean up the temporary model file after simulation
        modded_xml_path = temp_model_path + "_modded.xml"
        try:
            if os.path.exists(modded_xml_path):
                os.remove(modded_xml_path)
                # print(f"Removed temporary model file: {modded_xml_path}")
        except OSError as e:
            print(
                f"Warning: Could not remove temporary model file {modded_xml_path}: {e}"
            )

    return results


# example
if __name__ == "__main__":
    # --- Configuration for Rerun ---
    MODEL_NAME = "quadruped_v3"  # The prefix used during the optimization run

    # --- Log File Identification ---
    # These should match the parameters used when running optimise.py
    GENERATIONS_IN_LOG = 4  # Number of generations run to create the log
    POPULATION_IN_LOG = 8  # Population size used to create the log
    # --- Path to the log file ---
    LOG_FILE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}_full_history_gen{GENERATIONS_IN_LOG}_pop{POPULATION_IN_LOG}.csv"

    # --- Specific Run to Replay ---
    GENERATION_TO_RUN = 3  # Generation number from the log (1-based)
    INDIVIDUAL_TO_RUN = 4  # Individual index from that generation (0-based)

    # --- VSR Grid Dimensions (MUST match the original run) ---
    VSR_GRID_DIMS = (10, 10, 10)  # The max grid size used

    # --- Rerun Parameters ---
    RERUN_DURATION = 60  # How long to run this specific replay
    RERUN_HEADLESS = False  # Set to False to show viewer

    # --- Check if Log File Exists ---
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Error: Log file not found at {LOG_FILE_PATH}")
        print(
            "Please ensure the path and parameters (GENERATIONS_IN_LOG, POPULATION_IN_LOG) are correct."
        )
    else:
        # Define a base path for temporary files generated during rerun
        temp_model_base = f"vsr_models/{MODEL_NAME}/temp_rerun_{GENERATION_TO_RUN}_{INDIVIDUAL_TO_RUN}"

        # --- Attempt Rerun ---
        print(
            f"\nAttempting rerun for Gen {GENERATION_TO_RUN}, Idx {INDIVIDUAL_TO_RUN}..."
        )
        rerun_results = rerun_simulation_from_log(
            log_csv_path=LOG_FILE_PATH,
            generation=GENERATION_TO_RUN,
            individual_index=INDIVIDUAL_TO_RUN,
            vsr_grid_dims=VSR_GRID_DIMS,
            duration=RERUN_DURATION,
            headless=RERUN_HEADLESS,
            temp_model_path=temp_model_base,
        )

        # --- Display Results ---
        if rerun_results is not None:
            print("\n--- Rerun Simulation Results ---")
            fitness, x_dist, y_dist, reached = rerun_results
            print(f"  Fitness: {fitness:.4f}")
            print(f"  Final X Distance: {x_dist:.4f}")
            print(f"  Final Y Distance: {y_dist:.4f}")
            print(f"  Target Reached: {reached}")
        else:
            print("\nSimulation rerun failed or was aborted.")
