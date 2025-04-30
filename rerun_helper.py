import ast  # for safe evaluation of string representations of lists
import os
import traceback

import mujoco
import numpy as np
import pandas as pd

from optimise import _get_vsr_details_and_controller_dims, unflatten_params
from simulate import run_simulation
from simulate_headless import run_simulation_headless
from vsr import VoxelRobot


def rerun_simulation_from_log(
    log_csv_path: str,
    generation: int,
    individual_index: int,
    model: mujoco.MjModel,  # Pass the actual loaded model object
    duration: int,
    control_timestep: float,
    headless: bool = True,
):
    """
    Reruns a simulation for a specific individual recorded in the optimization log.

    Args:
        log_csv_path (str): Path to the optimization history CSV file.
        generation (int): The generation number (1-based) to rerun.
        individual_index (int): The index (0-based) of the individual within the generation.
        model (mujoco.MjModel): The *loaded* MuJoCo model object corresponding to the VSR
                                 used when generating the log file.
        duration (int): Duration of the simulation in seconds.
        control_timestep (float): Timestep for controller updates in seconds.
        headless (bool): If True, runs the headless simulation. If False, attempts
                         to run the simulation with the viewer.

    Returns:
        tuple: The results from the simulation run (e.g., fitness, x_dist, y_dist, reached)
               Returns None if an error occurs during setup or execution.
    """
    print("\n--- Rerunning Simulation ---")
    print(f"Log file: {log_csv_path}")
    print(f"Generation: {generation}, Individual Index: {individual_index}")
    print(f"Mode: {'Headless' if headless else 'Viewer'}")

    if model is None:
        print("Error: A valid MuJoCo model object must be provided.")
        return None

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

    # 2. Find the Specific Row ---
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
            # print("Available generations:", history_df['generation'].unique())
            return None
        elif len(target_row) > 1:
            print(
                f"Warning: Multiple records found for Generation {generation}, Individual {individual_index}. Using the first one."
            )
            target_row = target_row.iloc[0:1]  # Select first row

        record = target_row.iloc[0]  # get the series for the target row
        print("Record found.")  # fitness: {record.get('fitness', 'N/A')}")

    except KeyError as e:
        print(
            f"Error: Column missing in CSV file: {e}. Expected 'generation', 'individual_index', 'params_vector'."
        )
        return None
    except Exception as e:
        print(f"Error finding record in DataFrame: {e}")
        return None

    # 3. Get Parameter Shapes for *this* model
    try:
        print("Determining controller dimensions from provided model...")
        n_voxels, voxel_coords, input_size, output_size = (
            _get_vsr_details_and_controller_dims(model)
        )
        weight_shape = (output_size, input_size)
        bias_shape = (output_size,)
        print(f"Determined Shapes: Weights={weight_shape}, Biases={bias_shape}")
    except Exception as e:
        print(f"Error determining parameter shapes from the provided model: {e}")
        print(traceback.format_exc())
        return None

    # 4. Extract and Convert Parameter Vector
    try:
        param_vector_str = record["params_vector"]
        # safe evaluate the string representation of the list
        param_vector_list = ast.literal_eval(param_vector_str)
        param_vector = np.array(param_vector_list, dtype=np.float64)
        print(f"Parameter vector extracted (length: {len(param_vector)}).")
    except KeyError:
        print("Error: 'params_vector' column not found in the log file.")
        return None
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error converting 'params_vector' string to numpy array: {e}")
        print("The column should contain a string representation of a list of numbers.")
        return None
    except Exception as e:
        print(f"Unexpected error processing parameter vector: {e}")
        return None

    # 5. Unflatten Parameters
    try:
        weights, biases = unflatten_params(param_vector, weight_shape, bias_shape)
        print("Parameters unflattened successfully.")
    except ValueError as e:
        print(f"Error unflattening parameters: {e}")
        print(
            "This might indicate a mismatch between the log file's parameters and the provided model's expected structure."
        )
        return None
    except Exception as e:
        print(f"Unexpected error during unflattening: {e}")
        return None

    # 6. Run Simulation
    results = None
    try:
        if headless:
            if run_simulation_headless is None:
                print(
                    "Error: Headless simulation function (run_simulation_headless) not available."
                )
                return None
            print("Starting headless simulation...")
            results = run_simulation_headless(
                model=model,
                duration=duration,
                control_timestep=control_timestep,
                weights=weights,
                biases=biases,
            )
            print("Headless simulation finished.")
        else:
            if run_simulation is None:
                print(
                    "Error: Viewer simulation function (run_simulation) not available."
                )
                return None
            print("Starting simulation with viewer...")
            results = run_simulation(
                model=model,  # Pass model object directly if modified
                duration=duration,
                control_timestep=control_timestep,
                weights=weights,
                biases=biases,
            )

    except Exception as e:
        print(f"Error during simulation execution: {e}")
        print(traceback.format_exc())
        return None

    return results


# example
if __name__ == "__main__":
    # model path
    MODEL_NAME = "quadruped_v3"  # Or the model used for the log
    MODEL_BASE_PATH = f"vsr_models/{MODEL_NAME}/{MODEL_NAME}"
    MODEL_CSV_PATH = MODEL_BASE_PATH + ".csv"

    # If and only if log file was saved by the optimization script
    GENERATION_LOGGED = 10
    POPULATION_LOGGED = 14
    LOG_CSV_PATH = f"{MODEL_BASE_PATH}_full_history_gen{GENERATION_LOGGED}_pop{POPULATION_LOGGED}.csv"

    # select the specific run
    GENERATION_TO_RUN = 7  # rerun generation
    INDIVIDUAL_TO_RUN = 10  # individual index 3 from that generation

    # CRUCIALLY: Load the EXACT Model used for the Log
    loaded_model = None
    if VoxelRobot is None:
        print("Cannot proceed without VoxelRobot class.")
    elif not os.path.exists(MODEL_CSV_PATH):
        print(f"Error: Model CSV for rerun not found at {MODEL_CSV_PATH}")
    else:
        try:
            print(f"\nLoading MuJoCo model from {MODEL_CSV_PATH}...")
            vsr_instance = VoxelRobot(10, 10, 10)  # Use appropriate dims
            vsr_instance.load_model_csv(MODEL_CSV_PATH)
            xml_string = vsr_instance.generate_model(
                MODEL_BASE_PATH
            )  # generates _modded.xml
            loaded_model = mujoco.MjModel.from_xml_string(xml_string)
            print("MuJoCo model loaded successfully for rerun.")
        except Exception as e:
            print(f"Failed to load MuJoCo model for rerun: {e}")
            print(traceback.format_exc())

    # if model loaded run sim with viewer
    if loaded_model:
        print("\nAttempting rerun with viewer:")

        viewer_results = rerun_simulation_from_log(
            log_csv_path=LOG_CSV_PATH,
            generation=GENERATION_TO_RUN,
            individual_index=INDIVIDUAL_TO_RUN,
            model=loaded_model,
            duration=60,
            control_timestep=0.2,
            headless=False,  # Request viewer
        )
        if viewer_results is not None:
            print("\nViewer Rerun Results (if run_simulation was adapted):")
            print(viewer_results)
        else:
            print(
                "\nViewer Rerun Skipped or Failed (check function availability/modification)."
            )
