import argparse
import ast
import math
import os
import sys
import traceback
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

from controller import DistributedNeuralController
from vsr import VoxelRobot


def voxel_motor_mapping(
    model: mujoco.MjModel,
) -> tuple[list[tuple[int, int, int]], dict[int, list[int]], dict[int, list[int]]]:
    """
    Map motors and tendons to their respective voxel coordinates.

    Args:
        model (mujoco.MjModel): The MuJoCo model.

    Returns:
        list[tuple[int, int, int]]: List of active voxel coordinates.
    """

    voxel_motor_map = {}
    voxel_tendon_map = {}  # map voxels to their tendon indices

    # STEP 1: Map motors to voxels
    for i in range(model.nu):
        motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if motor_name and motor_name.startswith("voxel_"):
            parts = motor_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))
            if voxel_coord not in voxel_motor_map:
                voxel_motor_map[voxel_coord] = []
            voxel_motor_map[voxel_coord].append(i)

    # STEP 2: Map tendons to voxels (tendon names match motor names structure)
    for i in range(model.ntendon):
        tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        if tendon_name and tendon_name.startswith("voxel_"):
            parts = tendon_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))
            if voxel_coord not in voxel_tendon_map:
                voxel_tendon_map[voxel_coord] = []
            voxel_tendon_map[voxel_coord].append(i)

    # STEP 3: Ensure mappings are consistent
    active_voxel_coords = sorted(
        list(set(voxel_motor_map.keys()) | set(voxel_tendon_map.keys()))
    )  # get all defined voxels

    active_voxel_coords_validated = validate_mapping(
        active_voxel_coords,
        voxel_motor_map,
        voxel_tendon_map,
    )

    return active_voxel_coords_validated, voxel_motor_map, voxel_tendon_map


def validate_mapping(
    active_voxel_coords: list,
    voxel_motor_map: dict,
    voxel_tendon_map: dict,
):
    """
    Filter active_voxel_coords to include only those with both motors and tendons mapped.

    Args:
        active_voxel_coords (list): List of active voxel coordinates.
        voxel_motor_map (dict): Mapping of voxel coordinates to motor indices.
        voxel_tendon_map (dict): Mapping of voxel coordinates to tendon indices.

    Returns:
        list: Filtered list of active voxel coordinates.
    """
    valid_active_voxel_coords = []
    for voxel in active_voxel_coords:
        has_motors = voxel in voxel_motor_map and len(voxel_motor_map[voxel]) == 4
        has_tendons = voxel in voxel_tendon_map and len(voxel_tendon_map[voxel]) == 4
        if has_motors and has_tendons:
            valid_active_voxel_coords.append(voxel)
    active_voxel_coords = valid_active_voxel_coords
    n_active_voxels = len(active_voxel_coords)

    if n_active_voxels == 0:
        raise ValueError("No valid active voxels found during simulation setup.")

    return active_voxel_coords


def get_vertex_body_ids(
    model: mujoco.MjModel,
):
    vertex_body_ids = []
    i = 0
    while True:  # iterate through potential names
        body_name = f"vsr_{i}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            break
        vertex_body_ids.append(body_id)
        i += 1
    if not vertex_body_ids:
        raise ValueError("Could not find any vertex body IDs")
    vertex_body_ids = np.array(vertex_body_ids)

    return vertex_body_ids


def run_simulation(
    model: mujoco.MjModel,
    duration: int,
    control_timestep: float,
    controller_type: str,
    mlp_plus_hidden_sizes: list,
    rnn_hidden_size: int,
    weights: np.ndarray[Any, np.dtype[np.float64]] = None,
    biases: np.ndarray[Any, np.dtype[np.float64]] = None,
    param_vector: np.ndarray = None,  # if param_vector is used, ignore weights/biases
    headless: bool = True,
):
    """
    Run a simulation of the voxel robot with the given parameters.

    Args:
        model (mujoco.MjModel): The MuJoCo model.
        duration (int): Duration of the simulation in seconds.
        control_timestep (float): Time step for control updates.
        controller_type (str): Type of controller to use ('mlp', 'mlp_plus', 'rnn').
        mlp_plus_hidden_sizes (list): Hidden layer sizes for MLP+ controller.
        rnn_hidden_size (int): Hidden size for RNN controller.
        weights (np.ndarray): Weights for the controller.
        biases (np.ndarray): Biases for the controller.
        param_vector (np.ndarray): Parameter vector for the controller.
        headless (bool): If True, run without GUI.
    """
    data = mujoco.MjData(model)

    if not headless:
        print(f"MuJoCo Model timestep: {model.opt.timestep}")
        print(f"Controller Type: {controller_type}")
        if controller_type == "mlp_plus":
            print(f"  MLP+ Hidden: {mlp_plus_hidden_sizes}")
        if controller_type == "rnn":
            print(f"  RNN Hidden: {rnn_hidden_size}")

    # STEP 1: Create voxel to motor mapping
    active_voxel_coords, voxel_motor_map, voxel_tendon_map = voxel_motor_mapping(model)
    n_active_voxels = len(active_voxel_coords)

    if not headless:
        print(f"Proceeding with {n_active_voxels} fully mapped active voxels.")
    else:
        print(".", end="", flush=True)

    # STEP 2: Get vertex body ids for CoM calculations
    vertex_body_ids = get_vertex_body_ids(model)

    # STEP 3: Controller setup

    # controller config setup (same as NeuralController class)
    N_SENSORS_PER_VOXEL = 8  # 4 tendon lengths + 4 tendon velocities
    N_COMM_CHANNELS = 2  # as per paper's experiments (nc=2)
    # N_COMM_DIRECTIONS = 6  # voxels have 6 neighbors
    # N_TIME_INPUTS = 2  # number of sin(t)/cos(t) inputs

    # driving voxel (the one with the lowest x, then y, then z among valid ones)
    driving_voxel = active_voxel_coords[0] if active_voxel_coords else None

    controller = DistributedNeuralController(
        controller_type=controller_type,  # 'mlp' 'mlp_plus' 'rnn'
        n_voxels=n_active_voxels,
        voxel_coords=active_voxel_coords,
        n_sensors_per_voxel=N_SENSORS_PER_VOXEL,
        n_comm_channels=N_COMM_CHANNELS,
        param_vector=param_vector,  # if param_vector is used, ignore weights/biases
        weights=weights,  # Only used if param_vector is None and type is mlp
        biases=biases,  # Only used if param_vector is None and type is mlp
        mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        driving_voxel_coord=driving_voxel,
        time_signal_frequency=0.5,
    )

    # STEP 4: Simulation loop
    paused = False
    last_control_time = 0.0
    target_reached = False
    x_dist_target = np.inf
    y_dist_target = np.inf

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    SETTLE_DURATION = 3.0  # seconds to apply max contraction initially
    INITIAL_CTRL_VALUE = 1.0  # expand tendons initially (0 = shortest length for motor)

    controller.reset()  # needed for RNN and communication init

    if not headless:  # then run with viewer
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            # disable some viewer options (performance)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

            # disable some rendering flags (performance)
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0

            # viewer cam settings
            viewer.cam.lookat[:] = [-30, 0, 2.5]
            viewer.cam.distance = 70
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25

            while data.time < duration:
                sim_time = data.time

                # settling phase, no movement until VSR settles
                if sim_time <= SETTLE_DURATION and not paused:
                    for (
                        voxel_coord
                    ) in active_voxel_coords:  # iterate through valid voxels
                        for motor_id in voxel_motor_map[voxel_coord]:
                            data.ctrl[motor_id] = INITIAL_CTRL_VALUE
                    last_control_time = sim_time
                elif (
                    sim_time > SETTLE_DURATION
                    and sim_time >= last_control_time + control_timestep
                    and not paused
                ):
                    # STEP 1: Gather sensors data for all active voxels
                    sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        tendon_indices = voxel_tendon_map[voxel_coord]
                        sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                        sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]

                    # STEP 2: Calculate velocity, target distances, and orientation

                    # velocity
                    current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)
                    # use cvel which includes angular; take linear part [3:6]
                    current_com_vel = np.mean(data.cvel[vertex_body_ids][:, 3:], axis=0)

                    target_body_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "target"
                    )
                    robot_body_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "vsr"
                    )
                    if robot_body_id == -1:
                        robot_body_id = vertex_body_ids[
                            0
                        ]  # fallback to first vertex if 'vsr' body not found

                    # distances
                    target_pos = data.subtree_com[target_body_id]
                    x_dist_target = target_pos[0] - current_com_pos[0]
                    y_dist_target = target_pos[1] - current_com_pos[1]

                    # orientation
                    vec_to_target = target_pos - current_com_pos
                    vec_to_target_xy = vec_to_target[[0, 1]]
                    norm = np.linalg.norm(vec_to_target_xy)
                    target_orientation_vector = vec_to_target_xy / (norm + 1e-6)

                    if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                        target_reached = True

                    # STEP 3: Run controller step
                    # actuation_outputs are in range [-1, 1]
                    actuation_outputs = controller.step(
                        sensor_data_all,
                        sim_time,
                        current_com_vel,
                        target_orientation_vector,
                    )

                    # STEP 4: Initial mapping to [0, 1]
                    # initial_motor_signals shape: (n_voxels, 1)
                    initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                    # STEP 5: Apply clipped actuation to motors
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        motor_control_signal = np.clip(
                            initial_motor_signals[i, 0], 0.0, 1.0
                        )
                        for motor_id in voxel_motor_map[voxel_coord]:
                            data.ctrl[motor_id] = motor_control_signal

                    last_control_time = sim_time
                # end of control Step

                if not paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

    else:  # headless execution, no viewer
        while data.time < duration:
            sim_time = data.time

            # settling phase, no movement until VSR settles
            if sim_time <= SETTLE_DURATION:
                for voxel_coord in active_voxel_coords:
                    for motor_id in voxel_motor_map[voxel_coord]:
                        data.ctrl[motor_id] = INITIAL_CTRL_VALUE
                last_control_time = sim_time

            elif (
                sim_time > SETTLE_DURATION
                and sim_time >= last_control_time + control_timestep
            ):
                # STEP 1: Gather sensor data for ALL active voxels
                sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
                for i, voxel_coord in enumerate(active_voxel_coords):
                    tendon_indices = voxel_tendon_map[voxel_coord]
                    sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                    sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]

                # STEP 2: Calculate velocity, target distances, and orientation

                # velocity
                current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)
                # use cvel which includes angular; take linear part [3:6]
                current_com_vel = np.mean(data.cvel[vertex_body_ids][:, 3:], axis=0)

                # distance
                target_body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "target"
                )
                target_pos = data.subtree_com[target_body_id]
                x_dist_target = target_pos[0] - current_com_pos[0]
                y_dist_target = target_pos[1] - current_com_pos[1]

                # orientation
                vec_to_target = target_pos - current_com_pos
                vec_to_target_xy = vec_to_target[[0, 1]]
                norm = np.linalg.norm(vec_to_target_xy)
                target_orientation_vector = vec_to_target_xy / (norm + 1e-6)
                if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                    target_reached = True

                # STEP 3: Run controller step
                actuation_outputs = controller.step(
                    sensor_data_all,
                    sim_time,
                    current_com_vel,
                    target_orientation_vector,
                )

                # STEP 4: Initial mapping to [0, 1]
                # initial_motor_signals shape: (n_voxels, 1)
                initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                # STEP 5: Apply clipped actuation to motors
                for i, voxel_coord in enumerate(active_voxel_coords):
                    motor_control_signal = np.clip(
                        initial_motor_signals[i, 0], 0.0, 1.0
                    )
                    for motor_id in voxel_motor_map[voxel_coord]:
                        data.ctrl[motor_id] = motor_control_signal

                last_control_time = sim_time
            # end of control step

            try:
                mujoco.mj_step(model, data)
            except mujoco.FatalError as e:
                print(f"MuJoCo Fatal Error: {e}. Time: {data.time}")
                return -np.inf, np.inf, np.inf, False  # use -inf for fitness on crash

    # STEP 6: Fitness calculation
    # euclidean final distance from last control step
    final_distance = (
        math.sqrt(x_dist_target**2 + y_dist_target**2)
        if np.isfinite(x_dist_target) and np.isfinite(y_dist_target)
        else np.inf
    )
    fitness = (
        -final_distance if np.isfinite(final_distance) else -np.inf
    )  # penalise instability/NaNs

    # ensure finite returns for logging
    x_dist_target = x_dist_target if np.isfinite(x_dist_target) else np.inf
    y_dist_target = y_dist_target if np.isfinite(y_dist_target) else np.inf

    return fitness, x_dist_target, y_dist_target, target_reached


def parse_literal(value_str):
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError, TypeError):
        # handle cases where the string might be [], None, or malformed
        if isinstance(value_str, str):
            if value_str.strip() == "[]":
                return []
            if value_str.strip().lower() in ["none", "nan", "null"]:
                return None
        # if its already a list/None
        if isinstance(value_str, (list, type(None))):
            return value_str
        print(f"Warning: Could not parse literal string: {value_str}. Returning None.")
        return None


# cli usage for an optimise.py log file:

# python simulate.py \
#     --csv_path <path_to_optimise_history.csv> \
#     --generation <gen_number> \
#     --individual_index <index_number> \
#     --vsr_grid_dims 10 10 10 \
#     [--duration <seconds>] \
#     [--control_timestep <seconds>] \
#     [--headless]

# cli usage example for an optimise.py log file:

# python simulate.py \
#     --csv_path vsr_models/quadruped_v3/results_optimise/my_rnn_run_1/full_history_rnn_h16_gen2_pop8.csv \
#     --generation 2 \
#     --individual_index 0 \
#     --vsr_grid_dims 10 10 10

# cli usage for an evolve.py log file:

# python simulate.py \
#     --csv_path <path_to_evolve_history.csv> \
#     --batch <batch_number> \
#     --mutation_index <mutation_number> \
#     --generation <gen_number_within_optimise> \
#     --individual_index <index_number_within_gen> \
#     --vsr_grid_dims 10 10 10 \
#     [--duration <seconds>] \
#     [--control_timestep <seconds>] \
#     [--headless]

# cli usage example for an evolve.py log file:

# python simulate.py \
#     --csv_path results_evolution/block_evo_mlp_run_A/full_evolution_history.csv \
#     --batch 5 \
#     --mutation_index 3 \
#     --generation 20 \
#     --individual_index 5 \
#     --vsr_grid_dims 10 10 10 \

# can use via cli, actual usage in optimise.py
if __name__ == "__main__":
    # STEP 1: Get arguments/paremeters
    parser = argparse.ArgumentParser(
        description="Run a single VSR simulation loaded from a history CSV log file."
    )

    # required arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the history CSV file (from optimise.py or evolve.py).",
    )
    parser.add_argument(
        "--generation",
        type=int,
        required=True,
        help="Generation number of the individual to simulate.",
    )
    parser.add_argument(
        "--individual_index",
        type=int,
        required=True,
        help="Index of the individual within the generation to simulate.",
    )
    parser.add_argument(
        "--vsr_grid_dims",
        type=int,
        nargs=3,
        required=True,  # required to build the VSR without original CSV
        help="Dimensions (X Y Z) of the voxel grid (e.g., 10 10 10). Must match the grid used during evolution/optimisation.",
    )

    # optional arguments (for identifying evolve.py logs)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch number (only needed for evolve.py logs).",
    )
    parser.add_argument(
        "--mutation_index",
        type=int,
        default=None,
        help="Mutation index within the batch (only needed for evolve.py logs).",
    )

    # simulation control args (override historical values if desired)
    parser.add_argument(
        "--duration",
        type=int,
        default=None,  # default to historical value if available
        help="Duration of the simulation in seconds (overrides value used in log). Default: Use historical duration (60s fallback).",
    )
    parser.add_argument(
        "--control_timestep",
        type=float,
        default=None,  # default to historical value if available
        help="Time step for control updates in seconds (overrides value used in log). Default: Use historical timestep (0.2s fallback).",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run simulation without the GUI viewer."
    )

    args = parser.parse_args()

    # STEP 2: Load data from .csv log file
    if not os.path.exists(args.csv_path):
        print(f"Error: History CSV file not found: {args.csv_path}")
        sys.exit(1)
    if (args.batch is not None and args.mutation_index is None) or (
        args.batch is None and args.mutation_index is not None
    ):
        print(
            "Error: --batch and --mutation_index must be provided together (or neither)."
        )
        sys.exit(1)
    if len(args.vsr_grid_dims) != 3 or any(d <= 0 for d in args.vsr_grid_dims):
        print("Error: --vsr_grid_dims requires 3 positive integers.")
        sys.exit(1)

    try:
        history_df = pd.read_csv(args.csv_path)

        # filter based on provided indices
        if args.batch is not None:  # evolve.py log
            # check if columns exist
            required_cols = [
                "batch",
                "mutation_index",
                "generation",
                "individual_index",
            ]
            if not all(col in history_df.columns for col in required_cols):
                print(
                    f"Error: CSV file '{args.csv_path}' is missing required columns for evolve log: {required_cols}"
                )
                sys.exit(1)
            target_row = history_df[
                (history_df["batch"] == args.batch)
                & (history_df["mutation_index"] == args.mutation_index)
                & (history_df["generation"] == args.generation)
                & (history_df["individual_index"] == args.individual_index)
            ]
        else:  # optimise.py log
            required_cols = ["generation", "individual_index"]
            if not all(col in history_df.columns for col in required_cols):
                print(
                    f"Error: CSV file '{args.csv_path}' is missing required columns for optimise log: {required_cols}"
                )
                sys.exit(1)
            target_row = history_df[
                (history_df["generation"] == args.generation)
                & (history_df["individual_index"] == args.individual_index)
            ]

        if target_row.empty:
            print(
                f"Error: No individual found in '{args.csv_path}' matching the provided indices."
            )
            print(
                f"  Batch: {args.batch}, Mutation: {args.mutation_index}, Gen: {args.generation}, Index: {args.individual_index}"
            )
            sys.exit(1)
        if len(target_row) > 1:
            print(
                f"Error: Multiple individuals found in '{args.csv_path}' matching the provided indices. Check the CSV."
            )
            sys.exit(1)

        # extract data from the unique row
        data = target_row.iloc[0]

        # parse necessary columns
        controller_type = data["controller_type"]
        gear_ratio = float(data["gear_ratio"])
        param_vector_str = data["params_vector_str"]
        mlp_hidden_str = data.get("mlp_plus_hidden_sizes_str", "[]")
        rnn_hidden_size = int(data.get("rnn_hidden_size", 0))
        voxel_coords_str = data["voxel_coords_str"]

        # safely parse string representations
        param_vector = np.array(parse_literal(param_vector_str))
        mlp_plus_hidden_sizes = parse_literal(mlp_hidden_str)
        voxel_coords_list = parse_literal(voxel_coords_str)

        # get simulation parameters (use cli overrides or historical/defaults)
        historical_control_ts = float(
            data.get("control_timestep", 0.2)
        )  # default fallback
        sim_duration = (
            args.duration if args.duration is not None else 60
        )  # simple fallback, historical duration not logged
        control_timestep = (
            args.control_timestep
            if args.control_timestep is not None
            else historical_control_ts
        )

        # validate parsed data
        if param_vector is None or not isinstance(param_vector, np.ndarray):
            raise ValueError("Failed to parse 'params_vector_str' into a numpy array.")
        if mlp_plus_hidden_sizes is None or not isinstance(mlp_plus_hidden_sizes, list):
            raise ValueError("Failed to parse 'mlp_plus_hidden_sizes_str' into a list.")
        if voxel_coords_list is None or not isinstance(voxel_coords_list, list):
            raise ValueError("Failed to parse 'voxel_coords_str' into a list.")
        if not all(isinstance(c, tuple) and len(c) == 3 for c in voxel_coords_list):
            raise ValueError("Parsed 'voxel_coords_list' does not contain 3D tuples.")

        print("\n--- Loaded Configuration from CSV Row ---")
        print(f"  Controller Type: {controller_type}")
        print(f"  Gear Ratio: {gear_ratio}")
        print(f"  MLP+ Hidden: {mlp_plus_hidden_sizes}")
        print(f"  RNN Hidden: {rnn_hidden_size}")
        print(f"  Num Voxels: {len(voxel_coords_list)}")
        print(f"  Parameter Vector Shape: {param_vector.shape}")

    except Exception as e:
        print(f"Error loading or parsing data from CSV: {e}")
        traceback.print_exc()
        sys.exit(1)

    # STEP 3: Setup VSR
    try:
        print("\nGenerating VSR structure and MuJoCo model...")
        vsr = VoxelRobot(*args.vsr_grid_dims, gear=gear_ratio)

        # clear the grid and set voxels based on loaded coordinates
        vsr.voxel_grid = np.zeros(args.vsr_grid_dims, dtype=np.uint8)
        for x, y, z in voxel_coords_list:
            # check bounds just in case coordinates are outside the specified grid dims
            if (
                0 <= x < args.vsr_grid_dims[0]
                and 0 <= y < args.vsr_grid_dims[1]
                and 0 <= z < args.vsr_grid_dims[2]
            ):
                vsr.set_val(x, y, z, 1)
            else:
                print(
                    f"Warning: Voxel coordinate ({x},{y},{z}) from CSV is outside specified grid dimensions {args.vsr_grid_dims}. Skipping."
                )

        # check contiguity after building
        vsr._check_contiguous()

        # generate XML string
        # create a dummy path for generate_model internal saving if needed
        temp_model_path = f"temp_simulate_model_{os.getpid()}"
        xml_string = vsr.generate_model(temp_model_path)
        # clean up dummy files generate_model might create
        try:
            if os.path.exists(temp_model_path + ".xml"):
                os.remove(temp_model_path + ".xml")
            if os.path.exists(temp_model_path + "_modded.xml"):
                os.remove(temp_model_path + "_modded.xml")
        except OSError:
            pass  # Ignore cleanup errors

        model = mujoco.MjModel.from_xml_string(xml_string)
        print("Model generated successfully.")

    except Exception as e:
        print(f"Error generating VSR or MuJoCo model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # STEP 4: Run simulations
    print("\n--- Running Simulation ---")
    print(f"  Duration: {sim_duration}s")
    print(f"  Control Timestep: {control_timestep}s")
    print(f"  Headless: {args.headless}")
    try:
        results = run_simulation(
            model=model,
            duration=sim_duration,
            control_timestep=control_timestep,
            controller_type=controller_type,
            mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
            rnn_hidden_size=rnn_hidden_size,
            param_vector=param_vector,
            headless=args.headless,
        )
        print("\n--- Simulation Results ---")
        print(f"  Fitness: {results[0]:.4f}")
        print(f"  Final X Distance to Target: {results[1]:.4f}")
        print(f"  Final Y Distance to Target: {results[2]:.4f}")
        print(f"  Target Reached: {results[3]}")

    except Exception as e:
        print("\n--- Simulation Failed ---")
        print(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)
