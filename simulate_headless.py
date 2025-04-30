from typing import Any
import mujoco
import mujoco.viewer
from vsr import VoxelRobot

# Ensure you're importing the correct controller file name
from controller import DistributedNeuralController
import numpy as np
import math
import random
import time
import os


def run_simulation_headless(
    model,
    duration: int,
    control_timestep: float,
    weights: np.ndarray[Any, np.dtype[np.float64]],
    biases: np.ndarray[Any, np.dtype[np.float64]],
):

    # --- Simulation Setup ---
    data = mujoco.MjData(model)

    # --- Voxel and Motor Mapping ---
    voxel_motor_map = {}
    voxel_tendon_map = {}  # Also map voxels to their tendon indices

    # Map motors
    for i in range(model.nu):
        motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if motor_name and motor_name.startswith("voxel_"):
            parts = motor_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))  # Extract voxel (x, y, z)
            if voxel_coord not in voxel_motor_map:
                voxel_motor_map[voxel_coord] = []
            voxel_motor_map[voxel_coord].append(i)

    # Map tendons (assuming tendon names match motor names structure)
    for i in range(model.ntendon):
        tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        if tendon_name and tendon_name.startswith("voxel_"):
            parts = tendon_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))
            if voxel_coord not in voxel_tendon_map:
                voxel_tendon_map[voxel_coord] = []
            voxel_tendon_map[voxel_coord].append(i)

    # Ensure mappings are consistent
    active_voxel_coords = sorted(list(voxel_motor_map.keys()))  # Get all defined voxels
    n_active_voxels = len(active_voxel_coords)

    # Filter active_voxel_coords to include only those with both motors and tendons mapped
    valid_active_voxel_coords = []
    for voxel in active_voxel_coords:
        has_motors = voxel in voxel_motor_map and len(voxel_motor_map[voxel]) == 4
        has_tendons = voxel in voxel_tendon_map and len(voxel_tendon_map[voxel]) == 4
        if has_motors and has_tendons:
            valid_active_voxel_coords.append(voxel)
        else:
            print(
                f"Warning: Voxel {voxel} skipped due to missing/incomplete motors or tendons."
            )

    active_voxel_coords = valid_active_voxel_coords
    n_active_voxels = len(active_voxel_coords)

    # --- Controller Setup ---
    N_SENSORS_PER_VOXEL = 8   # 4 tendon lengths + 4 tendon velocities
    N_COMM_CHANNELS = 2     # As per paper's experiments (nc=2)
    N_COMM_DIRECTIONS = 6   # <--- Updated: Now using 6 neighbors
    N_TIME_INPUTS = 2       # Number of sin(t)/cos(t) inputs

    # Select a driving voxel (e.g., the one with the lowest x, then y, then z among valid ones)
    driving_voxel = active_voxel_coords[0] if active_voxel_coords else None

    controller = DistributedNeuralController(
        n_voxels=n_active_voxels,
        voxel_coords=active_voxel_coords,  # Use the filtered list
        n_sensors_per_voxel=N_SENSORS_PER_VOXEL,
        n_comm_channels=N_COMM_CHANNELS,
        driving_voxel_coord=driving_voxel,
        time_signal_frequency=0.5,  # Example frequency for internal time signal
        weights=weights,
        biases=biases
    )

    # --- Simulation Loop ---
    scene_option = mujoco.MjvOption()
    paused = False
    last_control_time = 0.0
    sim_step = 0

    # Target range for rescaled motor signals
    TARGET_MIN_CTRL = 0.0
    TARGET_MAX_CTRL = 1.0
    TARGET_RANGE = TARGET_MAX_CTRL - TARGET_MIN_CTRL

    # Small value to prevent division by zero
    EPSILON = 1e-6

    def key_callback(keycode):
        global paused
        if chr(keycode) == " ":
            paused = not paused

    # Initial settling phase settings
    SETTLE_DURATION = 3.0  # Seconds to apply max contraction initially
    INITIAL_CTRL_VALUE = (
        1.0  # Contract tendons initially (0 = shortest length for motor)
    )
    target_reached = False

    print(f"Running simulation, voxels: {n_active_voxels}")

    while data.time < duration:
        sim_time = data.time
        step_start = time.time()

        # --- Initial Settling Phase ---
        if sim_time <= SETTLE_DURATION:
            for voxel_coord in active_voxel_coords:  # Iterate through valid voxels
                motor_indices = voxel_motor_map[voxel_coord]
                for motor_id in motor_indices:
                    data.ctrl[motor_id] = INITIAL_CTRL_VALUE
            last_control_time = sim_time  # Keep updating to prevent immediate controller activation after settling
        # --- Control Step ---
        elif (
            sim_time > SETTLE_DURATION
            and sim_time >= last_control_time + control_timestep
        ):
            # 1. Gather Sensor Data for ALL active voxels
            sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
            for i, voxel_coord in enumerate(active_voxel_coords):
                # No need to check mappings again, already filtered
                tendon_indices = voxel_tendon_map[voxel_coord]
                sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]
                # Consider adding sensor normalization/clipping here if needed

            # 2. Run Controller Step
            # actuation_outputs are in range [-1, 1]
            actuation_outputs = controller.step(sensor_data_all, sim_time)

            # 3. Initial Mapping to [0, 1]
            # initial_motor_signals shape: (n_voxels, 1)
            initial_motor_signals = (actuation_outputs + 1.0) / 2.0

            # 4. Apply Rescaled Actuation to Motors
            for i, voxel_coord in enumerate(active_voxel_coords):
                # Get the rescaled signal for this voxel
                # Clip ensures safety, though rescaling should keep it within [0,1] if target range is within [0,1]
                motor_control_signal = np.clip(initial_motor_signals[i, 0], 0.0, 1.0)

                motor_indices = voxel_motor_map[voxel_coord]
                for motor_id in motor_indices:
                    data.ctrl[motor_id] = motor_control_signal

            last_control_time = sim_time

            # use this instead of data.xpos[target_body_id, 0] because it gives centre of mass
            target_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "target"
            )
            robot_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "vsr"
            )
            target_x = data.subtree_com[target_body_id, 0]
            target_y = data.subtree_com[target_body_id, 1]
            robot_x = data.subtree_com[robot_body_id, 0]
            robot_y = data.subtree_com[robot_body_id, 1]
            x_dist_target = target_x - robot_x
            y_dist_target = target_y - robot_y

            if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                target_reached = True
        # --- End Control Step ---

        try:
            mujoco.mj_step(model, data)
        except mujoco.FatalError as e:
            print(f"MuJoCo Fatal Error: {e}. Time: {data.time}")
            # Simulation exploded, return worst fitness
            return -1000.0, x_dist_target, y_dist_target, target_reached

        # Optional: Print step time for performance check
        # time_elapsed = time.time() - step_start
        # print(f"Step {sim_step} Time: {time_elapsed:.4f}s")

    # euclidean final distance
    final_distance = math.sqrt(x_dist_target**2 + y_dist_target**2)
    fitness = -final_distance
    # if target_reached:
    #     fitness += 100 

    return fitness, x_dist_target, y_dist_target, target_reached

if __name__ == "__main__":
    MODEL = "quadruped_v3"  # Example model name
    model_path = f"vsr_models/{MODEL}/{MODEL}"
    duration = 60  # seconds (adjust as needed)
    control_timestep = 0.05  # Apply control every n seconds

    N_SENSORS_PER_VOXEL = 8   # 4 tendon lengths + 4 tendon velocities
    N_COMM_CHANNELS = 2     # As per paper's experiments (nc=2)
    N_COMM_DIRECTIONS = 6   # <--- Updated: Now using 6 neighbors
    N_TIME_INPUTS = 2       # Number of sin(t)/cos(t) inputs

    input_size = N_SENSORS_PER_VOXEL + N_COMM_DIRECTIONS * N_COMM_CHANNELS + 1 + N_TIME_INPUTS
    output_size = 1 + N_COMM_DIRECTIONS * N_COMM_CHANNELS
    weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
    biases = np.random.uniform(-0.1, 0.1, output_size)

    results = run_simulation_headless(model_path, duration, control_timestep, weights, biases)
    print(results)  # e.g (-58.846473979239164, 9.528210528960823, False)
