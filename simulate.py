# simulate.py
import math
from typing import Any  # Keep Any for weight/bias type hints for now

import mujoco
import mujoco.viewer
import numpy as np

from controller import DistributedNeuralController  # Import the updated class
from vsr import VoxelRobot


def run_simulation(
    model,
    duration: int,
    control_timestep: float,
    # Parameters for the controller (can be None if param_vector is used)
    weights: np.ndarray[Any, np.dtype[np.float64]] = None,
    biases: np.ndarray[Any, np.dtype[np.float64]] = None,
    param_vector: np.ndarray = None,  # Add param_vector option
    # Controller configuration
    controller_type: str = "mlp",  # Default to original MLP
    mlp_plus_hidden_sizes: list = [32, 32],  # Default config
    rnn_hidden_size: int = 32,  # Default config
    # Other args
    headless: bool = True,
):
    data = mujoco.MjData(model)
    if not headless:
        print(f"MuJoCo Model timestep: {model.opt.timestep}")
        print(f"Controller Type: {controller_type}")
        if controller_type == "mlp_plus":
            print(f"  MLP+ Hidden: {mlp_plus_hidden_sizes}")
        if controller_type == "rnn":
            print(f"  RNN Hidden: {rnn_hidden_size}")

    # Voxel and motor mapping
    voxel_motor_map = {}
    voxel_tendon_map = {}  # map voxels to their tendon indices

    # map motors
    for i in range(model.nu):
        motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if motor_name and motor_name.startswith("voxel_"):
            parts = motor_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))
            if voxel_coord not in voxel_motor_map:
                voxel_motor_map[voxel_coord] = []
            voxel_motor_map[voxel_coord].append(i)

    # map tendons (tendon names match motor names structure)
    for i in range(model.ntendon):
        tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        if tendon_name and tendon_name.startswith("voxel_"):
            parts = tendon_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))
            if voxel_coord not in voxel_tendon_map:
                voxel_tendon_map[voxel_coord] = []
            voxel_tendon_map[voxel_coord].append(i)

    # ensure mappings are consistent
    active_voxel_coords = sorted(
        list(set(voxel_motor_map.keys()) | set(voxel_tendon_map.keys()))
    )  # get all defined voxels

    # filter active_voxel_coords to include only those with both motors and tendons mapped
    # just to be extra safe
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

    if not headless:
        print(f"Proceeding with {n_active_voxels} fully mapped active voxels.")
    else:
        print(".", end="", flush=True)

    # Get Vertex Body IDs for COM calculations
    # flex_body_names_str = model.flex_bodyid.map(lambda x: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, x))
    # iterate through potential names
    vertex_body_ids = []
    i = 0
    while True:
        body_name = f"vsr_{i}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            break
        vertex_body_ids.append(body_id)
        i += 1
    if not vertex_body_ids:
        raise ValueError("Could not find any vertex body IDs")
    vertex_body_ids = np.array(vertex_body_ids)

    # Controller Setup
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
        # Parameter initialisation: Vector takes precedence
        param_vector=param_vector,
        weights=weights,  # Only used if vector is None and type is mlp
        biases=biases,  # Only used if vector is None and type is mlp
        # Network specific config
        mlp_plus_hidden_sizes=mlp_plus_hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        # General config
        driving_voxel_coord=driving_voxel,
        time_signal_frequency=0.5,
    )

    # ACTUAL SIMULATION LOOP
    paused = False
    last_control_time = 0.0
    target_reached = False
    x_dist_target = np.inf  # Initialize distances
    y_dist_target = np.inf

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    SETTLE_DURATION = 3.0  # Seconds to apply max contraction initially
    INITIAL_CTRL_VALUE = (
        1.0  # Contract tendons initially (0 = shortest length for motor)
    )

    controller.reset()  # needed for RNN and communication init

    if not headless:
        # --- Viewer Setup ---
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

                # Settling phase, no movement until VSR settles
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
                    # 1. Gather sensors data for all active voxels
                    sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        tendon_indices = voxel_tendon_map[voxel_coord]
                        sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                        sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]

                    # 2. Calculate Global State and Target Info
                    current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)
                    current_com_vel = np.mean(
                        data.cvel[vertex_body_ids][:, 3:], axis=0
                    )  # Linear velocity

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

                    target_pos = data.subtree_com[target_body_id]
                    # Use COM pos calculated earlier for robot position
                    # robot_pos = data.subtree_com[robot_body_id] # This might be less accurate than vertex mean

                    # distances
                    x_dist_target = target_pos[0] - current_com_pos[0]
                    y_dist_target = target_pos[1] - current_com_pos[1]

                    # orientation
                    vec_to_target = target_pos - current_com_pos
                    vec_to_target_xy = vec_to_target[[0, 1]]
                    norm = np.linalg.norm(vec_to_target_xy)
                    target_orientation_vector = vec_to_target_xy / (norm + 1e-6)

                    if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                        target_reached = True

                    # 3. Run controller step
                    # actuation_outputs are in range [-1, 1]
                    actuation_outputs = controller.step(
                        sensor_data_all,
                        sim_time,
                        current_com_vel,
                        target_orientation_vector,
                    )

                    # 4. Initial mapping to [0, 1]
                    # initial_motor_signals shape: (n_voxels, 1)
                    initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                    # 5. Apply clipped actuation to motors
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        motor_control_signal = np.clip(
                            initial_motor_signals[i, 0], 0.0, 1.0
                        )
                        for motor_id in voxel_motor_map[voxel_coord]:
                            data.ctrl[motor_id] = motor_control_signal

                    last_control_time = sim_time
                # ENDOF Control Step

                if not paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

    else:  # headless execution
        while data.time < duration:
            sim_time = data.time

            # Settling phase, no movement until VSR settles
            if sim_time <= SETTLE_DURATION:
                for voxel_coord in active_voxel_coords:
                    for motor_id in voxel_motor_map[voxel_coord]:
                        data.ctrl[motor_id] = INITIAL_CTRL_VALUE
                last_control_time = sim_time

            elif (
                sim_time > SETTLE_DURATION
                and sim_time >= last_control_time + control_timestep
            ):
                # 1. Gather sensor data for ALL active voxels
                sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
                for i, voxel_coord in enumerate(active_voxel_coords):
                    tendon_indices = voxel_tendon_map[voxel_coord]
                    sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                    sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]

                # 2. Calculate target distances and orientation
                current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)

                # distance
                # use cvel which includes angular; take linear part [3:6]
                current_com_vel = np.mean(data.cvel[vertex_body_ids][:, 3:], axis=0)
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

                # 3. Run controller step
                actuation_outputs = controller.step(
                    sensor_data_all,
                    sim_time,
                    current_com_vel,
                    target_orientation_vector,
                )

                # 4. Initial mapping to [0, 1]
                # initial_motor_signals shape: (n_voxels, 1)
                initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                # 5. Apply clipped actuation to motors
                for i, voxel_coord in enumerate(active_voxel_coords):
                    motor_control_signal = np.clip(
                        initial_motor_signals[i, 0], 0.0, 1.0
                    )
                    for motor_id in voxel_motor_map[voxel_coord]:
                        data.ctrl[motor_id] = motor_control_signal

                last_control_time = sim_time
            # ENDOF Control Step

            try:
                mujoco.mj_step(model, data)
            except mujoco.FatalError as e:
                print(f"MuJoCo Fatal Error: {e}. Time: {data.time}")
                return -np.inf, np.inf, np.inf, False  # Use -inf for fitness on crash

    # --- Fitness Calculation ---
    # euclidean final distance from last control step
    final_distance = (
        math.sqrt(x_dist_target**2 + y_dist_target**2)
        if np.isfinite(x_dist_target) and np.isfinite(y_dist_target)
        else np.inf
    )
    fitness = (
        -final_distance if np.isfinite(final_distance) else -np.inf
    )  # Penalize instability/NaNs

    # ensure finite returns for logging
    x_dist_target = x_dist_target if np.isfinite(x_dist_target) else np.inf
    y_dist_target = y_dist_target if np.isfinite(y_dist_target) else np.inf

    return fitness, x_dist_target, y_dist_target, target_reached


# example on how to use, actual usage in optimise.py
if __name__ == "__main__":
    MODEL = "quadruped_v3"  # test model from vsr_model
    model_path = f"vsr_models/{MODEL}/{MODEL}"
    DURATION = 60
    CONTROL_TIMESTEP = 0.2
    GEAR = 100
    VSR_DIMS = (10, 10, 10)

    vsr = VoxelRobot(*VSR_DIMS, gear=GEAR)
    try:
        vsr.load_model_csv(model_path + ".csv")
        xml_string = vsr.generate_model(model_path)
        model = mujoco.MjModel.from_xml_string(xml_string)
        print("Model loaded/generated.")
    except Exception as e:
        print(f"Error loading/generating model: {e}")
        exit()

    # --- Test different controller types ---
    for ctrl_type in ["mlp", "mlp_plus", "rnn"]:
        print(f"\n--- Testing Controller Type: {ctrl_type} ---")

        # determine param vector size for random init
        # instantiate temporarily just to get size
        temp_controller = DistributedNeuralController(
            controller_type=ctrl_type,
            n_voxels=1,  # dummy, actual needed later
            voxel_coords=[(0, 0, 0)],  # dummy coords for init
            n_sensors_per_voxel=8,
            n_comm_channels=2,
            # default configs even if not used by this type
            mlp_plus_hidden_sizes=[16],
            rnn_hidden_size=16,
        )
        param_size = temp_controller.get_total_parameter_count()
        print(f"Parameter vector size for {ctrl_type}: {param_size}")

        # generate random parameters for testing
        random_params = np.random.uniform(-0.5, 0.5, param_size)

        results = run_simulation(
            model=model,
            duration=DURATION,
            control_timestep=CONTROL_TIMESTEP,
            param_vector=random_params,  # use param_vector input
            controller_type=ctrl_type,
            # pass configs used for size calculation if type matches
            mlp_plus_hidden_sizes=[16] if ctrl_type == "mlp_plus" else [],
            rnn_hidden_size=16 if ctrl_type == "rnn" else 0,
            headless=False,  # show viewer for testing
        )
        print(
            f"Results for {ctrl_type}: {results}"
        )  # (fitness, x_dist, y_dist, reached)
