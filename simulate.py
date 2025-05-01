import math
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from controller import DistributedNeuralController
from vsr import VoxelRobot


def run_simulation(
    model,
    duration: int,
    control_timestep: float,
    weights: np.ndarray[Any, np.dtype[np.float64]],
    biases: np.ndarray[Any, np.dtype[np.float64]],
    headless: bool = True,
):
    data = mujoco.MjData(model)
    if not headless:
        print(f"MuJoCo Model timestep: {model.opt.timestep}")

    # Voxel and motor mapping
    voxel_motor_map = {}
    voxel_tendon_map = {}  # map voxels to their tendon indices

    # map motors
    for i in range(model.nu):
        motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if motor_name and motor_name.startswith("voxel_"):
            parts = motor_name.split("_")
            voxel_coord = tuple(map(int, parts[1:4]))  # extract voxel (x, y, z)
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
    active_voxel_coords = sorted(list(voxel_motor_map.keys()))  # get all defined voxels
    n_active_voxels = len(active_voxel_coords)

    if not headless:
        print(f"Found {n_active_voxels} active voxels.")

    # filter active_voxel_coords to include only those with both motors and tendons mapped
    # just to be extra safe
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

    if not headless:
        print(f"Proceeding with {n_active_voxels} fully mapped active voxels.")
    else:
        print(".", end="", flush=True)

    # Get Vertex Body IDs for COM calculations
    vertex_body_ids = []
    # flex_body_names_str = model.flex_bodyid.map(lambda x: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, x)) 
    # iterate through potential names
    vertex_body_ids = []
    i = 0
    while True:
        body_name = f"vsr_{i}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1: # No more bodies found with this pattern
            break
        vertex_body_ids.append(body_id)
        i += 1

    if not vertex_body_ids:
        raise ValueError("Could not find any vertex body IDs (e.g., 'vsr_0', 'vsr_1', ...)")
    vertex_body_ids = np.array(vertex_body_ids) # Convert to numpy array for easier indexing
    if not headless:
        print(f"Found {len(vertex_body_ids)} vertex bodies for COM calculations.")

    # Controller Setup
    N_SENSORS_PER_VOXEL = 8  # 4 tendon lengths + 4 tendon velocities
    N_COMM_CHANNELS = 2  # as per paper's experiments (nc=2)
    # N_COMM_DIRECTIONS = 6  # voxels have 6 neighbors
    # N_TIME_INPUTS = 2  # number of sin(t)/cos(t) inputs

    # driving voxel (the one with the lowest x, then y, then z among valid ones)
    driving_voxel = active_voxel_coords[0] if active_voxel_coords else None

    controller = DistributedNeuralController(
        n_voxels=n_active_voxels,
        voxel_coords=active_voxel_coords,  # using filtered list
        n_sensors_per_voxel=N_SENSORS_PER_VOXEL,
        n_comm_channels=N_COMM_CHANNELS,
        driving_voxel_coord=driving_voxel,
        time_signal_frequency=0.5,  # adjustable
        weights=weights,
        biases=biases,
    )

    # ACTUAL SIMULATION LOOP
    # scene_option = mujoco.MjvOption()
    paused = False
    last_control_time = 0.0

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    # Initial settling phase settings
    SETTLE_DURATION = 3.0  # Seconds to apply max contraction initially
    INITIAL_CTRL_VALUE = (
        1.0  # Contract tendons initially (0 = shortest length for motor)
    )
    target_reached = False

    if not headless:
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

            # disable some rendering flags (performance
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
                        motor_indices = voxel_motor_map[voxel_coord]
                        for motor_id in motor_indices:
                            data.ctrl[motor_id] = INITIAL_CTRL_VALUE
                    last_control_time = sim_time  # keep updating to prevent immediate controller activation after settling
                # Control step
                elif (
                    sim_time > SETTLE_DURATION
                    and sim_time >= last_control_time + control_timestep
                    and not paused
                ):
                    # 1. Gather Sensor Data for ALL active voxels
                    sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        # No need to check mappings again, already filtered
                        tendon_indices = voxel_tendon_map[voxel_coord]
                        sensor_data_all[i, :4] = data.ten_length[tendon_indices]
                        sensor_data_all[i, 4:] = data.ten_velocity[tendon_indices]
                    
                    if len(vertex_body_ids) > 0:
                        current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)
                        # Use cvel which includes angular; take linear part [3:6]
                        current_com_vel = np.mean(data.cvel[vertex_body_ids][:, 3:], axis=0)
                    else: # Fallback if no vertex bodies found (should not happen)
                        print("Warning: No vertex bodies found for COM calculations. Using zeros.")
                        current_com_pos = np.zeros(3)
                        current_com_vel = np.zeros(3)

                    # 2. Calculate target distances and orientation

                    # distance
                    # this instead of data.xpos[target_body_id, 0] because it gives centre of mass
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

                    # orientation
                    target_pos = data.subtree_com[target_body_id] # Use subtree_com for target
                    vec_to_target = target_pos - current_com_pos
                    vec_to_target_xy = vec_to_target[[0, 1]] # Project onto X and Y
                    norm = np.linalg.norm(vec_to_target_xy)
                    target_orientation_vector = vec_to_target_xy / (norm + 1e-6) # Normalize, add epsilon for safety

                    if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                        target_reached = True

                    # 3. Run controller step
                    # actuation_outputs are in range [-1, 1]
                    actuation_outputs = controller.step(sensor_data_all, sim_time, current_com_vel, target_orientation_vector)

                    # 4. Initial mapping to [0, 1]
                    # initial_motor_signals shape: (n_voxels, 1)
                    initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                    # 5. Apply Clipped actuation to motors
                    for i, voxel_coord in enumerate(active_voxel_coords):
                        # clip to be extra safe
                        motor_control_signal = np.clip(
                            initial_motor_signals[i, 0], 0.0, 1.0
                        )

                        motor_indices = voxel_motor_map[voxel_coord]
                        for motor_id in motor_indices:
                            data.ctrl[motor_id] = motor_control_signal

                    last_control_time = sim_time
                # ENDOF Control Step

                if not paused:
                    mujoco.mj_step(model, data)
                    viewer.sync()

                viewer.sync()
    else:  # headless execution
        while data.time < duration:
            sim_time = data.time

            # Settling phase, no movement until VSR settles
            if sim_time <= SETTLE_DURATION:
                for voxel_coord in active_voxel_coords:  # iterate through valid voxels
                    motor_indices = voxel_motor_map[voxel_coord]
                    for motor_id in motor_indices:
                        data.ctrl[motor_id] = INITIAL_CTRL_VALUE
                last_control_time = sim_time  # keep updating to prevent immediate controller activation after settling
            # Control step
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
                
                if len(vertex_body_ids) > 0:
                    current_com_pos = np.mean(data.xpos[vertex_body_ids], axis=0)
                    # Use cvel which includes angular; take linear part [3:6]
                    current_com_vel = np.mean(data.cvel[vertex_body_ids][:, 3:], axis=0)
                else: # Fallback if no vertex bodies found (should not happen)
                    print("Warning: No vertex bodies found for COM calculations. Using zeros.")
                    current_com_pos = np.zeros(3)
                    current_com_vel = np.zeros(3)

                # 2. Calculate target distances and orientation

                # distance
                # this instead of data.xpos[target_body_id, 0] because it gives centre of mass
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

                # orientation
                target_pos = data.subtree_com[target_body_id] # Use subtree_com for target
                vec_to_target = target_pos - current_com_pos
                vec_to_target_xy = vec_to_target[[0, 1]] # Project onto X and Y
                norm = np.linalg.norm(vec_to_target_xy)
                target_orientation_vector = vec_to_target_xy / (norm + 1e-6) # Normalize, add epsilon for safety

                if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
                    target_reached = True

                # 3. Run controller step
                # actuation_outputs are in range [-1, 1]
                actuation_outputs = controller.step(sensor_data_all, sim_time, current_com_vel, target_orientation_vector)

                # 4. Initial mapping to [0, 1]
                # initial_motor_signals shape: (n_voxels, 1)
                initial_motor_signals = (actuation_outputs + 1.0) / 2.0

                # 5. Apply Clipped actuation to motors
                for i, voxel_coord in enumerate(active_voxel_coords):
                    # clip to be extra safe
                    motor_control_signal = np.clip(
                        initial_motor_signals[i, 0], 0.0, 1.0
                    )

                    motor_indices = voxel_motor_map[voxel_coord]
                    for motor_id in motor_indices:
                        data.ctrl[motor_id] = motor_control_signal

                last_control_time = sim_time
            # ENDOF Control Step

            try:
                mujoco.mj_step(model, data)
            except mujoco.FatalError as e:
                print(f"MuJoCo Fatal Error: {e}. Time: {data.time}")
                # Simulation crashed, return worst fitness
                return -1000.0, x_dist_target, y_dist_target, target_reached

    # euclidean final distance
    final_distance = math.sqrt(x_dist_target**2 + y_dist_target**2)
    fitness = -final_distance

    return fitness, x_dist_target, y_dist_target, target_reached


# example on how to use, actual usage in optimise.py
if __name__ == "__main__":
    MODEL = "quadruped_v2"  # test model from vsr_model
    model_path = f"vsr_models/{MODEL}/{MODEL}"
    DURATION = 60  # seconds
    CONTROL_TIMESTEP = 0.05  # Apply control every n seconds
    GEAR = 100

    # Simulation setup
    vsr = VoxelRobot(10, 10, 10, gear=GEAR)  # adjustable
    vsr.load_model_csv(model_path + ".csv")
    # vsr.visualise_model()
    xml_string = vsr.generate_model(model_path)

    print("VSR Model generated.")
    print("No. of vertexes: ", vsr.num_vertex())

    model = mujoco.MjModel.from_xml_string(xml_string)

    N_SENSORS_PER_VOXEL = 8  # 4 tendon lengths + 4 tendon velocities
    N_COMM_CHANNELS = 2  # as per paper's experiments (nc=2)
    N_COMM_DIRECTIONS = 6  # voxels have 6 neighbors
    N_TIME_INPUTS = 2  # Number of sin(t)/cos(t) inputs

    input_size = (
        N_SENSORS_PER_VOXEL + N_COMM_DIRECTIONS * N_COMM_CHANNELS + 1 + N_TIME_INPUTS
    )
    output_size = 1 + N_COMM_DIRECTIONS * N_COMM_CHANNELS
    weights = np.random.uniform(-0.1, 0.1, (output_size, input_size))
    biases = np.random.uniform(-0.1, 0.1, output_size)

    results = run_simulation(
        model, DURATION, CONTROL_TIMESTEP, weights, biases, headless=False
    )
    print(results)  # e.g (-58.846473979239164, 9.528210528960823, False)
