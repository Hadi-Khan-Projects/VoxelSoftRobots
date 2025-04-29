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

MODEL = "voxel_v1" # Example model name
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
DURATION = 600  # seconds (adjust as needed)
# Apply control every n seconds
CONTROL_TIMESTEP = 0.2

# --- Simulation Setup ---
vsr = VoxelRobot(10, 10, 10) # Adjust size if needed for your model
vsr.load_model_csv(FILEPATH + ".csv")
# vsr.visualise_model()
xml_string = vsr.generate_model(FILEPATH)

print("VSR Model generated.")
print("No. of vertexes: ", vsr.num_vertex())

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
print(f"MuJoCo Model timestep: {model.opt.timestep}")

# --- Voxel and Motor Mapping ---
voxel_motor_map = {}
voxel_tendon_map = {} # Also map voxels to their tendon indices

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
active_voxel_coords = sorted(list(voxel_motor_map.keys())) # Get all defined voxels
n_active_voxels = len(active_voxel_coords)

print(f"Found {n_active_voxels} active voxels.")

# Filter active_voxel_coords to include only those with both motors and tendons mapped
valid_active_voxel_coords = []
for voxel in active_voxel_coords:
    has_motors = voxel in voxel_motor_map and len(voxel_motor_map[voxel]) == 4
    has_tendons = voxel in voxel_tendon_map and len(voxel_tendon_map[voxel]) == 4
    if has_motors and has_tendons:
        valid_active_voxel_coords.append(voxel)
    else:
        print(f"Warning: Voxel {voxel} skipped due to missing/incomplete motors or tendons.")

active_voxel_coords = valid_active_voxel_coords
n_active_voxels = len(active_voxel_coords)
print(f"Proceeding with {n_active_voxels} fully mapped active voxels.")


# --- Controller Setup ---
N_SENSORS_PER_VOXEL = 8 # 4 tendon lengths + 4 tendon velocities
N_COMM_CHANNELS = 2    # As per paper's experiments (nc=2)

# Select a driving voxel (e.g., the one with the lowest x, then y, then z among valid ones)
driving_voxel = active_voxel_coords[0] if active_voxel_coords else None

controller = DistributedNeuralController(
    n_voxels=n_active_voxels,
    voxel_coords=active_voxel_coords, # Use the filtered list
    n_sensors_per_voxel=N_SENSORS_PER_VOXEL,
    n_comm_channels=N_COMM_CHANNELS,
    driving_voxel_coord=driving_voxel,
    time_signal_frequency=0.5 # Example frequency for internal time signal
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
SETTLE_DURATION = 3.0 # Seconds to apply max contraction initially
INITIAL_CTRL_VALUE = 1.0 # Contract tendons initially (0 = shortest length for motor)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    # Configure viewer options
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

    # Disable rendering flags
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0

    # Cam settings
    viewer.cam.lookat[:] = [-30, 0, 2.5]
    viewer.cam.distance = 70
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -25

    while data.time < DURATION:
        sim_time = data.time
        step_start = time.time()

        # --- Initial Settling Phase ---
        if sim_time <= SETTLE_DURATION and not paused:
            for voxel_coord in active_voxel_coords: # Iterate through valid voxels
                 motor_indices = voxel_motor_map[voxel_coord]
                 for motor_id in motor_indices:
                     data.ctrl[motor_id] = INITIAL_CTRL_VALUE
            last_control_time = sim_time # Keep updating to prevent immediate controller activation after settling
        # --- Control Step ---
        elif sim_time > SETTLE_DURATION and sim_time >= last_control_time + CONTROL_TIMESTEP and not paused:
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

            # --- Global Rescaling --- START ---
            if n_active_voxels > 0:
                min_actual = np.min(initial_motor_signals)
                max_actual = np.max(initial_motor_signals)
                current_range = max_actual - min_actual

                if current_range < EPSILON:
                    # If all signals are the same, map them to the middle of the target range
                    rescaled_signals = np.full_like(initial_motor_signals, (TARGET_MIN_CTRL + TARGET_MAX_CTRL) / 2.0)
                else:
                    # Perform min-max scaling to the target range [TARGET_MIN_CTRL, TARGET_MAX_CTRL]
                    normalized_signals = (initial_motor_signals - min_actual) / current_range # Normalize to [0, 1]
                    rescaled_signals = normalized_signals * TARGET_RANGE + TARGET_MIN_CTRL # Scale to target range
            else:
                rescaled_signals = np.array([]) # Handle case with no active voxels
            # --- Global Rescaling --- END ---

            # 4. Apply Rescaled Actuation to Motors
            for i, voxel_coord in enumerate(active_voxel_coords):
                # Get the rescaled signal for this voxel
                # Clip ensures safety, though rescaling should keep it within [0,1] if target range is within [0,1]
                motor_control_signal = np.clip(rescaled_signals[i, 0], 0.0, 1.0)

                motor_indices = voxel_motor_map[voxel_coord]
                for motor_id in motor_indices:
                    data.ctrl[motor_id] = motor_control_signal

            last_control_time = sim_time
        # --- End Control Step ---

        # --- Physics Step ---
        if not paused:
            try:
                mujoco.mj_step(model, data)
                sim_step += 1
            except mujoco.FatalError as e:
                print(f"MuJoCo Fatal Error: {e}")
                # Consider logging data or saving state here before breaking
                break # Stop simulation on error
        # --------------------

        # --- Sync Viewer ---
        viewer.sync()
        # -------------------

        # Optional: Print step time for performance check
        # time_elapsed = time.time() - step_start
        # print(f"Step {sim_step} Time: {time_elapsed:.4f}s")


print("Simulation finished.")