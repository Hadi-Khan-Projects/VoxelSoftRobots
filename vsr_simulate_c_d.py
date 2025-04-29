import mujoco
import mujoco.viewer
from vsr import VoxelRobot
from vsr_controller_distributed import DistributedNeuralController # Import the controller
import numpy as np
import math
import random
import time
import os

MODEL = "voxel_v1" # Example model name
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
DURATION = 6000  # seconds (reduced for quicker testing)
CONTROL_TIMESTEP = 0.005 # Apply control every 0.02 seconds (50 Hz)

# --- Simulation Setup ---
vsr = VoxelRobot(10, 10, 10) # Adjust size if needed for your model
vsr.load_model_csv(FILEPATH + ".csv")
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

for voxel in active_voxel_coords:
    if voxel not in voxel_tendon_map:
         print(f"Warning: Voxel {voxel} has motors but no mapped tendons.")
         # Handle this case if necessary, maybe remove voxel from active list
         continue # Skip assertion for now
    assert len(voxel_motor_map[voxel]) == 4, f"Voxel {voxel} has {len(voxel_motor_map[voxel])} motors, expected 4."
    assert len(voxel_tendon_map[voxel]) == 4, f"Voxel {voxel} has {len(voxel_tendon_map[voxel])} tendons, expected 4."


# --- Controller Setup ---
N_SENSORS_PER_VOXEL = 8 # 4 tendon lengths + 4 tendon velocities
N_COMM_CHANNELS = 2    # As per paper's experiments (nc=2)

# Select a driving voxel (e.g., the one with the lowest x, then y, then z)
# Make sure the chosen voxel actually exists in the loaded model
driving_voxel = active_voxel_coords[0] if active_voxel_coords else None

controller = DistributedNeuralController(
    n_voxels=n_active_voxels,
    voxel_coords=active_voxel_coords,
    n_sensors_per_voxel=N_SENSORS_PER_VOXEL,
    n_comm_channels=N_COMM_CHANNELS,
    driving_voxel_coord=driving_voxel
)

# --- Simulation Loop ---
scene_option = mujoco.MjvOption()
paused = False
last_control_time = 0.0
sim_step = 0

def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    # Configure viewer options
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

    while data.time < DURATION:
        sim_time = data.time
        step_start = time.time() # For performance monitoring

        # --- Control Step ---
        if sim_time >= last_control_time + CONTROL_TIMESTEP and not paused:
            # 1. Gather Sensor Data for ALL active voxels
            sensor_data_all = np.zeros((n_active_voxels, N_SENSORS_PER_VOXEL))
            for i, voxel_coord in enumerate(active_voxel_coords):
                if voxel_coord not in voxel_tendon_map: continue # Skip if no tendons mapped

                tendon_indices = voxel_tendon_map[voxel_coord]
                tendon_lengths = data.ten_length[tendon_indices]
                tendon_velocities = data.ten_velocity[tendon_indices]
                # Simple normalization/scaling might be needed here depending on expected ranges
                # For now, use raw values. Max length is sqrt(3) for unit cube diagonal.
                sensor_data_all[i, :4] = tendon_lengths
                sensor_data_all[i, 4:] = tendon_velocities

            # 2. Run Controller Step
            actuation_outputs = controller.step(sensor_data_all, sim_time) # Shape: (n_voxels, 1)

            # 3. Apply Actuation to Motors
            for i, voxel_coord in enumerate(active_voxel_coords):
                if voxel_coord not in voxel_motor_map: continue # Skip if no motors mapped

                mlp_actuation = actuation_outputs[i, 0] # Value in [-1, 1]
                # Map [-1, 1] to MuJoCo's ctrlrange [0, 1]
                # Simple mapping: apply same signal to all 4 motors
                # You might experiment with more complex mappings later
                motor_control_signal = (mlp_actuation + 1.0) / 2.0
                # print(motor_control_signal)
                motor_control_signal = np.clip(motor_control_signal, 0.0, 1.0) # Ensure it's within range

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
                break # Stop simulation on error
        # --------------------

        # --- Sync Viewer ---
        viewer.sync()
        # -------------------

        # Optional: Print step time for performance check
        # time_elapsed = time.time() - step_start
        # print(f"Step {sim_step} Time: {time_elapsed:.4f}s")


print("Simulation finished.")