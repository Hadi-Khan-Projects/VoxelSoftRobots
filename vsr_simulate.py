import mujoco
import mujoco.viewer
from vsr import VoxelRobot
import math
import random
import time
import os

MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
DURATION = 300  # seconds

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# Load, visualise, and then generate model
vsr.load_model_csv(FILEPATH + ".csv")
# vsr.visualise_model()
xml_string = vsr.generate_model(FILEPATH)

print("No. of vertexes: ", vsr.num_vertex())

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Create voxel-to-motor mapping
voxel_motor_map = {}

for i in range(model.nu):
    motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if motor_name:
        parts = motor_name.split("_")
        voxel_coord = tuple(map(int, parts[1:4]))  # Extract voxel (x, y, z)
        if voxel_coord not in voxel_motor_map:
            voxel_motor_map[voxel_coord] = []
        voxel_motor_map[voxel_coord].append(i)

# Ensure each voxel has exactly 4 motors
for voxel, motors in voxel_motor_map.items():
    assert len(motors) == 4, f"Voxel {voxel} has {len(motors)} motors, expected 4."

scene_option = mujoco.MjvOption()
paused = False

def key_callback(keycode):
    global paused
    if chr(keycode) == " ":
        paused = not paused

frequency = 0.1  # frequency in Hz

print(voxel_motor_map)
print(len(voxel_motor_map))

# Select some random voxels
NUM_ACTIVE = 80
active_voxels = list(voxel_motor_map.keys())[:80]

last_update_time = time.time()

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 1
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    
    while data.time < DURATION:
        current_time = time.time()
        
        # Set motor control for selected voxels
        for voxel in active_voxels:
            for motor_id in voxel_motor_map[voxel]:
                control_signal = 10 * math.sin(20.0 * math.pi * frequency * data.time)
                data.ctrl[motor_id] = control_signal
        
        # Update and print information every 0.1 seconds
        if current_time - last_update_time >= 0.1:
            last_update_time = current_time
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("Current signals of first 10 active motors:")
            for i, voxel in enumerate(active_voxels[:10]):
                motor_ids = voxel_motor_map[voxel]
                signals = [data.ctrl[motor_id] for motor_id in motor_ids]
                print(f"Voxel {voxel}: {signals}")
            
            print("\nCurrent contact points:")
            for i in range(data.ncon):
                contact = data.contact[i]
                pos = contact.pos  # Contact position
                vert_id = contact.vert[1]

                voxel = None
                for voxel_coord, motor_ids in voxel_motor_map.items():
                    if vert_id in motor_ids:  # Assuming vertex ID corresponds to motor ID (or similar)
                        voxel = voxel_coord
                        break
                
                print(f"Contact {i}: Position {pos}, vertex {vert_id}, voxel {voxel}")

        if not paused:
            mujoco.mj_step(model, data)
            viewer.sync()
