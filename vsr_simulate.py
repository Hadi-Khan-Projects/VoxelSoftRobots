import mujoco
import mujoco.viewer
from vsr import VoxelRobot
import math
import random

MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
DURATION = 300  # seconds

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# Load, visualise, and then generate model
vsr.load_model_csv(FILEPATH + ".csv")
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
    if chr(keycode) == " ":
        global paused
        paused = not paused


frequency = 0.01  # frequency in Hz

# print(voxel_motor_map)
# print(len(voxel_motor_map))

# Select some random voxels
NUM_ACTIVE = 7
active_voxels = random.sample(list(voxel_motor_map.keys()), NUM_ACTIVE)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while data.time < DURATION:
        # Set motor control for selected voxels
        for voxel in active_voxels:
            for motor_id in voxel_motor_map[voxel]:
                data.ctrl[motor_id] = 5 * math.sin(
                    20.0 * math.pi * frequency * data.time
                )

        if not paused:
            mujoco.mj_step(model, data)
            viewer.sync()
