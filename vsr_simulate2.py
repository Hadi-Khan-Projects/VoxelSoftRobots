import mujoco
import mujoco.viewer
from vsr import VoxelRobot
import math
import random
import time
import os

MODEL = "quadruped_v2"
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
# for example (4, 1, 2) : [336, 337, 338, 339]
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
    viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0    

    viewer.cam.lookat[:] = [-10, 0, 2.5]  # Look at the target
    viewer.cam.distance = 40  # move back amount
    viewer.cam.azimuth = 90  # rotations
    viewer.cam.elevation = -25  # angle

    # use this instead of data.xpos[target_body_id, 0] because it gives centre of mass
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vsr")
    target_x = data.subtree_com[target_body_id, 0]
    target_y = data.subtree_com[target_body_id, 1]
    robot_x = data.subtree_com[robot_body_id, 0]
    robot_y = data.subtree_com[robot_body_id, 1]
    x_dist_target = target_x - robot_x
    y_dist_target = target_y - robot_y

    # FEATURES
    vert_contacts = {}
    x_dist_target_prev = x_dist_target
    y_dist_target_prev = y_dist_target

    print("\nFinding initial contact points:")
    for i in range(data.ncon):
        contact = data.contact[i]
        pos = contact.pos  # Contact position
        vert_id = contact.vert[1]

        voxel = None
        for voxel_coord, motor_ids in voxel_motor_map.items():
            if (
                vert_id in motor_ids
            ):  # Assuming vertex ID corresponds to motor ID (or similar)
                voxel = voxel_coord
                break

        print(f"Contact {i}: Position {pos}, vertex {vert_id}, voxel {voxel}")
        vert_contacts[vert_id] = (True, 0.0)

    while data.time < DURATION:
        current_time = time.time()

        # MOTOR
        for voxel in active_voxels:
            for motor_id in voxel_motor_map[voxel]:
                control_signal = 10 * math.sin(20.0 * math.pi * frequency * data.time)
                data.ctrl[motor_id] = control_signal

        # every 0.1 seconds
        if current_time - last_update_time >= 0.1:
            last_update_time = current_time
            os.system("cls" if os.name == "nt" else "clear")

            # CONTACTS TRUE/FALSE + CONTACTS HEIGHT
            for vert_id in vert_contacts.keys():
                vert_contacts[vert_id] = (False, data.xpos[vert_id, 2])

            for i in range(data.ncon):
                contact = data.contact[i]
                pos = contact.pos  # Contact position
                vert_id = contact.vert[1]

                voxel = None
                for voxel_coord, motor_ids in voxel_motor_map.items():
                    if vert_id in motor_ids:
                        voxel = voxel_coord
                        break

                # z height typically between -0.01 and 10.0, but we enforce
                if (pos[2] <= 0.0):
                    vert_contacts[vert_id] = (True, 0.0)
                elif (pos[2] >= 10.0):
                    vert_contacts[vert_id] = (True, 10.0)
                else:
                    vert_contacts[vert_id] = (True, pos[2])

            for vert_id, contact in vert_contacts.items():
                print(
                    f"vertex: {vert_id}, in contact: {contact[0]}, height: {contact[1]}"
                )

            # X,Y DISTANCE FROM TARGET (<body name="target">)
            target_x = data.subtree_com[target_body_id, 0]
            target_y = data.subtree_com[target_body_id, 1]
            robot_x = data.subtree_com[robot_body_id, 0]
            robot_y = data.subtree_com[robot_body_id, 1]

            print(f"Target: {target_x}, {target_y}, Robot: {robot_x}, {robot_y}")

            x_dist_target = -(target_x - robot_x)
            y_dist_target = -(target_y - robot_y)
            print(
                f"Distance to Target: X = {x_dist_target:.2f}, Y = {y_dist_target:.2f}"
            )

            # change typically between -0.5 to 0.5
            change_in_x = x_dist_target - x_dist_target_prev
            change_in_y = y_dist_target - y_dist_target_prev
            print(
                f"Change in distance: X = {change_in_x:.2f}, Y = {change_in_y:.2f}"
            )

            x_dist_target_prev = x_dist_target
            y_dist_target_prev = y_dist_target

        # TARGET TOUCHED?
        # Temp, needs proper contact check
        if abs(x_dist_target) < 10 and abs(y_dist_target) < 10:
            break

        if not paused:
            mujoco.mj_step(model, data)
            viewer.sync()
