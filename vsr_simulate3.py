import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
import re
import random

MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
ORIGINAL_XML_PATH = FILEPATH + ".xml"
MODIFIED_XML_PATH = FILEPATH + "_modded.xml"
TIMESTEP = "0.001"
DURATION = 15.0  # shorter duration for repeated trials
NUM_ITERATIONS = 5

# Step 1: Build your VoxelRobot + parse CSV, etc.
vsr = VoxelRobot(10, 10, 10)
vsr.load_model(FILEPATH + ".csv")

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# load, visualise, and then generate model
vsr.load_model(FILEPATH + ".csv")
# vsr.visualise_model()
point, element = vsr.generate_model()

xml_string = f"""
<mujoco>
    <compiler autolimits="true"/>
    <include file="scene.xml"/>
    <compiler autolimits="true"/>
    <option solver="Newton" tolerance="1e-6" timestep="{TIMESTEP}" integrator="implicitfast"/>
    <size memory="2000M"/>

    <worldbody>
        <flexcomp name="vsr" type="direct" dim="3"
            point="{point}"
            element="{element}"
            radius="0.005" rgba="0.1 0.9 0.1 1" mass="30">
            <contact condim="3" solref="0.01 1" solimp="0.95 0.99 0.0001" selfcollide="none"/>
            <edge damping="1"/>
            <elasticity young="5e2" poisson="0.3"/>
        </flexcomp>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Save model as XML for reference
mujoco.mj_saveLastXML(filename=FILEPATH + ".xml", m=model)

# Parse original XML
tree = ET.parse(ORIGINAL_XML_PATH)
root = tree.getroot()

# Regex for vsr_n
vsr_body_pattern = re.compile(r"^vsr_(\d+)$")

# store newly named joints for actuator creation
all_joints = []  # Store tuples like (joint_name, body_pos)

for body in root.findall("./worldbody/body"):
    body_pos = body.get("pos", "")
    if body_pos:
        pos_str = body_pos.replace(" ", "_")
        # Find all joint elements in this body
        joints = body.findall("joint")
        if len(joints) == 3:
            # vsr_pos_x
            joints[0].set("name", f"vsr_{pos_str}_x")
            # vsr_pos_y
            joints[1].set("name", f"vsr_{pos_str}_y")
            # vsr_pos_z
            joints[2].set("name", f"vsr_{pos_str}_z")

            # Save them for later actuator creation
            all_joints.append((f"vsr_{pos_str}_x", pos_str))
            all_joints.append((f"vsr_{pos_str}_y", pos_str))
            all_joints.append((f"vsr_{pos_str}_z", pos_str))
        else:
            print(
                f"Warning: body at position {body_pos} does not have exactly 3 joints."
            )

actuator = root.find("actuator")
if actuator is None:
    actuator = ET.SubElement(root, "actuator")

# Add a motor for each joint found
for joint_name, pos_str in all_joints:
    motor = ET.SubElement(actuator, "motor")
    motor.set("name", f"{joint_name}_motor")
    motor.set("joint", joint_name)
    motor.set("gear", "1")

tree.write(MODIFIED_XML_PATH, encoding="utf-8", xml_declaration=True)

print(f"Modified XML saved to {MODIFIED_XML_PATH}")

# Step 2: Build simulation from the (already) modified XML
def build_simulation(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data

model, data = build_simulation(MODIFIED_XML_PATH)

# Step 3: Define our runner function
def run_simulation(model, data, vsr, phases, duration=5.0, amplitude=0.5, frequency=1.0):
    mujoco.mj_resetData(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vsr_0")
    init_pos = data.xpos[body_id].copy()

    while data.time < duration:
        for key, (phase_x, phase_y, phase_z) in phases.items():
            x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor")
            y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor")
            z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor")

            # apply sinusoidal control
            if x_motor_id >= 0:
                data.ctrl[x_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_x)
            if y_motor_id >= 0:
                data.ctrl[y_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_y)
            if z_motor_id >= 0:
                data.ctrl[z_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_z)

        mujoco.mj_step(model, data)
    
    final_pos = data.xpos[body_id]
    dist = math.sqrt((final_pos[0] - init_pos[0])**2 + (final_pos[1] - init_pos[1])**2 + (final_pos[2] - init_pos[2])**2)
    return dist

# Step 4: Simple random search over phases
def random_search_for_phases(vsr, model, data, num_iterations=20, duration=5.0):
    best_distance = -1e6
    best_phases = {}

    for i in range(num_iterations):
        # random phases in [0, 2*pi)
        phases = {}
        for key in vsr.point_dict.keys():
            phases[key] = (
                random.random()*2*math.pi,
                random.random()*2*math.pi,
                random.random()*2*math.pi
            )
        
        dist = run_simulation(model, data, vsr, phases, duration=duration)

        if dist > best_distance:
            best_distance = dist
            best_phases = phases
            print(f"[{i}] distance = {dist:.4f}, new best!")
        else:
            print(f"[{i}] distance = {dist:.4f}")

    return best_phases, best_distance

best_phases, best_dist = random_search_for_phases(vsr, model, data,
                                                  num_iterations=NUM_ITERATIONS,
                                                  duration=DURATION)

print(f"Random search complete. Best distance traveled: {best_dist:.4f}")

# Step 5: Optionally re-run a final simulation with best phases and watch in viewer
mujoco.mj_resetData(model, data)
viewer = mujoco.viewer.launch_passive(model, data)

start_time = time.time()
while (time.time() - start_time) < 10.0:  # for example, watch for 10 seconds
    for key, (phase_x, phase_y, phase_z) in best_phases.items():
        x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor")
        y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor")
        z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor")

        if x_motor_id >= 0:
            data.ctrl[x_motor_id] = 0.5 * math.sin(2*math.pi*1.0*data.time + phase_x)
        if y_motor_id >= 0:
            data.ctrl[y_motor_id] = 0.5 * math.sin(2*math.pi*1.0*data.time + phase_y)
        if z_motor_id >= 0:
            data.ctrl[z_motor_id] = 0.5 * math.sin(2*math.pi*1.0*data.time + phase_z)

    mujoco.mj_step(model, data)
    viewer.sync()

print("Done watching final best policy.")