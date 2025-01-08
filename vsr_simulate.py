import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
# import numpy as np
import math
import xml.etree.ElementTree as ET
import re

# Create a 10*10*10 numpy grid, populate it with 0's
vsr = VoxelRobot(10, 10, 10)

filepath = "vsr_models/quadruped"

vsr.load_model(filepath+".parquet")
vsr.visualise_model()
point, element = vsr.generate_model()

TIMESTEP = "0.001"
duration = 300  # seconds
framerate = 60  # Hz

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
mujoco.mj_saveLastXML(filename=filepath+".xml", m=model)





original_xml_path = filepath + ".xml"
modified_xml_path = filepath + "_modded.xml"

# Parse original XML
tree = ET.parse(original_xml_path)
root = tree.getroot()

# Regex for vsr_n
vsr_body_pattern = re.compile(r"^vsr_(\d+)$")

# store newly named joints for actuator creation
all_joints = []  # Store tuples like (joint_name, body_index)

for body in root.findall("./worldbody/body"):
    body_name = body.get("name", "")
    match = vsr_body_pattern.match(body_name)
    if match:
        n = match.group(1)  # the number n
        # Find all joint elements in this body
        joints = body.findall("joint")
        if len(joints) == 3:
            # vsr_n_x
            joints[0].set("name", f"vsr_{n}_x")
            # vsr_n_y
            joints[1].set("name", f"vsr_{n}_y")
            # vsr_n_z
            joints[2].set("name", f"vsr_{n}_z")

            # Save them for later actuator creation
            all_joints.append((f"vsr_{n}_x", n))
            all_joints.append((f"vsr_{n}_y", n))
            all_joints.append((f"vsr_{n}_z", n))
        else:
            print(f"Warning: body {body_name} does not have exactly 3 joints.")

actuator = root.find("actuator")
if actuator is None:
    actuator = ET.SubElement(root, "actuator")

# Add a motor for each joint found
for joint_name, n in all_joints:
    motor = ET.SubElement(actuator, "motor")
    motor.set("name", f"{joint_name}_motor")
    motor.set("joint", joint_name)
    motor.set("gear", "1")

tree.write(modified_xml_path, encoding="utf-8", xml_declaration=True)

print(f"Modified XML saved to {modified_xml_path}")

model = mujoco.MjModel.from_xml_path(modified_xml_path)
data = mujoco.MjData(model)



scene_option = mujoco.MjvOption()

viewer = mujoco.viewer.launch_passive(model, data)

amplitude = 3      # desired amplitude of oscillation
frequency = 1.0      # frequency in Hz
phase = 1.0          # phase offset

start_time = time.time()
while data.time < duration:
    # For each vsr_n body, apply a sinusoidal command to its x, y, and z actuators.
    for n in range(vsr.num_vertex()):
        # Get the actuator IDs. You can get them by name:
        x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{n}_x_motor")
        y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{n}_y_motor")
        z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{n}_z_motor")
        
        # Compute a sinusoidal control signal. For example, let’s oscillate along the z-axis:
        control_signal_x = amplitude * math.sin(2*math.pi*frequency*data.time + phase)
        # control_signal_y = amplitude * math.sin(2*math.pi*frequency*data.time + phase + math.pi/2)
        # control_signal_z = amplitude * math.sin(2*math.pi*frequency*data.time + phase + math.pi)
        
        data.ctrl[x_motor_id] = control_signal_x
        # data.ctrl[y_motor_id] = control_signal_y
        # data.ctrl[z_motor_id] = control_signal_z

    # Step the simulation
    mujoco.mj_step(model, data)
    viewer.sync()
