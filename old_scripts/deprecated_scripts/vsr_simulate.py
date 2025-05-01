import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
import re

MODEL = "quadruped_v2"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
ORIGINAL_XML_PATH = FILEPATH + ".xml"
MODIFIED_XML_PATH = FILEPATH + "_modded.xml"
TIMESTEP = "0.001"
DURATION = 300  # seconds

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# load, visualise, and then generate model
vsr.load_model_csv(FILEPATH + ".csv")
# vsr.visualise_model()
point, element = vsr.generate_model()

print(vsr.num_vertex())

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
            radius="0.005" rgba="0.1 0.9 0.1 1" mass="{vsr.num_vertex()/10}">
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

model = mujoco.MjModel.from_xml_path(MODIFIED_XML_PATH)
data = mujoco.MjData(model)

scene_option = mujoco.MjvOption()

viewer = mujoco.viewer.launch_passive(model, data)

amplitude = 3.0  # desired amplitude of oscillation
frequency = 1.0  # frequency in Hz
phase = 1.0  # phase offset

start_time = time.time()
while data.time < DURATION:
    # For each vsr_n body, apply a sinusoidal command to its x, y, and z actuators.
    vsr.point_grid

    # for x, y, z in vsr.point_dict.keys():
    for key, phase in vsr.point_dict.items():
        x, y, z = key
        phase_x, phase_y, phase_z = phase
        x_motor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{x}_{y}_{z}_x_motor"
        )
        y_motor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{x}_{y}_{z}_y_motor"
        )
        z_motor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{x}_{y}_{z}_z_motor"
        )

        control_signal_x = amplitude * math.sin(
            2 * math.pi * frequency * data.time + phase_x
        )
        control_signal_y = amplitude * math.sin(
            2 * math.pi * frequency * data.time + phase_y
        )
        control_signal_z = amplitude * math.sin(
            2 * math.pi * frequency * data.time + phase_z
        )

        data.ctrl[x_motor_id] = control_signal_x
        # data.ctrl[y_motor_id] = control_signal_y
        # data.ctrl[z_motor_id] = control_signal_z

    # Step the simulation
    mujoco.mj_step(model, data)
    viewer.sync()
