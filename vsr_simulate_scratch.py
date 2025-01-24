import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
import re

MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
ORIGINAL_XML_PATH = FILEPATH + ".xml"
MODIFIED_XML_PATH = FILEPATH + "_modded.xml"
TIMESTEP = "0.001"
DURATION = 300  # seconds

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# load, visualise, and then generate model
vsr.load_model(FILEPATH + ".csv")
# vsr.visualise_model()
point, element = vsr.generate_model()

print(vsr.num_vertex())

# ----------------------------------------------------------------------
# 1) Create a MuJoCo string defining a flexcomp from your generated points/elements
#    (the same as before).
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 2) Parse ORIGINAL_XML_PATH and rename the existing body joints, etc.
#    (the same logic as before).
# ----------------------------------------------------------------------
tree = ET.parse(ORIGINAL_XML_PATH)
root = tree.getroot()

# Regex for vsr_n
vsr_body_pattern = re.compile(r"^vsr_(\d+)$")

# Store newly named joints for actuator creation
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

            all_joints.append((f"vsr_{pos_str}_x", pos_str))
            all_joints.append((f"vsr_{pos_str}_y", pos_str))
            all_joints.append((f"vsr_{pos_str}_z", pos_str))
        else:
            print(f"Warning: body at position {body_pos} does not have exactly 3 joints.")

# ----------------------------------------------------------------------
# 3) Add a <site> to each body so we can anchor tendons to it.
#    We'll name the site "corner_<body_name>" and place it at local pos="0 0 0".
# ----------------------------------------------------------------------
for body in root.findall("./worldbody/body"):
    body_name = body.get("name")
    if body_name:
        site = ET.SubElement(body, "site")
        site.set("name", f"corner_{body_name}")
        site.set("pos", "0 0 0")
        site.set("size", "0.005")  # small sphere for visualization

# ----------------------------------------------------------------------
# 4) Create a <tendon> block (if it doesn't exist), then add a <spatial> tendon.
#    For demonstration, we'll pick two bodies to connect with a single tendon.
# ----------------------------------------------------------------------
# For example, let’s pick the first two bodies in the worldbody (if they exist).
all_bodies = [b.get("name") for b in root.findall("./worldbody/body") if b.get("name")]
if len(all_bodies) >= 2:
    bodyA = all_bodies[0]
    bodyB = all_bodies[1]
    # Or pick any two bodies you want in your 10x10x10 structure

    tendon_elem = root.find("tendon")
    if tendon_elem is None:
        tendon_elem = ET.SubElement(root, "tendon")

    # Add one <spatial> tendon from corner_bodyA to corner_bodyB
    spatial = ET.SubElement(tendon_elem, "spatial")
    spatial.set("name", "demo_tendon")
    spatial.set("width", "0.003")
    spatial.set("rgba", "1 0 0 1")
    spatial.set("stiffness", "1")  # no built-in spring force (purely actuator-driven)
    spatial.set("damping", "0")    # no built-in damping

    # The tendon passes through these two sites
    s1 = ET.SubElement(spatial, "site")
    s1.set("site", f"corner_{bodyA}")
    s2 = ET.SubElement(spatial, "site")
    s2.set("site", f"corner_{bodyB}")

# ----------------------------------------------------------------------
# 5) Add motor actuators for all the newly named joints (as before).
# ----------------------------------------------------------------------
actuator = root.find("actuator")
if actuator is None:
    actuator = ET.SubElement(root, "actuator")

# (a) add a motor for each joint found
for joint_name, pos_str in all_joints:
    motor = ET.SubElement(actuator, "motor")
    motor.set("name", f"{joint_name}_motor")
    motor.set("joint", joint_name)
    motor.set("gear", "100")

# (b) also attach a muscle actuator to the new tendon if it was created
if len(all_bodies) >= 2:
    muscle_act = ET.SubElement(actuator, "motor")
    muscle_act.set("name", "demo_muscle")
    muscle_act.set("tendon", "demo_tendon")
    # You can control how tension scales with ctrlrange, gear, etc.
    muscle_act.set("ctrlrange", "0 1")
    muscle_act.set("gear", "100")

# ----------------------------------------------------------------------
# 6) Write the modified XML to disk
# ----------------------------------------------------------------------
tree.write(MODIFIED_XML_PATH, encoding="utf-8", xml_declaration=True)
print(f"Modified XML saved to {MODIFIED_XML_PATH}")

# ----------------------------------------------------------------------
# 7) Load and run the simulation
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODIFIED_XML_PATH)
data = mujoco.MjData(model)

scene_option = mujoco.MjvOption()
viewer = mujoco.viewer.launch_passive(model, data)

amplitude = 3.0  # desired amplitude of oscillation
frequency = 0.01  # frequency in Hz
phase = 1.0      # phase offset

start_time = time.time()
while data.time < DURATION:
    # --- Example: If you want to drive the new muscle with some signal:
    muscle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "demo_muscle")
    if muscle_id >= 0:
        data.ctrl[muscle_id] = 5 * math.sin(20.0*math.pi*frequency*data.time)
    #
    # or do nothing, let it stay at ctrl=0

    mujoco.mj_step(model, data)
    viewer.sync()
