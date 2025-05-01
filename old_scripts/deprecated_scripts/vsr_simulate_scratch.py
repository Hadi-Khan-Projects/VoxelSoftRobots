import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

MODEL = "voxel_v2"
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
            <elasticity young="250" poisson="0.3"/>
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

# Find or create the <tendon> block just once
tendon_elem = root.find("tendon")
if tendon_elem is None:
    tendon_elem = ET.SubElement(root, "tendon")

# We also want to add muscle actuators for each edge:
actuator_elem = root.find("actuator")
if actuator_elem is None:
    actuator_elem = ET.SubElement(root, "actuator")

# ----------------------------------------------------------------------
# 2A) Collect all bodies in a dictionary keyed by their (x,y,z) position
# ----------------------------------------------------------------------
all_bodies = {}  # maps (x, y, z) -> body_name

for body in root.findall("./worldbody/body"):
    body_name = body.get("name")
    body_pos_str = body.get("pos", "")
    
    if not body_pos_str:
        continue

    # Parse "pos" attribute e.g. "3 4 2" -> (3, 4, 2)
    # Make sure these are integer or float if your body positions are not guaranteed integer
    try:
        x_str, y_str, z_str = body_pos_str.split()
        x_int = int(float(x_str))
        y_int = int(float(y_str))
        z_int = int(float(z_str))
        all_bodies[(x_int, y_int, z_int)] = body_name
    except ValueError:
        # If pos is not strictly an integer triple, skip or handle differently
        pass

# ----------------------------------------------------------------------
# 2B) For each body, look for ALL neighbors in the 26 directions around it,
#     i.e. dx,dy,dz in {-1, 0, 1} excluding (0,0,0).
#     We'll store unique body pairs in a set so we don't duplicate tendons.
# ----------------------------------------------------------------------

tendon_names = []
adjacent_pairs = set()  # Will hold pairs of ( (x1, y1, z1), (x2, y2, z2) ), sorted

# Precompute all possible neighbor displacements
neighbor_shifts = [
    (dx, dy, dz)
    for dx in [-1, 0, 1]
    for dy in [-1, 0, 1]
    for dz in [-1, 0, 1]
    if not (dx == 0 and dy == 0 and dz == 0)
]

for (x, y, z), bodyA_name in all_bodies.items():
    for dx, dy, dz in neighbor_shifts:
        nx, ny, nz = x + dx, y + dy, z + dz
        if (nx, ny, nz) in all_bodies:
            bodyB_name = all_bodies[(nx, ny, nz)]
            # Create a sorted pair so we don't duplicate (A->B) and (B->A)
            if (nx, ny, nz) < (x, y, z):
                pair = ((nx, ny, nz), (x, y, z))
            else:
                pair = ((x, y, z), (nx, ny, nz))
            adjacent_pairs.add(pair)

# Now we have a set of unique adjacent body pairs, including diagonals.
# Create one spatial tendon per unique pair.
for (x1, y1, z1), (x2, y2, z2) in adjacent_pairs:
    bodyA_name = all_bodies[(x1, y1, z1)]
    bodyB_name = all_bodies[(x2, y2, z2)]
    t_name = f"edge_{x1}_{y1}_{z1}_to_{x2}_{y2}_{z2}"

    spatial = ET.SubElement(
        tendon_elem,
        "spatial",
        name=t_name,
        width="0.003",
        rgba="1 0 0 1",
        stiffness="1",
        damping="0"
    )
    # The tendon passes through site corner_bodyA -> corner_bodyB
    ET.SubElement(spatial, "site", site=f"corner_{bodyA_name}")
    ET.SubElement(spatial, "site", site=f"corner_{bodyB_name}")

    tendon_names.append(t_name)

print(f"Created {len(tendon_names)} new tendons (including diagonals) with motors.")

# ----------------------------------------------------------------------
# 2C) Add a <motor> for each new tendon (a "muscle" or "cable" style actuator)
# ----------------------------------------------------------------------
for t_name in tendon_names:
    motor = ET.SubElement(
        actuator_elem,
        "motor",
        name=f"{t_name}_motor",
        tendon=t_name,
        ctrlrange="0 1",
        gear="10"
    )

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
# actuator = root.find("actuator")
# if actuator is None:
#     actuator = ET.SubElement(root, "actuator")

# # (a) add a motor for each joint found
# for joint_name, pos_str in all_joints:
#     motor = ET.SubElement(actuator, "motor")
#     motor.set("name", f"{joint_name}_motor")
#     motor.set("joint", joint_name)
#     motor.set("gear", "100")

# # (b) also attach a muscle actuator to the new tendon if it was created
# if len(all_bodies) >= 2:
#     muscle_act = ET.SubElement(actuator, "motor")
#     muscle_act.set("name", "demo_muscle")
#     muscle_act.set("tendon", "demo_tendon")
#     # You can control how tension scales with ctrlrange, gear, etc.
#     muscle_act.set("ctrlrange", "0 1")
#     muscle_act.set("gear", "100")

# ----------------------------------------------------------------------
# 6) Write the modified XML to disk
# ----------------------------------------------------------------------

# Convert the ElementTree to a string
rough_string = ET.tostring(root, encoding="utf-8", method="xml")

# Use minidom to pretty-print the XML string
parsed = minidom.parseString(rough_string)
pretty_xml = parsed.toprettyxml(indent="  ")  # Adjust indent size as needed

# Write the pretty-printed XML to a file
with open(MODIFIED_XML_PATH, "w", encoding="utf-8") as f:
    f.write(pretty_xml)

def clean_extra_newlines(xml_content):
    # Remove double newlines between tags
    cleaned = re.sub(r'\n\s*\n', '\n', xml_content)
    return cleaned

# Example usage
with open(MODIFIED_XML_PATH, 'r') as file:
    content = file.read()

cleaned_content = clean_extra_newlines(content)

with open(MODIFIED_XML_PATH, 'w') as file:
    file.write(cleaned_content)

print(f"Modified XML saved with proper formatting to {MODIFIED_XML_PATH}")

# ----------------------------------------------------------------------
# 7) Load and run the simulation
# ----------------------------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODIFIED_XML_PATH)
data = mujoco.MjData(model)

scene_option = mujoco.MjvOption()
# viewer = mujoco.viewer.launch_passive(model, data)

amplitude = 3.0  # desired amplitude of oscillation
frequency = 0.01  # frequency in Hz
phase = 1.0      # phase offset

paused = False

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while data.time < DURATION:
        # Example: drive all motors with some pattern
        for i in range(model.nu):
            data.ctrl[i] = 2 * math.sin(20.0*math.pi*frequency*data.time)
        
        if not paused:
            mujoco.mj_step(model, data)
            viewer.sync()

