import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
DURATION = 300  # seconds

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)

# load, visualise, and then generate model
vsr.load_model_csv(FILEPATH + ".csv")
# vsr.visualise_model()
xml_string = vsr.generate_model(FILEPATH)

print("No. of vertexes: ", vsr.num_vertex())

# ----------------------------------------------------------------------
# 1) Create a MuJoCo string defining a flexcomp from your generated points/elements
#    (the same as before).
# ----------------------------------------------------------------------
# xml_string = f"""
# <mujoco>

#     <compiler autolimits="true"/>
#     <include file="scene.xml"/>
#     <compiler autolimits="true"/>
#     <option solver="Newton" tolerance="1e-6" timestep="{TIMESTEP}" integrator="implicitfast"/>
#     <size memory="2000M"/>

#     <worldbody>
#         <flexcomp name="vsr" type="direct" dim="3"
#             point="{point}"
#             element="{element}"
#             radius="0.005" rgba="0.1 0.9 0.1 1" mass="{vsr.num_vertex()/10}">
#             <contact condim="3" solref="0.01 1" solimp="0.95 0.99 0.0001" selfcollide="none"/>
#             <edge damping="1"/>
#             <elasticity young="250" poisson="0.3"/>
#         </flexcomp>
#     </worldbody>

#     <tendon>{spatial}</tendon>

#     <actuator>{motor}</actuator>

# </mujoco>
# """

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# # Save model as XML for reference
# mujoco.mj_saveLastXML(filename=FILEPATH + ".xml", m=model)

scene_option = mujoco.MjvOption()
# viewer = mujoco.viewer.launch_passive(model, data)

paused = False

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

frequency = 0.01  # frequency in Hz

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while data.time < DURATION:

        # drive motor
        for i in range(model.nu):
            data.ctrl[i] = 5 * math.sin(20.0*math.pi*frequency*data.time)
        
        if not paused:
            mujoco.mj_step(model, data)
            viewer.sync()

