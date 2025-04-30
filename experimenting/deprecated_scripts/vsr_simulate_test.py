import mujoco
import mujoco.viewer
import time
from vsr import VoxelRobot
import math
import xml.etree.ElementTree as ET
import re

FILEPATH = "vsr_models/voxel_v2/voxel_v2_test.xml"
DURATION = 300 

model = mujoco.MjModel.from_xml_path(FILEPATH)
data = mujoco.MjData(model)

scene_option = mujoco.MjvOption()

viewer = mujoco.viewer.launch_passive(model, data)

amplitude = 3.0  # desired amplitude of oscillation
frequency = 1.0  # frequency in Hz
phase = 1.0  # phase offset

start_time = time.time()
while data.time < DURATION:

    # Step the simulation
    mujoco.mj_step(model, data)
    viewer.sync()
