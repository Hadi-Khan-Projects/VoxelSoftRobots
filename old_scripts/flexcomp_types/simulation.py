import mujoco
import mujoco.viewer
import time

tests = [
    "basic_types.xml", # 0
    "basic_types_rotated.xml", # 1
    "direct_type.xml", # 2
]

test = 0

with open(tests[test], "r") as file:
    xml = file.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 20  # seconds
framerate = 60  # Hz

viewer = mujoco.viewer.launch_passive(model, data)

start_time = time.time()
while data.time < duration:
    mujoco.mj_step(model, data)

    viewer.sync()

    time.sleep(1 / framerate)

viewer.close()
