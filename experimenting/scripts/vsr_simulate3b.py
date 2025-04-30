import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math
import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from vsr import VoxelRobot

MIN_DIM = 1
MAX_DIM = 20
TIMESTEP = "0.001"
GEAR = 60

class VoxelRobot:
    def __init__(self, x: int, y: int, z: int) -> None:
        """Initialise an (x, y, z) VSR grid of 0's."""
        if x < MIN_DIM or y < MIN_DIM or z < MIN_DIM:
            raise ValueError("VSR's voxel grid dimensions must be positive integers.")
        if x > MAX_DIM or y > MAX_DIM or z > MAX_DIM:
            raise ValueError("VSR's voxel grid dimensions must not exceed 20.")

        self.max_x = x
        self.max_y = y
        self.max_z = z
        self.voxel_grid = np.zeros((x, y, z), dtype=np.uint8)
        self.point_grid = np.zeros((x + 1, y + 1, z + 1), dtype=np.uint8)
        self.point_dict = {}

    def set_val(self, x: int, y: int, z: int, value) -> None:
        """Set the value at position (x, y, z) to 0 or 1."""
        if value not in [0, 1]:
            raise ValueError("VSR's voxel grid value not set as 0 or 1.")
        self.voxel_grid[x, y, z] = value

    def visualise_model(self) -> None:
        """Visualise the VSR's structure using matplotlib."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Coordinates of all 1's in the grid
        x, y, z = np.where(self.voxel_grid == 1)
        voxel_plot_size = 0.9  # slightly smaller than 1 to avoid overlap

        # Plot each voxel as small cube
        for xi, yi, zi in zip(x, y, z):
            ax.bar3d(
                xi - voxel_plot_size / 2,
                yi - voxel_plot_size / 2,
                zi - voxel_plot_size / 2,
                voxel_plot_size,
                voxel_plot_size,
                voxel_plot_size,
                color="red",
                alpha=0.6,
            )

        ax.set_xlim(0, self.max_x)
        ax.set_ylim(0, self.max_y)
        ax.set_zlim(0, self.max_z)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xticks(np.arange(0, self.max_x + 1, 1))
        ax.set_yticks(np.arange(0, self.max_y + 1, 1))
        ax.set_zticks(np.arange(0, self.max_z + 1, 1))

        plt.show()

    def _check_contiguous(self) -> bool:
        """Check if the VSR's structure is contiguous."""
        visited = np.zeros_like(self.voxel_grid, dtype=bool)
        start_voxel = np.argwhere(self.voxel_grid == 1)
        if start_voxel.size == 0:
            return False
        start_voxel = tuple(start_voxel[0])
        self._dfs(start_voxel, visited)
        if not np.array_equal(self.voxel_grid == 1, visited):
            raise ValueError("VSR's structure is not contiguous.")

    def _dfs(self, voxel, visited) -> None:
        """Depth first search to check for VSR contiguity."""
        stack = [voxel]
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        while stack:
            x, y, z = stack.pop()
            if visited[x, y, z]:
                continue
            visited[x, y, z] = True
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < self.voxel_grid.shape[0] and 0 <= ny < self.voxel_grid.shape[1] and 0 <= nz < self.voxel_grid.shape[2]:
                    if self.voxel_grid[nx, ny, nz] == 1 and not visited[nx, ny, nz]:
                        stack.append((nx, ny, nz))

    def save_model_csv(self, filename) -> None:
        """Save the current voxel grid to a .csv file of a given name"""
        data = []
        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.voxel_grid[x, y, z] == 1:
                        data.append((x, y, z))

        df = pd.DataFrame(data, columns=["x", "y", "z"], dtype=np.uint8)
        df.to_csv(filename, index=False)

    def load_model_csv(self, filename) -> None:
        """Reset the current voxel grid to all zeroes and load grid from .csv file"""
        df = pd.read_csv(filename)
        self.voxel_grid = np.zeros((self.max_x, self.max_y, self.max_z), dtype=np.uint8)
        for _, row in df.iterrows():
            self.voxel_grid[row["x"], row["y"], row["z"]] = 1

    def generate_model(self, filepath) -> str:
        """Generate the MuJoCo model for the VSR."""
        self._check_contiguous()
        points_string, elements_string = self._generate_flexcomp_geometry()

        xml_string = f"""
        <mujoco>
            <compiler autolimits="true"/>
            <include file="scene.xml"/>
            <compiler autolimits="true"/>
            <option solver="Newton" tolerance="1e-6" timestep="{TIMESTEP}" integrator="implicitfast"/>
            <size memory="2000M"/>
            <worldbody>
                <body name="target" pos="-40 3 3.5">
                    <geom type="box" size="1 1 3" rgba="1 0 0 0.7"/>
                </body>
                <flexcomp name="vsr" type="direct" dim="3"
                    point="{points_string}"
                    element="{elements_string}"
                    radius="0.005" rgba="0.1 0.9 0.1 1" mass="{self.num_vertex()/10}">
                    <contact condim="3" solref="0.01 1" solimp="0.95 0.99 0.0001" selfcollide="none"/>
                    <edge damping="1"/>
                    <elasticity young="250" poisson="0.3"/>
                </flexcomp>
            </worldbody>
        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        mujoco.mj_saveLastXML(filename=filepath + ".xml", m=model)

        # ADD SITES
        tree = ET.parse(filepath + ".xml")
        root = tree.getroot()

        # Add site to each body
        for body in root.findall("./worldbody/body"):
            body_name = body.get("name")
            if not body.get('pos'):
                body.set("pos", "0 0 0")
            if body_name:
                site = ET.SubElement(body, "site")
                site.set("name", f"site_{body.get('pos').replace(' ', '_')}")
                site.set("pos", "0 0 0")
                site.set("size", "0.005")  # small sphere for visualization

        # ADD SPATIAL TENDONS
        tendon_elem = root.find("tendon")
        if tendon_elem is None:
            tendon_elem = ET.SubElement(root, "tendon")

        actuator_elem = root.find("actuator")
        if actuator_elem is None:
            actuator_elem = ET.SubElement(root, "actuator")

        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.voxel_grid[x, y, z] == 1:
                        # +x +y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x}_{y}_{z}_to_{x+1}_{y+1}_{z+1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="1 0 0 1",
                            stiffness="1",
                            damping="0"
                        )
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x+1}_{y+1}_{z+1}")
                        ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x}_{y}_{z}_to_{x+1}_{y+1}_{z+1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(GEAR)
                        )

                        # -x -y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x+1}_{y+1}_{z}_to_{x}_{y}_{z+1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="0 1 0 1",
                            stiffness="1",
                            damping="0"
                        )
                        ET.SubElement(spatial, "site", site=f"site_{x+1}_{y+1}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y}_{z+1}")
                        ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x+1}_{y+1}_{z}_to_{x}_{y}_{z+1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(GEAR)
                        )

                        # +x -y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x}_{y+1}_{z}_to_{x+1}_{y}_{z+1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="0 0 1 1",
                            stiffness="1",
                            damping="0"
                        )
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y+1}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x+1}_{y}_{z+1}")
                        ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x}_{y+1}_{z}_to_{x+1}_{y}_{z+1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(GEAR)
                        )

                        # -x +y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x+1}_{y}_{z}_to_{x}_{y+1}_{z+1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="1 0 1 1",
                            stiffness="1",
                            damping="0"
                        )
                        ET.SubElement(spatial, "site", site=f"site_{x+1}_{y}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y+1}_{z+1}")
                        ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x+1}_{y}_{z}_to_{x}_{y+1}_{z+1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(GEAR)
                        )
        # SAVE
        self._save_modded_xml(tree, filepath)
        return ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
    
    def _generate_flexcomp_geometry(self) -> tuple:
        """Generate the points and elements for the voxel flexcomp model."""
        self.point_grid = np.zeros(
            (self.max_x + 1, self.max_y + 1, self.max_z + 1), dtype=np.int8
        )
        points_count = 0
        points_map = {}  # (x, y, z) : point index
        points_string = ""

        for x in range(self.max_x + 1):
            for y in range(self.max_y + 1):
                for z in range(self.max_z + 1):
                    if self._is_point_vertex(x, y, z):
                        self.point_grid[x, y, z] = 1
                        self.point_dict[(x, y, z)] = (0.0, 0.0, 0.0)
                        points_map[(x, y, z)] = points_count
                        points_string += " ".join(map(str, (x, y, z))) + "   "
                        points_count += 1

        elements_string = ""
        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.voxel_grid[x, y, z] == 1:
                        points = [
                            points_map[(x, y, z)],      # 0
                            points_map[(x + 1, y, z)],    # 1
                            points_map[(x + 1, y + 1, z)],# 2
                            points_map[(x, y + 1, z)],    # 3
                            points_map[(x, y, z + 1)],    # 4
                            points_map[(x + 1, y, z + 1)],# 5
                            points_map[(x + 1, y + 1, z + 1)],# 6
                            points_map[(x, y + 1, z + 1)],# 7
                        ]

                        elements = [
                            (0, 3, 7, 6),
                            (0, 7, 4, 6),
                            (0, 4, 5, 6),
                            (0, 5, 1, 6),
                            (0, 1, 2, 6),
                            (0, 2, 3, 6),
                        ]

                        for tetrahedron in elements:
                            for vertex in tetrahedron:
                                elements_string += " " + str(points[vertex])
                            elements_string += "    "
        return points_string.strip(), elements_string.strip()

    def _is_point_possible(self, x, y, z) -> bool:
        """Check if (x, y, z) is a valid index in the voxel grid."""
        return 0 <= x < self.max_x and 0 <= y < self.max_y and 0 <= z < self.max_z

    def _is_point_vertex(self, x, y, z) -> bool:
        """Check if (x, y, z) is a vertex of one of the VSR's voxels."""
        return (
            (self._is_point_possible(x, y, z) and self.voxel_grid[x, y, z] == 1)
            or (self._is_point_possible(x - 1, y, z) and self.voxel_grid[x - 1, y, z] == 1)
            or (self._is_point_possible(x - 1, y - 1, z) and self.voxel_grid[x - 1, y - 1, z] == 1)
            or (self._is_point_possible(x, y - 1, z) and self.voxel_grid[x, y - 1, z] == 1)
            or (self._is_point_possible(x, y, z - 1) and self.voxel_grid[x, y, z - 1] == 1)
            or (self._is_point_possible(x - 1, y, z - 1) and self.voxel_grid[x - 1, y, z - 1] == 1)
            or (self._is_point_possible(x - 1, y - 1, z - 1) and self.voxel_grid[x - 1, y - 1, z - 1] == 1)
            or (self._is_point_possible(x, y - 1, z - 1) and self.voxel_grid[x, y - 1, z - 1] == 1)
        )

    def num_vertex(self) -> int:
        """Return number of vertexes required to generate the VSR."""
        points_count = 0
        for x in range(self.max_x + 1):
            for y in range(self.max_y + 1):
                for z in range(self.max_z + 1):
                    if self._is_point_vertex(x, y, z):
                        points_count += 1
        return points_count
    
    def _save_modded_xml(self, tree, filepath) -> None:
        """Formats an XML tree and saves it."""
        root = tree.getroot()
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")
        cleaned_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])
        new_filepath = filepath + "_modded.xml"
        with open(new_filepath, "w", encoding="utf-8") as f:
            f.write(cleaned_xml)
        print(f"Formatted XML saved to {new_filepath}")

    # --- Genetic Morphology Update (Modification b) ---
    def genetic_morphology_update(self) -> None:
        """
        Randomly add or remove a voxel from the grid.
        The change is accepted only if the new structure remains contiguous.
        """
        x = random.randint(0, self.max_x - 1)
        y = random.randint(0, self.max_y - 1)
        z = random.randint(0, self.max_z - 1)
        old_value = self.voxel_grid[x, y, z]
        # Flip the voxel state: add if absent, remove if present.
        self.voxel_grid[x, y, z] = 1 - old_value
        try:
            self._check_contiguous()
            print(f"Genetic update: flipped voxel ({x}, {y}, {z}) from {old_value} to {self.voxel_grid[x, y, z]}")
        except ValueError:
            self.voxel_grid[x, y, z] = old_value
            print(f"Genetic update: change at voxel ({x}, {y}, {z}) rejected due to non-contiguity")
            raise

# ---------------------------
# Simulation and Voxel Setup
# ---------------------------
MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
EPISODE_DURATION = 45  # seconds per episode
NUM_EPISODES = 1000  # number of training episodes

# Create a 10x10x10 vsr and load its initial morphology.
vsr = VoxelRobot(10, 10, 10)
vsr.load_model_csv(FILEPATH + ".csv")
# vsr.visualise_model()
xml_string = vsr.generate_model(FILEPATH)
print("No. of vertexes: ", vsr.num_vertex())

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Build voxel-to-motor mapping
voxel_motor_map = {}
for i in range(model.nu):
    motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if motor_name:
        parts = motor_name.split("_")
        voxel_coord = tuple(map(int, parts[1:4]))  # (x,y,z)
        if voxel_coord not in voxel_motor_map:
            voxel_motor_map[voxel_coord] = []
        voxel_motor_map[voxel_coord].append(i)

# Ensure each voxel has exactly 4 motors.
for voxel, motors in voxel_motor_map.items():
    assert len(motors) == 4, f"Voxel {voxel} has {len(motors)} motors, expected 4."

# ---------------------------
# Define Primary Voxels and Their Neighbors
# ---------------------------
primary_voxels = [(1, 1, 1), (1, 4, 1), (3, 1, 3), (3, 4, 3), (6, 1, 1), (6, 4, 1)]
neighbor_offsets = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
    (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1),
    (-1, -1, 1), (-1, -1, -1)
]
controlled_voxels_map = {}
for voxel in primary_voxels:
    controlled = [voxel]
    for dx, dy, dz in neighbor_offsets:
        neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
        if neighbor in voxel_motor_map:
            controlled.append(neighbor)
    controlled_voxels_map[voxel] = controlled

sorted_vert_ids = sorted(voxel_motor_map.keys())
feature_dim = 2 * len(sorted_vert_ids) + 2

# ---------------------------
# Define the LSTM-based RNN Policy (Modification a)
# ---------------------------
class VoxelRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(VoxelRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        # The output dimension now is len(primary_voxels) + 1 (extra for update interval)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]  # use only the last time-step
        out = self.linear(out)
        return out, hidden

# Instantiate the network with an output dimension of len(primary_voxels) + 1.
hidden_size = 64
voxel_rnn = VoxelRNN(
    input_dim=feature_dim, hidden_size=hidden_size, output_dim=len(primary_voxels) + 1
)
optimizer = optim.Adam(voxel_rnn.parameters(), lr=1e-3)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns

target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vsr")
target_x = data.subtree_com[target_body_id, 0]
target_y = data.subtree_com[target_body_id, 1]
robot_x = data.subtree_com[robot_body_id, 0]
robot_y = data.subtree_com[robot_body_id, 1]
x_dist_target = -(target_x - robot_x)
y_dist_target = -(target_y - robot_y)

print("Starting training...")

for episode in range(NUM_EPISODES):
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    distances_x = [x_dist_target] * 5
    distances_y = [y_dist_target] * 5
    hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))
    
    # Initialize current update interval with an initial value.
    current_update_interval = 0.05
    last_update_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set visualization options
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXSKIN] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0


        # Set visualization options (omitted for brevity)
        viewer.cam.lookat[:] = [-18, 0, 2.5]
        viewer.cam.distance = 53
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        episode_reward = 0.0
        log_probs = []
        rewards = []

        # ---- SET INITIAL ACTIONS for Primary Voxels and Their Neighbors ----
        for i, primary in enumerate(primary_voxels):
            control_signal = -10 if i < len(primary_voxels) // 2 else 10
            for voxel in controlled_voxels_map[primary]:
                for motor_id in voxel_motor_map[voxel]:
                    data.ctrl[motor_id] = control_signal

        while data.time < EPISODE_DURATION:
            if data.time - last_update_time >= current_update_interval:
                last_update_time = data.time

                # --- Adaptive Time Interval and Action Sampling (Modification a) ---
                features_tensor = torch.randn(1, 1, feature_dim)  # dummy input
                rnn_output, hidden = voxel_rnn(features_tensor, hidden)
                # Split network output: first part for actions, last scalar for update interval.
                action_logits = rnn_output[0, :len(primary_voxels)]
                time_interval_logit = rnn_output[0, -1]
                probs = torch.sigmoid(action_logits)
                dist = torch.distributions.Bernoulli(probs)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).sum()
                log_probs.append(log_prob)

                # Transform the time interval output to ensure it is positive.
                current_update_interval = torch.nn.functional.softplus(time_interval_logit) + 0.05
                current_update_interval = current_update_interval.item()

                # Set motor controls based on the sampled actions.
                for i, primary in enumerate(primary_voxels):
                    control_signal = 10 if actions[i].item() == 1 else -10
                    for voxel in controlled_voxels_map[primary]:
                        for motor_id in voxel_motor_map[voxel]:
                            data.ctrl[motor_id] = control_signal

                target_x = data.subtree_com[target_body_id, 0]
                target_y = data.subtree_com[target_body_id, 1]
                robot_x = data.subtree_com[robot_body_id, 0]
                robot_y = data.subtree_com[robot_body_id, 1]

                distances_x.append(-(target_x - robot_x))
                distances_y.append(-(target_y - robot_y))
                reward_x = distances_x[-5] - distances_x[-1]
                reward_y = distances_y[-5] - distances_y[-1]
                reward = reward_x + reward_y
                rewards.append(reward)
                episode_reward += reward

                if abs(target_x - robot_x) < 10 and abs(target_y - robot_y) < 10:
                    break

            mujoco.mj_step(model, data)
            viewer.sync()

    returns = compute_returns(rewards, gamma=0.99)
    loss = -sum(lp * R for lp, R in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {episode + 1}/{NUM_EPISODES} reward: {episode_reward:.2f} distance from target: {abs(target_x - robot_x):.2f}, {abs(target_y - robot_y):.2f}")

    # --- Genetic Morphology Update every 10 episodes (Modification b) ---
    if (episode + 1) % 10 == 0:
        print("Performing genetic morphology update...")
        try:
            vsr.genetic_morphology_update()
            vsr.save_model_csv(FILEPATH + ".csv")
            xml_string = vsr.generate_model(FILEPATH)
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
            # Rebuild voxel_motor_map after morphology change.
            voxel_motor_map = {}
            for i in range(model.nu):
                motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if motor_name:
                    parts = motor_name.split("_")
                    voxel_coord = tuple(map(int, parts[1:4]))
                    if voxel_coord not in voxel_motor_map:
                        voxel_motor_map[voxel_coord] = []
                    voxel_motor_map[voxel_coord].append(i)
            print("Morphology update successful.")
        except Exception as e:
            print("Morphology update failed:", e)

print("Training complete!")
