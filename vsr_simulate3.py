import math
import random
import time
import os
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import torch.optim as optim
from vsr import VoxelRobot

# ---------------------------
# Simulation and Voxel Setup
# ---------------------------
MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
EPISODE_DURATION = 45  # seconds per episode
NUM_EPISODES = 300  # number of training episodes
update_interval = 0.2  # seconds

# Create a 10x10x10 empty vsr
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
# Only control these 6 primary voxels:
primary_voxels = [(1, 1, 1), (1, 4, 1), (3, 1, 3), (3, 4, 3), (6, 1, 1), (6, 4, 1)]

# For each primary voxel, include its face-adjacent neighbors (if they exist) in the control group.
# neighbor_offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
# For each primary voxel, include its face-adjacent and corner-adjacent neighbors (if they exist) in the control group.
neighbor_offsets = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
    (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1),
    (-1, -1, 1), (-1, -1, -1)
]
controlled_voxels_map = {}
for voxel in primary_voxels:
    controlled = [voxel]  # include the primary voxel itself
    for dx, dy, dz in neighbor_offsets:
        neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
        if neighbor in voxel_motor_map:
            controlled.append(neighbor)
    controlled_voxels_map[voxel] = controlled

# Feature vector size: 2 per vertex + 2 global features
sorted_vert_ids = sorted(voxel_motor_map.keys())
feature_dim = 2 * len(sorted_vert_ids) + 2


# ---------------------------
# Define the LSTM-based RNN Policy
# ---------------------------
class VoxelRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers=1):
        super(VoxelRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, 24)
        self.output_layer = nn.Linear(24, output_dim)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.linear(out)
        return out, hidden


# Instantiate our network.
hidden_size = 64
# Output dimension is now 6, one per primary voxel.
voxel_rnn = VoxelRNN(
    input_dim=feature_dim, hidden_size=hidden_size, output_dim=len(primary_voxels)
)
optimizer = optim.Adam(voxel_rnn.parameters(), lr=1e-3)


# ---------------------------
# Helper Function: Compute Discounted Returns
# ---------------------------
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize returns for stability.
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    return returns


# ---------------------------
# Set up Target and Robot IDs
# ---------------------------
target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vsr")

target_x = data.subtree_com[target_body_id, 0]
target_y = data.subtree_com[target_body_id, 1]
robot_x = data.subtree_com[robot_body_id, 0]
robot_y = data.subtree_com[robot_body_id, 1]
x_dist_target = -(target_x - robot_x)
y_dist_target = -(target_y - robot_y)

# ---------------------------
# Training Loop (REINFORCE) with Viewer
# ---------------------------
print("Starting training...")

for episode in range(NUM_EPISODES):
    # Reset simulation state for the new episode.
    mujoco.mj_resetData(model, data)
    data.time = 0.0

    distances_x = [
        x_dist_target,
        x_dist_target,
        x_dist_target,
        x_dist_target,
        x_dist_target,
    ]
    distances_y = [
        y_dist_target,
        y_dist_target,
        y_dist_target,
        y_dist_target,
        y_dist_target,
    ]
    # Reinitialize LSTM hidden state.
    hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    # Open MuJoCo viewer for this episode.
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

        viewer.cam.lookat[:] = [-18, 0, 2.5]
        viewer.cam.distance = 53
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        episode_reward = 0.0
        log_probs = []
        rewards = []
        last_update_time = data.time

        # ---- SET INITIAL ACTIONS for Primary Voxels and Their Neighbors ----
        for i, primary in enumerate(primary_voxels):
            # For example, assign -10 to the first half and 10 to the rest.
            control_signal = -10 if i < len(primary_voxels) // 2 else 10
            for voxel in controlled_voxels_map[primary]:
                for motor_id in voxel_motor_map[voxel]:
                    data.ctrl[motor_id] = control_signal

        # Run episode simulation
        while data.time < EPISODE_DURATION:
            if data.time - last_update_time >= update_interval:
                last_update_time = data.time

                # Forward pass through the RNN policy.
                features_tensor = torch.randn(1, 1, feature_dim)  # Dummy random input
                rnn_output, hidden = voxel_rnn(features_tensor, hidden)
                probs = torch.sigmoid(rnn_output).squeeze(0)

                # Sample actions from policy.
                dist = torch.distributions.Bernoulli(probs)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).sum()
                log_probs.append(log_prob)

                # Set motor controls based on RNN predictions for each primary voxel group.
                for i, primary in enumerate(primary_voxels):
                    control_signal = 10 if actions[i].item() == 1 else -10
                    for voxel in controlled_voxels_map[primary]:
                        for motor_id in voxel_motor_map[voxel]:
                            data.ctrl[motor_id] = control_signal

                # Reward function: negative Manhattan distance to target.
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

                # Terminate early if close to target.
                if abs(target_x - robot_x) < 10 and abs(target_y - robot_y) < 10:
                    break

            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()

    # Compute loss and update policy
    returns = compute_returns(rewards, gamma=0.99)
    loss = -sum(lp * R for lp, R in zip(log_probs, returns))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"Episode {episode + 1}/{NUM_EPISODES} reward: {episode_reward:.2f} distance from target: {abs(target_x - robot_x):.2f}, {abs(target_y - robot_y):.2f}"
    )

print("Training complete!")
