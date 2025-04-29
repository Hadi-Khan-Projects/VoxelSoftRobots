#!/usr/bin/env python3
"""
vsr_rnn_train.py

An example of integrating an RNN-based controller into the MuJoCo simulation of a
Voxel-based Soft Robot (VSR). This code uses a simple recurrent neural network (RNN)
policy (with an RNNCell and a final linear layer) that receives the robot’s state
(obtained from data.qpos and data.qvel) and outputs a control signal for each actuator.
The outputs are squashed to (0,1) to meet the actuator’s control range. A fixed
standard deviation is used so that the policy is stochastic; we then use the REINFORCE
algorithm (Monte Carlo policy gradients) to update the network after each episode.

Before running, be sure that:
  - Your VSR model (from vsr.py and CSV) is working.
  - You have installed MuJoCo and PyTorch.
  - You adjust parameters (reward, episode duration, etc.) as appropriate for your study.
"""

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vsr import VoxelRobot

# ========================
# Simulation & Training Parameters
# ========================
MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"  # assumes a CSV exists here
EPISODE_DURATION = 20.0      # seconds per episode (adjust as needed)
NUM_EPISODES = 100           # number of episodes to train
GAMMA = 0.99               # discount factor for returns
LR = 1e-3                  # learning rate for the optimizer

# ========================
# Create and Load the VSR Model
# ========================
# Create a 10x10x10 VoxelRobot, load its CSV description, and generate the MuJoCo XML.
vsr = VoxelRobot(10, 10, 10)
vsr.load_model_csv(FILEPATH + ".csv")
xml_string = vsr.generate_model(FILEPATH)

# Build the MuJoCo model and simulation data
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# ========================
# Define the RNN Controller Policy
# ========================
# For this controller, we will use as “observation” the concatenation of qpos and qvel.
input_dim = len(data.qpos) + len(data.qvel)
output_dim = model.nu  # one control per actuator
hidden_dim = 64        # hidden state size (can be tuned)

class RNNController(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNController, self).__init__()
        # Use a simple RNN cell (you might also experiment with LSTM/GRU)
        self.rnn = nn.RNNCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h):
        h = self.rnn(x, h)
        # Use sigmoid to squash outputs into (0,1) – the valid control range.
        out = torch.sigmoid(self.fc(h))
        return out, h

# Instantiate the controller and set up an optimizer.
controller = RNNController(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(controller.parameters(), lr=LR)

# If a GPU is available, use it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
controller.to(device)

# ========================
# Helper Function to Get the Current State
# ========================
def get_state(data):
    """Concatenate qpos and qvel from the MuJoCo simulation data."""
    return np.concatenate([data.qpos, data.qvel])

# ========================
# Choose a Reference for Reward
# ========================
# In this example we reward the robot for moving its base body forward along the x-axis.
# (Here we assume the first body in the model is named "vsr_0".)
base_body_id = model.body("vsr_0").id

# ========================
# Main Training Loop
# ========================
print("Starting training...")

for episode in range(NUM_EPISODES):
    # Reset the simulation at the start of an episode.
    mujoco.mj_resetData(model, data)
    data.time = 0.0  # reset simulation time (if needed)
    
    # Get the initial state and convert to a PyTorch tensor.
    state_np = get_state(data)
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Initialize the hidden state to zero.
    h = torch.zeros(1, hidden_dim, device=device)
    
    # Lists to store the log probabilities of actions and rewards obtained.
    log_probs = []
    rewards = []
    episode_reward = 0.0
    
    # Use the initial x-position of the base for computing incremental rewards.
    previous_x = data.xpos[base_body_id, 0]
    
    # Run the simulation until the episode duration is reached.
    while data.time < EPISODE_DURATION:
        # Forward pass through the controller:
        #  - action_means: shape (1, output_dim) with values in (0,1)
        #  - h: updated hidden state.
        action_means, h = controller(state, h)
        
        # Create a stochastic policy by defining a Gaussian with fixed std.
        std = 0.05  # standard deviation (adjust as needed)
        dist = torch.distributions.Normal(action_means, std)
        action = dist.sample()
        # Ensure actions remain in [0, 1]
        action = torch.clamp(action, 0.0, 1.0)
        log_prob = dist.log_prob(action).sum()  # sum log-probabilities across actuators
        log_probs.append(log_prob)
        
        # Set the control commands in the simulation.
        action_np = action.squeeze(0).detach().cpu().numpy()
        data.ctrl[:] = action_np
        
        # Step the simulation forward by one time step.
        mujoco.mj_step(model, data)
        
        # Define a simple reward: change in x-position of the base body.
        current_x = data.xpos[base_body_id, 0]
        reward = current_x - previous_x
        previous_x = current_x
        rewards.append(reward)
        episode_reward += reward
        
        # Get the next state from the simulation.
        state_np = get_state(data)
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    # === Compute Discounted Returns ===
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    # Normalize returns for better training stability.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # === Compute the Policy Gradient Loss ===
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.stack(policy_loss).sum()
    
    # Update the controller using gradient descent.
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    print(f"Episode {episode+1}/{NUM_EPISODES} -- Total Reward: {episode_reward:.3f}")

print("Training complete.")
