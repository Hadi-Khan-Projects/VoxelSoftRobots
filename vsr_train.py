"""
vsr_rl.py

This script implements reinforcement learning (using policy gradient) for controlling
a Voxel-Based Soft Robot (VSR) built with MuJoCo. Instead of a hand‐coded sine controller,
we define a MuJoCo “environment” that wraps your VSR model and then use a simple neural
network (with a Beta–distributed stochastic policy) to output motor control signals.
The reward is taken as the forward displacement (change in COM x-coordinate) of the robot.
Over many episodes, the controller learns to “evolve” a locomotion strategy that increases
forward progress.

Before running, be sure that:
  • Your vsr.py is on the PYTHONPATH.
  • The file CSV for the robot (e.g. quadruped_v3.csv) is available in the correct folder.
  • MuJoCo and PyTorch are installed.
"""

import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from vsr import VoxelRobot
import math
import random
import time

############################
# 1. Load and Generate VSR Model
############################

# (These settings match your vsr.simulate.py script.)
MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"

# Create a VSR of size 10x10x10, load its CSV description, and generate the MuJoCo XML.
vsr = VoxelRobot(10, 10, 10)
vsr.load_model_csv(FILEPATH + ".csv")
xml_string = vsr.generate_model(FILEPATH)
print("No. of vertexes: ", vsr.num_vertex())

# Create the MuJoCo model and data objects.
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

############################
# 2. Define a MuJoCo Environment for the VSR
############################

class VoxelEnv:
    """
    A minimal MuJoCo-based environment for the VSR.
    
    The observation is defined as the robot’s center-of-mass (COM) position (x,y,z)
    and COM linear velocity (x,y,z) (a 6-dimensional vector). The reward at each step
    is the change in COM x-coordinate (i.e. forward progress).
    
    We apply the same control (an action vector of length model.nu, i.e. one per actuator)
    for a number of simulation steps (to avoid setting control at every 0.001 s).
    """
    def __init__(self, model, data, sim_steps_per_action=10, episode_duration=5.0):
        self.model = model
        self.data = data
        self.sim_steps_per_action = sim_steps_per_action
        self.episode_duration = episode_duration  # seconds per episode

        # Identify robot bodies by name (assuming names start with "vsr")
        self.robot_body_ids = []
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is not None and body_name.startswith("vsr"):
                self.robot_body_ids.append(i)

        self.initial_time = 0.0

    def reset(self):
        """Reset the simulation state and return the initial observation."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.time = self.initial_time
        self.data.ctrl[:] = 0  # set all actuators to zero
        self.prev_com = self.get_com()
        self.current_step = 0
        return self.get_obs()

    def step(self, action):
        """
        Apply the action (a vector of motor controls) for sim_steps_per_action simulation steps,
        then compute and return the new observation, reward, done flag, and an info dict.
        """
        # Set all actuator controls (note: action should be an array of length model.nu)
        self.data.ctrl[:] = action

        # Run several simulation steps with the current control.
        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)
        
        # Compute the new center-of-mass.
        new_com = self.get_com()
        # Reward is the change in x-position (forward progress)
        reward = new_com[0] - self.prev_com[0]
        self.prev_com = new_com
        
        # Form the new observation.
        obs = self.get_obs()
        
        # Check if the episode is done.
        done = (self.data.time >= self.episode_duration)
        self.current_step += 1
        
        return obs, reward, done, {}

    def get_com(self):
        """Compute the robot’s center-of-mass (COM) position."""
        total_mass = 0.0
        com = np.zeros(3)
        for i in self.robot_body_ids:
            mass = self.model.body_mass[i]
            pos = self.data.xpos[i]
            total_mass += mass
            com += mass * pos
        return com / total_mass

    def get_com_velocity(self):
        """Compute the robot’s center-of-mass (COM) linear velocity."""
        total_mass = 0.0
        com_vel = np.zeros(3)
        for i in self.robot_body_ids:
            mass = self.model.body_mass[i]
            vel = self.data.cvel[i][:3]
            total_mass += mass
            com_vel += mass * vel
        return com_vel / total_mass

    def get_obs(self):
        """Return the observation as [com_x, com_y, com_z, com_vel_x, com_vel_y, com_vel_z]."""
        com = self.get_com()
        com_vel = self.get_com_velocity()
        return np.concatenate([com, com_vel])


# Create the environment.
env = VoxelEnv(model, data, sim_steps_per_action=10, episode_duration=30.0)

############################
# 3. Define the Policy Network
############################

# Here we use a stochastic policy that outputs parameters for a Beta distribution for each actuator.
# The Beta distribution is naturally supported on [0, 1] which matches the control range.
class BetaPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(BetaPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # will be split into alpha and beta for each action dimension
        )
    
    def forward(self, obs):
        """
        Given an observation (as a torch tensor), output the Beta distribution parameters.
        We use softplus + 1 to ensure that alpha and beta are >1.
        """
        params = self.net(obs)
        # Split the parameters for each action dimension
        action_dim = params.shape[-1] // 2
        alpha = params[..., :action_dim]
        beta  = params[..., action_dim:]
        # Ensure the parameters are positive (and >1 for a reasonably “peaked” distribution).
        alpha = torch.nn.functional.softplus(alpha) + 1.0
        beta = torch.nn.functional.softplus(beta) + 1.0
        return alpha, beta

def select_action(policy, obs):
    """
    Given an observation, use the policy network to sample an action from a Beta distribution.
    Returns the action (as a numpy array) and the log probability (a torch scalar) for policy gradient.
    """
    obs_tensor = torch.from_numpy(obs).float()
    alpha, beta = policy(obs_tensor)
    dist = distributions.Beta(alpha, beta)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum()  # sum log probs over all actuators
    return action.detach().numpy(), log_prob

############################
# 4. Training Loop using REINFORCE
############################

# Set random seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Dimensions:
obs_dim = 6                 # [com (3) + com_vel (3)]
action_dim = model.nu       # one control signal per actuator

# Instantiate the policy network and an optimizer.
policy = BetaPolicy(obs_dim, action_dim, hidden_dim=128)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# Training parameters.
num_episodes = 100          # adjust as needed
gamma = 0.99                # discount factor

def compute_returns(rewards, gamma):
    """Compute discounted returns for a list of rewards."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Optionally normalize returns to help learning.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

print("Starting training...")
for episode in range(num_episodes):
    obs = env.reset()
    episode_rewards = []
    log_probs = []
    done = False

    # Run one episode.
    while not done:
        action, log_prob = select_action(policy, obs)
        next_obs, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        episode_rewards.append(reward)
        obs = next_obs

    # Compute cumulative discounted reward.
    returns = compute_returns(episode_rewards, gamma)

    # Compute policy gradient loss.
    loss = 0
    for log_prob, R in zip(log_probs, returns):
        loss += -log_prob * R  # gradient ascent on expected reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(episode_rewards)
    print(f"Episode {episode+1:3d}: Total Reward = {total_reward:.3f}, Loss = {loss.item():.3f}")

print("Training complete.")

############################
# 5. (Optional) Test the Learned Policy in a Simulation Viewer
############################

# If you wish to visually inspect the learned locomotion you can run one final episode with
# a MuJoCo viewer. (Make sure to run this on a system with a display.)
RUN_VIEWER = True  # Set to True if you want to see the simulation.

if RUN_VIEWER:
    import mujoco.viewer

    obs = env.reset()
    # Launch the viewer in passive mode.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        done = False
        while not done:
            # Use the learned policy (choose the mean of the Beta distribution by sampling many times
            # and taking the average or by using the mode if desired – here we simply sample).
            action, _ = select_action(policy, obs)
            obs, _, done, _ = env.step(action)
            viewer.sync()
            # time.sleep(0.01)
