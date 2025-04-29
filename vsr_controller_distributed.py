# controller.py
import numpy as np
import math

def tanh(x):
    """ Element-wise tanh activation function. """
    return np.tanh(x)

class DistributedNeuralController:
    def __init__(self, n_voxels, voxel_coords, n_sensors_per_voxel, n_comm_channels, driving_voxel_coord=None):
        """
        Initializes the distributed neural controller.

        Args:
            n_voxels (int): Total number of active voxels.
            voxel_coords (list): List of (x, y, z) tuples for active voxels.
            n_sensors_per_voxel (int): Number of sensor inputs per voxel.
            n_comm_channels (int): Number of communication values exchanged per side (nc).
            driving_voxel_coord (tuple, optional): Coordinates (x, y, z) of the voxel receiving the driving signal. Defaults to None.
        """
        self.n_voxels = n_voxels
        self.voxel_coords = list(voxel_coords) # Ensure it's a list of tuples
        self.voxel_coord_to_index = {coord: i for i, coord in enumerate(self.voxel_coords)}
        self.n_sensors = n_sensors_per_voxel
        self.n_comm = n_comm_channels
        self.driving_voxel_coord = driving_voxel_coord
        self.driving_voxel_index = -1
        if self.driving_voxel_coord and self.driving_voxel_coord in self.voxel_coord_to_index:
             self.driving_voxel_index = self.voxel_coord_to_index[self.driving_voxel_coord]

        # MLP Input Size: Sensors + 4*Comm_Inputs + Driving_Signal
        self.input_size = self.n_sensors + 4 * self.n_comm + 1
        # MLP Output Size: Actuation + 4*Comm_Outputs
        self.output_size = 1 + 4 * self.n_comm

        # --- Parameters to be Evolved ---
        # Initialize weights and biases (randomly for now, replace with evolved values later)
        # Small random values centered around 0 are usually a good start
        self.weights = np.random.uniform(-0.1, 0.1, (self.output_size, self.input_size))
        self.biases = np.random.uniform(-0.1, 0.1, self.output_size)
        # --------------------------------

        # --- State Variables ---
        # Store communication outputs from the *previous* control step
        # Shape: (n_voxels, 4 directions (N,E,S,W), n_comm_channels)
        # Initialize with zeros
        self.previous_comm_outputs = np.zeros((self.n_voxels, 4, self.n_comm))
        # To store outputs calculated in the current step before they become 'previous'
        self.current_comm_outputs_buffer = np.zeros((self.n_voxels, 4, self.n_comm))
        # ---------------------

        print(f"Controller initialized:")
        print(f"  Num Voxels: {self.n_voxels}")
        print(f"  Sensors/Voxel: {self.n_sensors}")
        print(f"  Comm Channels/Side: {self.n_comm}")
        print(f"  MLP Input Size: {self.input_size}")
        print(f"  MLP Output Size: {self.output_size}")
        if self.driving_voxel_index != -1:
            print(f"  Driving Voxel Index: {self.driving_voxel_index} (Coord: {self.driving_voxel_coord})")
        else:
             print("  No driving voxel specified or found.")


    def get_neighbor_comm_input(self, voxel_index, direction_offset):
        """
        Gets the communication input vector from a neighbor from the *previous* step.
        Handles boundary conditions (returns zeros if neighbor doesn't exist).
        """
        current_coord = self.voxel_coords[voxel_index]
        neighbor_coord = (current_coord[0] + direction_offset[0],
                          current_coord[1] + direction_offset[1],
                          current_coord[2] + direction_offset[2])

        if neighbor_coord in self.voxel_coord_to_index:
            neighbor_index = self.voxel_coord_to_index[neighbor_coord]
            # Determine which output side of the neighbor corresponds to our input side
            # Example: Our North input comes from neighbor's South output
            if direction_offset == (0, 0, 1):   # Our North input (Z+)
                neighbor_output_side_index = 2 # Neighbor's South output (Z-)
            elif direction_offset == (1, 0, 0): # Our East input (X+)
                neighbor_output_side_index = 3 # Neighbor's West output (X-)
            elif direction_offset == (0, 0, -1):# Our South input (Z-)
                neighbor_output_side_index = 0 # Neighbor's North output (Z+)
            elif direction_offset == (-1, 0, 0):# Our West input (X-)
                neighbor_output_side_index = 1 # Neighbor's East output (X+)
            # --- Add Y neighbors if needed ---
            elif direction_offset == (0, 1, 0): # Our Up input (Y+) - Assuming Y is up/down for communication axes
                 neighbor_output_side_index = -1 # Need to define mapping if Y is used
                 print("WARNING: Y-direction neighbor communication not fully defined")
                 return np.zeros(self.n_comm) # Placeholder
            elif direction_offset == (0, -1, 0):# Our Down input (Y-)
                 neighbor_output_side_index = -1 # Need to define mapping if Y is used
                 print("WARNING: Y-direction neighbor communication not fully defined")
                 return np.zeros(self.n_comm) # Placeholder
            # ---------------------------------
            else:
                raise ValueError(f"Invalid direction offset: {direction_offset}")

            # Return the specific communication channel values from the neighbor's previous output
            return self.previous_comm_outputs[neighbor_index, neighbor_output_side_index, :]
        else:
            # Neighbor doesn't exist (boundary)
            return np.zeros(self.n_comm)

    def step(self, sensor_data_all_voxels, time):
        """
        Performs one control step for all voxels.

        Args:
            sensor_data_all_voxels (np.ndarray): Array of sensor data, shape (n_voxels, n_sensors).
                                                 The order must match self.voxel_coords.
            time (float): Current simulation time.

        Returns:
            np.ndarray: Array of actuation signals, shape (n_voxels, 1). Range [-1, 1].
        """
        if sensor_data_all_voxels.shape != (self.n_voxels, self.n_sensors):
            raise ValueError(f"Incorrect sensor data shape. Expected {(self.n_voxels, self.n_sensors)}, got {sensor_data_all_voxels.shape}")

        actuation_signals = np.zeros((self.n_voxels, 1))
        self.current_comm_outputs_buffer.fill(0) # Reset buffer for this step's outputs

        # --- Calculate Driving Signal ---
        # Use a simple sine wave for the driving signal (as in the paper)
        driving_freq = 1.0 # Hz (from paper)
        driving_signal_value = math.sin(2.0 * math.pi * driving_freq * time)
        # -----------------------------

        # --- Iterate through each active voxel ---
        for i in range(self.n_voxels):
            # 1. Get Sensor Data
            local_sensors = sensor_data_all_voxels[i, :]

            # 2. Get Communication Inputs from Previous Step
            # Define neighbor directions relative to voxel coordinates (X, Y, Z)
            # Assuming Z is North/South, X is East/West for communication. Adapt if needed.
            comm_input_N = self.get_neighbor_comm_input(i, (0, 0, 1))  # Z+
            comm_input_E = self.get_neighbor_comm_input(i, (1, 0, 0))  # X+
            comm_input_S = self.get_neighbor_comm_input(i, (0, 0, -1)) # Z-
            comm_input_W = self.get_neighbor_comm_input(i, (-1, 0, 0)) # X-
            # Flatten communication inputs
            comm_inputs_flat = np.concatenate([comm_input_N, comm_input_E, comm_input_S, comm_input_W])

            # 3. Get Driving Signal Input
            driving_input = 0.0
            if i == self.driving_voxel_index:
                driving_input = driving_signal_value

            # 4. Construct Full MLP Input Vector
            mlp_input = np.concatenate([local_sensors, comm_inputs_flat, [driving_input]])
            if mlp_input.shape[0] != self.input_size:
                 raise RuntimeError(f"MLP input size mismatch for voxel {i}. Expected {self.input_size}, got {mlp_input.shape[0]}")


            # 5. Run MLP Forward Pass
            # output = activation(weights @ input + biases)
            output_raw = self.weights @ mlp_input + self.biases
            mlp_output = tanh(output_raw)

            # 6. Parse MLP Output
            # First element is actuation
            actuation_signals[i, 0] = mlp_output[0]

            # Remaining elements are communication outputs for N, E, S, W
            comm_out_N = mlp_output[1 : 1 + self.n_comm]
            comm_out_E = mlp_output[1 + self.n_comm : 1 + 2 * self.n_comm]
            comm_out_S = mlp_output[1 + 2 * self.n_comm : 1 + 3 * self.n_comm]
            comm_out_W = mlp_output[1 + 3 * self.n_comm : 1 + 4 * self.n_comm]

            # Store these outputs in the buffer for the *next* step's inputs
            self.current_comm_outputs_buffer[i, 0, :] = comm_out_N
            self.current_comm_outputs_buffer[i, 1, :] = comm_out_E
            self.current_comm_outputs_buffer[i, 2, :] = comm_out_S
            self.current_comm_outputs_buffer[i, 3, :] = comm_out_W
            # --- Add Y comm outputs if needed ---

        # --- End of Voxel Loop ---

        # Make the buffered outputs the 'previous' outputs for the next control step
        self.previous_comm_outputs = np.copy(self.current_comm_outputs_buffer)

        return actuation_signals

    def load_parameters(self, weights, biases):
        """ Loads evolved weights and biases. """
        if weights.shape != self.weights.shape or biases.shape != self.biases.shape:
            raise ValueError("Shape mismatch when loading parameters.")
        self.weights = weights
        self.biases = biases

    def get_parameters(self):
        """ Returns current weights and biases. """
        return self.weights, self.biases