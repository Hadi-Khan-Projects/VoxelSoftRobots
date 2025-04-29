# controller.py
import numpy as np
import math

def tanh(x):
    """ Element-wise tanh activation function. """
    return np.tanh(x)

class DistributedNeuralController:
    """
    Distributed Neural Controller where each voxel has an identical MLP.
    This version includes time signals (sin(t), cos(t)) as inputs to each voxel's MLP
    to help maintain activity with slower control rates.
    """
    # Define the number of time-based inputs (sin(t), cos(t))
    N_TIME_INPUTS = 2

    def __init__(self, n_voxels, voxel_coords, n_sensors_per_voxel, n_comm_channels, driving_voxel_coord=None, time_signal_frequency=1.0):
        """
        Initializes the distributed neural controller.

        Args:
            n_voxels (int): Total number of active voxels.
            voxel_coords (list): List of (x, y, z) tuples for active voxels.
            n_sensors_per_voxel (int): Number of sensor inputs per voxel.
            n_comm_channels (int): Number of communication values exchanged per side (nc).
            driving_voxel_coord (tuple, optional): Coordinates (x, y, z) of the voxel receiving the central driving signal. Defaults to None.
            time_signal_frequency (float): Frequency (Hz) for the sin(t)/cos(t) time inputs.
        """
        if not isinstance(voxel_coords, list):
             raise TypeError("voxel_coords must be a list of tuples.")
        if n_voxels != len(voxel_coords):
             raise ValueError("n_voxels must match the length of voxel_coords.")

        self.n_voxels = n_voxels
        self.voxel_coords = voxel_coords # Already a list from simulation.py
        self.voxel_coord_to_index = {coord: i for i, coord in enumerate(self.voxel_coords)}
        self.n_sensors = n_sensors_per_voxel
        self.n_comm = n_comm_channels
        self.driving_voxel_coord = driving_voxel_coord
        self.driving_voxel_index = -1
        if self.driving_voxel_coord and self.driving_voxel_coord in self.voxel_coord_to_index:
             self.driving_voxel_index = self.voxel_coord_to_index[self.driving_voxel_coord]

        # MLP Input Size: Sensors + 4*Comm_Inputs + Driving_Signal + Time_Signals
        self.input_size = self.n_sensors + 4 * self.n_comm + 1 + self.N_TIME_INPUTS
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

        # Store frequency for time signals
        self.time_signal_frequency = time_signal_frequency

        print(f"Controller initialized:")
        print(f"  Num Voxels: {self.n_voxels}")
        print(f"  Sensors/Voxel: {self.n_sensors}")
        print(f"  Comm Channels/Side: {self.n_comm}")
        print(f"  Time Signal Inputs: {self.N_TIME_INPUTS} (Freq: {self.time_signal_frequency} Hz)")
        print(f"  MLP Input Size: {self.input_size}")
        print(f"  MLP Output Size: {self.output_size}")
        if self.driving_voxel_index != -1:
            print(f"  Driving Voxel Index: {self.driving_voxel_index} (Coord: {self.driving_voxel_coord})")
        else:
             print("  No specific driving voxel specified or found.")


    def get_neighbor_comm_input(self, voxel_index, direction_offset):
        """
        Gets the communication input vector from a neighbor from the *previous* step.
        Handles boundary conditions (returns zeros if neighbor doesn't exist).

        Args:
            voxel_index (int): Index of the current voxel in self.voxel_coords.
            direction_offset (tuple): (dx, dy, dz) indicating neighbor direction.

        Returns:
            np.ndarray: Communication vector from the specified neighbor (size n_comm),
                        or zeros if the neighbor doesn't exist.
        """
        if not (0 <= voxel_index < self.n_voxels):
            raise IndexError("voxel_index out of bounds.")

        current_coord = self.voxel_coords[voxel_index]
        neighbor_coord = (current_coord[0] + direction_offset[0],
                          current_coord[1] + direction_offset[1],
                          current_coord[2] + direction_offset[2])

        if neighbor_coord in self.voxel_coord_to_index:
            neighbor_index = self.voxel_coord_to_index[neighbor_coord]

            # Determine which output side of the neighbor corresponds to our input side
            # Mapping assumes Z is North(+/up)/South(-/down), X is East(+)/West(-)
            # Define the mapping from our input direction to the neighbor's output side index
            # Neighbor output sides: 0:N(Z+), 1:E(X+), 2:S(Z-), 3:W(X-)
            if direction_offset == (0, 0, 1):   # Our North input (Z+) needs neighbor's South output (Z-)
                neighbor_output_side_index = 2
            elif direction_offset == (1, 0, 0): # Our East input (X+) needs neighbor's West output (X-)
                neighbor_output_side_index = 3
            elif direction_offset == (0, 0, -1):# Our South input (Z-) needs neighbor's North output (Z+)
                neighbor_output_side_index = 0
            elif direction_offset == (-1, 0, 0):# Our West input (X-) needs neighbor's East output (X+)
                neighbor_output_side_index = 1
            # --- Add Y neighbors if needed ---
            # Example: Assuming Y is another axis pair for communication
            # elif direction_offset == (0, 1, 0): # Our 'Top' input (Y+) needs neighbor's 'Bottom' output (Y-)
            #     neighbor_output_side_index = 5 # Define index 5 for Y- output
            # elif direction_offset == (0, -1, 0):# Our 'Bottom' input (Y-) needs neighbor's 'Top' output (Y+)
            #     neighbor_output_side_index = 4 # Define index 4 for Y+ output
            else:
                 # If Y is not used for comm, or invalid offset
                 # print(f"Warning: Communication direction {direction_offset} not handled.")
                 return np.zeros(self.n_comm) # Return zeros for unhandled directions

            # Check if the calculated index is valid for the stored outputs
            if 0 <= neighbor_output_side_index < self.previous_comm_outputs.shape[1]:
                return self.previous_comm_outputs[neighbor_index, neighbor_output_side_index, :]
            else:
                 # This case should ideally not happen with correct mapping
                 print(f"Warning: Invalid neighbor output side index {neighbor_output_side_index} calculated.")
                 return np.zeros(self.n_comm)
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

        # --- Calculate Driving Signal (if applicable) ---
        # Use a simple sine wave for the driving signal (as in the paper)
        driving_freq = 1.0 # Hz (from paper)
        driving_signal_value = math.sin(2.0 * math.pi * driving_freq * time)
        # -----------------------------

        # --- Calculate Time Signals ---
        # Using sin/cos makes the signal periodic and bounded, often easier for NNs
        time_signal_sin = math.sin(2.0 * math.pi * self.time_signal_frequency * time)
        time_signal_cos = math.cos(2.0 * math.pi * self.time_signal_frequency * time)
        time_inputs = np.array([time_signal_sin, time_signal_cos])
        # -----------------------------

        # --- Iterate through each active voxel ---
        for i in range(self.n_voxels):
            # 1. Get Sensor Data
            local_sensors = sensor_data_all_voxels[i, :]

            # 2. Get Communication Inputs from Previous Step
            # Define neighbor directions relative to voxel coordinates (X, Y, Z)
            # Using Z for N/S, X for E/W
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
            mlp_input = np.concatenate([
                local_sensors,
                comm_inputs_flat,
                [driving_input],
                time_inputs # Added time signals
            ])
            if mlp_input.shape[0] != self.input_size:
                 raise RuntimeError(f"MLP input size mismatch for voxel {i}. Expected {self.input_size}, got {mlp_input.shape[0]}")

            # 5. Run MLP Forward Pass
            # output = activation(weights @ input + biases)
            output_raw = self.weights @ mlp_input + self.biases
            mlp_output = tanh(output_raw) # Apply activation

            # 6. Parse MLP Output
            # First element is actuation
            actuation_signals[i, 0] = mlp_output[0]

            # Remaining elements are communication outputs for N, E, S, W
            comm_out_N = mlp_output[1 : 1 + self.n_comm]
            comm_out_E = mlp_output[1 + self.n_comm : 1 + 2 * self.n_comm]
            comm_out_S = mlp_output[1 + 2 * self.n_comm : 1 + 3 * self.n_comm]
            comm_out_W = mlp_output[1 + 3 * self.n_comm : 1 + 4 * self.n_comm]

            # Store these outputs in the buffer for the *next* step's inputs
            # Indices match the neighbor logic: 0:N(Z+), 1:E(X+), 2:S(Z-), 3:W(X-)
            self.current_comm_outputs_buffer[i, 0, :] = comm_out_N
            self.current_comm_outputs_buffer[i, 1, :] = comm_out_E
            self.current_comm_outputs_buffer[i, 2, :] = comm_out_S
            self.current_comm_outputs_buffer[i, 3, :] = comm_out_W
            # --- Add Y comm outputs if needed (indices 4, 5) ---

        # --- End of Voxel Loop ---

        # Make the buffered outputs the 'previous' outputs for the next control step
        self.previous_comm_outputs = np.copy(self.current_comm_outputs_buffer)

        return actuation_signals

    def load_parameters(self, weights, biases):
        """
        Loads evolved/trained weights and biases into the controller.

        Args:
            weights (np.ndarray): The weight matrix (shape: [output_size, input_size]).
            biases (np.ndarray): The bias vector (shape: [output_size]).
        """
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch. Expected {self.weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
             raise ValueError(f"Biases shape mismatch. Expected {self.biases.shape}, got {biases.shape}")
        self.weights = np.copy(weights)
        self.biases = np.copy(biases)
        print("Controller parameters loaded.")

    def get_parameters(self):
        """
        Returns the current weights and biases of the controller's MLP.

        Returns:
            tuple: (weights, biases)
        """
        return self.weights, self.biases

    def get_parameter_vector(self):
        """ Flattens weights and biases into a single vector for optimization algorithms. """
        return np.concatenate([self.weights.flatten(), self.biases.flatten()])

    def set_parameter_vector(self, param_vector):
        """ Sets weights and biases from a flattened vector. """
        expected_len = self.weights.size + self.biases.size
        if len(param_vector) != expected_len:
            raise ValueError(f"Parameter vector length mismatch. Expected {expected_len}, got {len(param_vector)}")

        weights_flat_len = self.weights.size
        self.weights = param_vector[:weights_flat_len].reshape(self.weights.shape)
        self.biases = param_vector[weights_flat_len:]