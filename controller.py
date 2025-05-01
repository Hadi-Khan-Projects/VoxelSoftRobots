import math

import numpy as np


def tanh(x):
    """Element-wise tanh activation function."""
    return np.tanh(x)


class DistributedNeuralController:
    """
    Distributed Neural Controller where each voxel has an identical MLP.
    Supports 6-neighbor communication (N, E, S, W, Up, Down).
    Includes time signals (sin(t), cos(t)) as inputs.
    """

    # define the number of time-based inputs (sin(t), cos(t))
    N_TIME_INPUTS = 2
    # define number of communication directions (N, E, S, W, Up, Down)
    N_COMM_DIRECTIONS = 6
    # define indices for communication directions (consistent mapping)
    # output side indices:
    COMM_IDX_N = 0  # Z+ Output
    COMM_IDX_E = 1  # X+ Output
    COMM_IDX_S = 2  # Z- Output
    COMM_IDX_W = 3  # X- Output
    COMM_IDX_U = 4  # Y+ Output (Up)
    COMM_IDX_D = 5  # Y- Output (Down)

    def __init__(
        self,
        n_voxels,
        voxel_coords,
        n_sensors_per_voxel,
        n_comm_channels,
        weights,
        biases,
        driving_voxel_coord=None,
        time_signal_frequency=1.0,
    ):
        """
        Initializes the distributed neural controller.

        Args:
            n_voxels (int): Total number of active voxels.
            voxel_coords (list): List of (x, y, z) tuples for active voxels.
            n_sensors_per_voxel (int): Number of sensor inputs per voxel.
            n_comm_channels (int): Number of communication values exchanged per side (nc).
            weights (np.ndarray): MLP weight matrix.
            biases (np.ndarray): MLP bias vector.
            driving_voxel_coord (tuple, optional): Coordinates (x, y, z) of the voxel receiving the central driving signal. Defaults to None.
            time_signal_frequency (float): Frequency (Hz) for the sin(t)/cos(t) time inputs.
        """
        if not isinstance(voxel_coords, list):
            raise TypeError("voxel_coords must be a list of tuples.")
        if n_voxels != len(voxel_coords):
            raise ValueError("n_voxels must match the length of voxel_coords.")

        self.n_voxels = n_voxels
        self.voxel_coords = voxel_coords
        self.voxel_coord_to_index = {
            coord: i for i, coord in enumerate(self.voxel_coords)
        }
        self.n_sensors = n_sensors_per_voxel
        self.n_comm = n_comm_channels
        self.driving_voxel_coord = driving_voxel_coord
        self.driving_voxel_index = -1
        if (
            self.driving_voxel_coord
            and self.driving_voxel_coord in self.voxel_coord_to_index
        ):
            self.driving_voxel_index = self.voxel_coord_to_index[
                self.driving_voxel_coord
            ]

        self.N_COM_VEL_INPUTS = 3  # vx, vy, vz
        self.N_TARGET_ORIENT_INPUTS = 2 # dx, dy (normalized vector in XY plane)

        # MLP Input Size: Sensors + N_COMM_DIRECTIONS*Comm_Inputs + Driving_Signal + Time_Signals + COM_Vel + Target_Orient
        self.input_size = (
            self.n_sensors
            + self.N_COMM_DIRECTIONS * self.n_comm
            + 1
            + self.N_TIME_INPUTS
            + self.N_COM_VEL_INPUTS
            + self.N_TARGET_ORIENT_INPUTS
        )
        # MLP Output Size: Actuation + N_COMM_DIRECTIONS*Comm_Outputs
        self.output_size = 1 + self.N_COMM_DIRECTIONS * self.n_comm

        # --- Check provided parameters shape ---
        expected_weights_shape = (self.output_size, self.input_size)
        expected_biases_shape = (self.output_size,)
        if weights.shape != expected_weights_shape:
            raise ValueError(
                f"Weights shape mismatch. Expected {expected_weights_shape}, got {weights.shape}"
            )
        if biases.shape != expected_biases_shape:
            raise ValueError(
                f"Biases shape mismatch. Expected {expected_biases_shape}, got {biases.shape}"
            )

        self.weights = weights
        self.biases = biases
        # ---------------------------------------

        # --- State Variables ---
        # Store communication outputs from the *previous* control step
        # shape: (n_voxels, N_COMM_DIRECTIONS, n_comm_channels)
        self.previous_comm_outputs = np.zeros(
            (self.n_voxels, self.N_COMM_DIRECTIONS, self.n_comm)
        )
        # to store outputs calculated in the current step before they become 'previous'
        self.current_comm_outputs_buffer = np.zeros(
            (self.n_voxels, self.N_COMM_DIRECTIONS, self.n_comm)
        )
        # ---------------------

        # store frequency for time signals
        self.time_signal_frequency = time_signal_frequency

        # print(f"Controller initialized:")
        # print(f"  Num Voxels: {self.n_voxels}")
        # print(f"  Sensors/Voxel: {self.n_sensors}")
        # print(f"  Comm Channels/Side: {self.n_comm}")
        # print(f"  Comm Directions: {self.N_COMM_DIRECTIONS} (N, E, S, W, Up, Down)")
        # print(f"  Time Signal Inputs: {self.N_TIME_INPUTS} (Freq: {self.time_signal_frequency} Hz)")
        # print(f"  COM Velocity Inputs: {self.N_COM_VEL_INPUTS}")
        # print(f"  Target Orientation Inputs: {self.N_TARGET_ORIENT_INPUTS}")
        # print(f"  MLP Input Size: {self.input_size}")
        # print(f"  MLP Output Size: {self.output_size}")
        # if self.driving_voxel_index != -1:
        #     print(f"  Driving Voxel Index: {self.driving_voxel_index} (Coord: {self.driving_voxel_coord})")
        # else:
        #      print("  No specific driving voxel specified or found.")

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
        neighbor_coord = (
            current_coord[0] + direction_offset[0],
            current_coord[1] + direction_offset[1],
            current_coord[2] + direction_offset[2],
        )

        if neighbor_coord in self.voxel_coord_to_index:
            neighbor_index = self.voxel_coord_to_index[neighbor_coord]

            # Determine which output side of the neighbor corresponds to our input side
            # mapping uses the COMM_IDX constants defined above for neighbor's output side.
            neighbor_output_side_index = -1  # Default invalid index

            if direction_offset == (
                0,
                0,
                1,
            ):  # Our North input (Z+) needs neighbor's South output (Z-)
                neighbor_output_side_index = self.COMM_IDX_S
            elif direction_offset == (
                1,
                0,
                0,
            ):  # Our East input (X+) needs neighbor's West output (X-)
                neighbor_output_side_index = self.COMM_IDX_W
            elif direction_offset == (
                0,
                0,
                -1,
            ):  # Our South input (Z-) needs neighbor's North output (Z+)
                neighbor_output_side_index = self.COMM_IDX_N
            elif direction_offset == (
                -1,
                0,
                0,
            ):  # Our West input (X-) needs neighbor's East output (X+)
                neighbor_output_side_index = self.COMM_IDX_E
            elif direction_offset == (
                0,
                1,
                0,
            ):  # Our Up input (Y+) needs neighbor's Down output (Y-)
                neighbor_output_side_index = self.COMM_IDX_D
            elif direction_offset == (
                0,
                -1,
                0,
            ):  # Our Down input (Y-) needs neighbor's Up output (Y+)
                neighbor_output_side_index = self.COMM_IDX_U
            else:
                # invalid offset for communication
                print(
                    f"Warning: Communication direction offset {direction_offset} not handled."
                )
                return np.zeros(self.n_comm)

            # Check if calculated index is valid for the stored outputs
            if 0 <= neighbor_output_side_index < self.N_COMM_DIRECTIONS:
                return self.previous_comm_outputs[
                    neighbor_index, neighbor_output_side_index, :
                ]
            else:
                # this case should ideally not happen with correct mapping
                print(
                    f"Warning: Invalid neighbor output side index {neighbor_output_side_index} calculated for offset {direction_offset}."
                )
                return np.zeros(self.n_comm)
        else:
            # neighbor doesn't exist (boundary)
            return np.zeros(self.n_comm)

    def step(self, sensor_data_all_voxels, time, com_velocity, target_orientation_vector):
        """
        Performs one control step for all voxels.

        Args:
            sensor_data_all_voxels (np.ndarray): Shape (n_voxels, n_sensors).
            time (float): Current simulation time.
            com_velocity (np.ndarray): Global COM velocity (shape [3,]).
            target_orientation_vector (np.ndarray): Normalized 2D vector from COM to target (shape [2,]).

        Returns:
            np.ndarray: Array of actuation signals, shape (n_voxels, 1). Range [-1, 1].
        """
        if sensor_data_all_voxels.shape != (self.n_voxels, self.n_sensors):
            raise ValueError(
                f"Incorrect sensor data shape. Expected {(self.n_voxels, self.n_sensors)}, got {sensor_data_all_voxels.shape}"
            )

        actuation_signals = np.zeros((self.n_voxels, 1))
        self.current_comm_outputs_buffer.fill(0)  # Reset buffer for this step's outputs

        # --- Calculate Driving Signal (if applicable) ---
        driving_freq = 1.0  # Hz
        driving_signal_value = math.sin(2.0 * math.pi * driving_freq * time)
        # -----------------------------

        # --- Calculate Time Signals ---
        time_signal_sin = math.sin(2.0 * math.pi * self.time_signal_frequency * time)
        time_signal_cos = math.cos(2.0 * math.pi * self.time_signal_frequency * time)
        time_inputs = np.array([time_signal_sin, time_signal_cos])
        # -----------------------------

        # --- Iterate through each active voxel ---
        for i in range(self.n_voxels):
            # 1. Get Sensor Data
            local_sensors = sensor_data_all_voxels[i, :]

            # 2. Get Communication Inputs from Previous Step for all 6 neighbors
            comm_input_N = self.get_neighbor_comm_input(i, (0, 0, 1))  # Z+
            comm_input_E = self.get_neighbor_comm_input(i, (1, 0, 0))  # X+
            comm_input_S = self.get_neighbor_comm_input(i, (0, 0, -1))  # Z-
            comm_input_W = self.get_neighbor_comm_input(i, (-1, 0, 0))  # X-
            comm_input_U = self.get_neighbor_comm_input(i, (0, 1, 0))  # Y+ (Up)
            comm_input_D = self.get_neighbor_comm_input(i, (0, -1, 0))  # Y- (Down)

            # Flatten communication inputs in a consistent order (N, E, S, W, U, D)
            comm_inputs_flat = np.concatenate(
                [
                    comm_input_N,
                    comm_input_E,
                    comm_input_S,
                    comm_input_W,
                    comm_input_U,
                    comm_input_D,
                ]
            )

            # 3. Get Driving Signal Input
            driving_input = 0.0
            if i == self.driving_voxel_index:
                driving_input = driving_signal_value

            # 4. Construct Full MLP Input Vector
            mlp_input = np.concatenate(
                [
                    local_sensors,
                    comm_inputs_flat,
                    com_velocity,
                    target_orientation_vector,
                    [driving_input],
                    time_inputs,
                ]
            )
            # Runtime check (optional but recommended during debugging)
            if mlp_input.shape[0] != self.input_size:
                raise RuntimeError(
                    f"MLP input size mismatch for voxel {i}. Expected {self.input_size}, "
                    f"got {mlp_input.shape[0]}. Check sensor/comm/vel/target/driving/time components."
                )

            # 5. Run MLP Forward Pass
            output_raw = self.weights @ mlp_input + self.biases
            mlp_output = tanh(output_raw)  # Apply activation

            # 6. Parse MLP Output
            # First element is actuation
            actuation_signals[i, 0] = mlp_output[0]

            # Remaining elements are communication outputs (N, E, S, W, U, D)
            # Extract based on n_comm size and the new output_size
            start_idx = 1
            comm_out_N = mlp_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_E = mlp_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_S = mlp_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_W = mlp_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_U = mlp_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_D = mlp_output[
                start_idx : start_idx + self.n_comm
            ]  # start_idx += self.n_comm # No need to increment last one

            # Store these outputs in the buffer for the *next* step's inputs
            # Use the defined COMM_IDX constants for clarity and consistency
            self.current_comm_outputs_buffer[i, self.COMM_IDX_N, :] = comm_out_N
            self.current_comm_outputs_buffer[i, self.COMM_IDX_E, :] = comm_out_E
            self.current_comm_outputs_buffer[i, self.COMM_IDX_S, :] = comm_out_S
            self.current_comm_outputs_buffer[i, self.COMM_IDX_W, :] = comm_out_W
            self.current_comm_outputs_buffer[i, self.COMM_IDX_U, :] = comm_out_U
            self.current_comm_outputs_buffer[i, self.COMM_IDX_D, :] = comm_out_D

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
        expected_weights_shape = (self.output_size, self.input_size)
        expected_biases_shape = (self.output_size,)
        if weights.shape != expected_weights_shape:
            raise ValueError(
                f"Weight shape mismatch. Expected {expected_weights_shape}, got {weights.shape}"
            )
        if biases.shape != expected_biases_shape:
            raise ValueError(
                f"Biases shape mismatch. Expected {expected_biases_shape}, got {biases.shape}"
            )

        self.weights = np.copy(weights)
        self.biases = np.copy(biases)
        # print("Controller parameters loaded.")

    def get_parameters(self):
        """Returns the current weights and biases."""
        return self.weights, self.biases

    def get_parameter_vector(self):
        """Flattens weights and biases into a single vector."""
        return np.concatenate([self.weights.flatten(), self.biases.flatten()])

    def set_parameter_vector(self, param_vector):
        """Sets weights and biases from a flattened vector."""
        expected_len = self.weights.size + self.biases.size
        if len(param_vector) != expected_len:
            raise ValueError(
                f"Parameter vector length mismatch. Expected {expected_len}, got {len(param_vector)}"
            )

        weights_flat_len = self.weights.size
        self.weights = param_vector[:weights_flat_len].reshape(self.weights.shape)
        self.biases = param_vector[weights_flat_len:]
