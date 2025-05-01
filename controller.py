# controller.py
import math

import numpy as np


# Activation Functions (using numpy)
def tanh(x):
    """Element-wise tanh activation function."""
    return np.tanh(x)


def relu(x):
    """Element-wise ReLU activation function."""
    return np.maximum(0, x)


class DistributedNeuralController:
    """
    Distributed Neural Controller supporting multiple internal network types:
    - 'mlp': Original single linear layer + tanh.
    - 'mlp_plus': Multi-layer perceptron with configurable hidden layers and ReLU activation (tanh output).
    - 'rnn': Simple recurrent neural network (Elman-style) layer + tanh output.

    Supports 6-neighbor communication (N, E, S, W, Up, Down).
    Includes time signals (sin(t), cos(t)) and global state (COM vel, Target orient) as inputs.
    """

    # constants
    N_TIME_INPUTS = 2
    N_COMM_DIRECTIONS = 6
    N_COM_VEL_INPUTS = 3
    N_TARGET_ORIENT_INPUTS = 2

    # communication index mapping (output side indices)
    COMM_IDX_N = 0  # Z+ Output
    COMM_IDX_E = 1  # X+ Output
    COMM_IDX_S = 2  # Z- Output
    COMM_IDX_W = 3  # X- Output
    COMM_IDX_U = 4  # Y+ Output (Up)
    COMM_IDX_D = 5  # Y- Output (Down)

    def __init__(
        self,
        controller_type: str,  # 'mlp', 'mlp_plus', 'rnn'
        n_voxels: int,
        voxel_coords: list,
        n_sensors_per_voxel: int,
        n_comm_channels: int,
        weights: np.ndarray = None,  # can be None initially
        biases: np.ndarray = None,  # can be None initially
        param_vector: np.ndarray = None,  # alternative initialisation
        mlp_plus_hidden_sizes: list = [32, 32],  # MLP+ specific config
        rnn_hidden_size: int = 32,  # RNN specific config
        driving_voxel_coord: tuple = None,
        time_signal_frequency: float = 1.0,
    ):
        """
        Initializes the distributed neural controller.

        Args:
            controller_type (str): Type of internal network ('mlp', 'mlp_plus', 'rnn').
            n_voxels (int): Total number of active voxels.
            voxel_coords (list): List of (x, y, z) tuples for active voxels.
            n_sensors_per_voxel (int): Number of sensor inputs per voxel.
            n_comm_channels (int): Number of communication values exchanged per side (nc).
            weights (np.ndarray, optional): MLP weight matrix (only for 'mlp' type init).
            biases (np.ndarray, optional): MLP bias vector (only for 'mlp' type init).
            param_vector (np.ndarray, optional): A flat vector containing all network parameters.
                                               If provided, overrides weights/biases.
            mlp_plus_hidden_sizes (list, optional): List of hidden layer sizes for 'mlp_plus'.
            rnn_hidden_size (int, optional): Size of the hidden state for 'rnn'.
            driving_voxel_coord (tuple, optional): Coordinates (x, y, z) of the driving voxel.
            time_signal_frequency (float): Frequency (Hz) for sin(t)/cos(t) inputs.
        """
        if controller_type not in ["mlp", "mlp_plus", "rnn"]:
            raise ValueError(f"Invalid controller_type: {controller_type}")
        if not isinstance(voxel_coords, list):
            raise TypeError("voxel_coords must be a list of tuples.")
        if n_voxels != len(voxel_coords):
            raise ValueError("n_voxels must match the length of voxel_coords.")

        self.controller_type = controller_type
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

        # STEP 1: Calculate base input/output sizes (Same for all types)
        # base input size: sensors + comm_inputs + driving_sig + time_sigs + COM_Vel + target_orient
        self.base_input_size = (
            self.n_sensors
            + self.N_COMM_DIRECTIONS * self.n_comm
            + 1  # Driving signal
            + self.N_TIME_INPUTS
            + self.N_COM_VEL_INPUTS
            + self.N_TARGET_ORIENT_INPUTS
        )
        # base output size: actuation + comm_outputs
        self.base_output_size = 1 + self.N_COMM_DIRECTIONS * self.n_comm

        # STEP 2: Network specific setup
        self.params = {}  # dictionary to store parameters (e.g., {'W1': ..., 'b1': ...})
        self.param_shapes = {}  # dictionary to store shapes for unflattening
        self.total_params = 0

        if self.controller_type == "mlp":
            self._setup_mlp()
        elif self.controller_type == "mlp_plus":
            self.mlp_plus_hidden_sizes = mlp_plus_hidden_sizes
            self._setup_mlp_plus()
        elif self.controller_type == "rnn":
            self.rnn_hidden_size = rnn_hidden_size
            self._setup_rnn()

        # STEP 3: Load parameters
        # priority: param_vector > weights/biases
        if param_vector is not None:
            self.set_parameter_vector(param_vector)
        elif (
            self.controller_type == "mlp" and weights is not None and biases is not None
        ):
            # allow direct weight/bias setting only for simple MLP for compatibility
            if (
                weights.shape == self.param_shapes["W"]
                and biases.shape == self.param_shapes["b"]
            ):
                self.params["W"] = np.copy(weights)
                self.params["b"] = np.copy(biases)
            else:
                raise ValueError("Provided weights/biases shape mismatch for MLP.")
        # else: parameters remain uninitialized (will be set by EA via set_parameter_vector)
        # print(f"Controller '{self.controller_type}' initialized with {self.total_params} parameters (uninitialized).")

        # STEP 4: Set state variables

        # communication state (external)
        self.previous_comm_outputs = np.zeros(
            (self.n_voxels, self.N_COMM_DIRECTIONS, self.n_comm)
        )
        self.current_comm_outputs_buffer = np.zeros(
            (self.n_voxels, self.N_COMM_DIRECTIONS, self.n_comm)
        )
        # RNN hidden state (internal, per voxel)
        if self.controller_type == "rnn":
            # shape: (n_voxels, rnn_hidden_size)
            self.rnn_hidden_state = np.zeros((self.n_voxels, self.rnn_hidden_size))
        else:
            self.rnn_hidden_state = None  # Not used

        # store frequency for time signals
        self.time_signal_frequency = time_signal_frequency

        # print(f"Controller '{self.controller_type}' setup complete.")
        # print(f"  Base Input Size: {self.base_input_size}")
        # print(f"  Base Output Size: {self.base_output_size}")
        # print(f"  Total Parameters: {self.total_params}")
        # if self.controller_type == 'mlp_plus':
        #     print(f"  MLP+ Hidden Sizes: {self.mlp_plus_hidden_sizes}")
        # elif self.controller_type == 'rnn':
        #     print(f"  RNN Hidden Size: {self.rnn_hidden_size}")

    def _setup_mlp(self):
        """
        Calculate shapes and total parameters for the simple MLP.
        This is a single linear layer with tanh activation.

        Args:
            None

        Returns:
            None
        """
        weight_shape = (self.base_output_size, self.base_input_size)
        bias_shape = (self.base_output_size,)
        self.param_shapes = {"W": weight_shape, "b": bias_shape}
        self.total_params = np.prod(weight_shape) + np.prod(bias_shape)
        # Initialize with zeros if not loading later
        self.params = {"W": np.zeros(weight_shape), "b": np.zeros(bias_shape)}

    def _setup_mlp_plus(self):
        """
        Calculate shapes and total parameters for the MLP+.
        This is a multi-layer perceptron with ReLU activation for hidden layers
        and tanh activation for the output layer.

        Args:
            None

        Returns:
            None
        """
        self.param_shapes = {}
        self.total_params = 0
        layer_input_size = self.base_input_size

        # Hidden layers
        for i, hidden_size in enumerate(self.mlp_plus_hidden_sizes):
            layer_name = f"hidden_{i}"
            w_shape = (hidden_size, layer_input_size)
            b_shape = (hidden_size,)
            self.param_shapes[f"W_{layer_name}"] = w_shape
            self.param_shapes[f"b_{layer_name}"] = b_shape
            self.total_params += np.prod(w_shape) + np.prod(b_shape)
            layer_input_size = hidden_size  # Output of this layer is input to next

        # Output layer
        w_shape_out = (self.base_output_size, layer_input_size)
        b_shape_out = (self.base_output_size,)
        self.param_shapes["W_out"] = w_shape_out
        self.param_shapes["b_out"] = b_shape_out
        self.total_params += np.prod(w_shape_out) + np.prod(b_shape_out)

        # Initialize with zeros
        self.params = {
            name: np.zeros(shape) for name, shape in self.param_shapes.items()
        }

    def _setup_rnn(self):
        """
        Calculate shapes and total parameters for the simple RNN.
        This is an Elman-style RNN with a single hidden layer.

        Args:
            None

        Returns:
            None
        """
        self.param_shapes = {}
        input_size = self.base_input_size
        hidden_size = self.rnn_hidden_size  # use the stored value
        output_size = self.base_output_size

        # elman RNN cell parameters
        w_ih_shape = (hidden_size, input_size)
        b_ih_shape = (hidden_size,)
        self.param_shapes["W_ih"] = w_ih_shape
        self.param_shapes["b_ih"] = b_ih_shape
        params_ih = np.prod(w_ih_shape) + np.prod(b_ih_shape)  # calculate count

        w_hh_shape = (hidden_size, hidden_size)
        b_hh_shape = (hidden_size,)
        self.param_shapes["W_hh"] = w_hh_shape
        self.param_shapes["b_hh"] = b_hh_shape
        params_hh = np.prod(w_hh_shape) + np.prod(b_hh_shape)  # calculate count

        # output layer parameters
        w_ho_shape = (output_size, hidden_size)
        b_ho_shape = (output_size,)
        self.param_shapes["W_ho"] = w_ho_shape
        self.param_shapes["b_ho"] = b_ho_shape
        params_ho = np.prod(w_ho_shape) + np.prod(b_ho_shape)  # calculate count

        # calculate total
        self.total_params = params_ih + params_hh + params_ho

        # initialise with zeros
        self.params = {
            name: np.zeros(shape) for name, shape in self.param_shapes.items()
        }

    def reset(self):
        """
        Resets communication and RNN hidden states (call before each trial).

        Args:
            None

        Returns:
            None
        """
        self.previous_comm_outputs.fill(0)
        self.current_comm_outputs_buffer.fill(0)
        if self.rnn_hidden_state is not None:
            self.rnn_hidden_state.fill(0)

    def get_neighbor_comm_input(self, voxel_index, direction_offset):
        """
        Gets communication input from a neighbor from the *previous* step.

        Args:
            voxel_index (int): Index of the current voxel.
            direction_offset (tuple): Direction offset to the neighbor (dx, dy, dz).

        Returns:
            np.ndarray: Communication input vector from the neighbor.
        """
        # ... (implementation remains the same as before) ...
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
            neighbor_output_side_index = -1

            if direction_offset == (0, 0, 1):
                neighbor_output_side_index = self.COMM_IDX_S  # N input <- S output
            elif direction_offset == (1, 0, 0):
                neighbor_output_side_index = self.COMM_IDX_W  # E input <- W output
            elif direction_offset == (0, 0, -1):
                neighbor_output_side_index = self.COMM_IDX_N  # S input <- N output
            elif direction_offset == (-1, 0, 0):
                neighbor_output_side_index = self.COMM_IDX_E  # W input <- E output
            elif direction_offset == (0, 1, 0):
                neighbor_output_side_index = self.COMM_IDX_D  # U input <- D output
            elif direction_offset == (0, -1, 0):
                neighbor_output_side_index = self.COMM_IDX_U  # D input <- U output
            else:
                print(f"Warning: Invalid direction offset {direction_offset}")
                return np.zeros(self.n_comm)

            if 0 <= neighbor_output_side_index < self.N_COMM_DIRECTIONS:
                return self.previous_comm_outputs[
                    neighbor_index, neighbor_output_side_index, :
                ]
            else:
                print(
                    f"Warning: Invalid neighbor output side index {neighbor_output_side_index}"
                )
                return np.zeros(self.n_comm)
        else:
            return np.zeros(self.n_comm)  # boundary condition

    def _forward_pass(self, input_vector, voxel_idx):
        """
        Performs the forward pass based on the controller type.

        Args:
            input_vector (np.ndarray): Input vector for the network.
            voxel_idx (int): Index of the current voxel (for RNN state management).

        Returns:
            np.ndarray: Output vector from the network.
        """
        if self.controller_type == "mlp":
            # linear layer -> Tanh
            output_raw = self.params["W"] @ input_vector + self.params["b"]
            return tanh(output_raw)

        elif self.controller_type == "mlp_plus":
            # hidden layers (linear -> ReLU) -> output layer (linear -> Tanh)
            x = input_vector
            for i in range(len(self.mlp_plus_hidden_sizes)):
                layer_name = f"hidden_{i}"
                W = self.params[f"W_{layer_name}"]
                b = self.params[f"b_{layer_name}"]
                x = relu(W @ x + b)
            # output layer
            W_out = self.params["W_out"]
            b_out = self.params["b_out"]
            output_raw = W_out @ x + b_out
            return tanh(output_raw)

        elif self.controller_type == "rnn":
            # simple elman RNN cell -> output layer -> Tanh
            # get previous hidden state for this specific voxel
            prev_hidden = self.rnn_hidden_state[voxel_idx, :]

            # RNN update: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
            W_ih = self.params["W_ih"]
            b_ih = self.params["b_ih"]
            W_hh = self.params["W_hh"]
            b_hh = self.params["b_hh"]
            current_hidden = tanh(
                W_ih @ input_vector + b_ih + W_hh @ prev_hidden + b_hh
            )

            # store the new hidden state for the next step *for this voxel*
            self.rnn_hidden_state[voxel_idx, :] = current_hidden

            # output layer: y_t = W_ho * h_t + b_ho
            W_ho = self.params["W_ho"]
            b_ho = self.params["b_ho"]
            output_raw = W_ho @ current_hidden + b_ho
            return tanh(output_raw)  # Final output activation

        else:
            raise NotImplementedError(
                f"Forward pass not implemented for type: {self.controller_type}"
            )

    def step(
        self, sensor_data_all_voxels, time, com_velocity, target_orientation_vector
    ):
        """
        Performs one control step for all voxels.

        Args:
            sensor_data_all_voxels (np.ndarray): Sensor data for all voxels.
            time (float): Current time.
            com_velocity (np.ndarray): Center of mass velocity.
            target_orientation_vector (np.ndarray): Target orientation vector.

        Returns:
            np.ndarray: Actuation signals for all voxels.
        """
        if sensor_data_all_voxels.shape != (self.n_voxels, self.n_sensors):
            raise ValueError("Incorrect sensor data shape.")

        actuation_signals = np.zeros((self.n_voxels, 1))
        self.current_comm_outputs_buffer.fill(0)  # reset buffer

        # calculate Driving Signal
        driving_freq = 1.0
        driving_signal_value = math.sin(2.0 * math.pi * driving_freq * time)

        # calculate Time Signals
        time_signal_sin = math.sin(2.0 * math.pi * self.time_signal_frequency * time)
        time_signal_cos = math.cos(2.0 * math.pi * self.time_signal_frequency * time)
        time_inputs = np.array([time_signal_sin, time_signal_cos])

        # iterate through each active voxel
        for i in range(self.n_voxels):
            # STEP 1: Get local sensor data
            local_sensors = sensor_data_all_voxels[i, :]

            # STEP 2: Get communication inputs from previous step
            comm_input_N = self.get_neighbor_comm_input(i, (0, 0, 1))
            comm_input_E = self.get_neighbor_comm_input(i, (1, 0, 0))
            comm_input_S = self.get_neighbor_comm_input(i, (0, 0, -1))
            comm_input_W = self.get_neighbor_comm_input(i, (-1, 0, 0))
            comm_input_U = self.get_neighbor_comm_input(i, (0, 1, 0))
            comm_input_D = self.get_neighbor_comm_input(i, (0, -1, 0))
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

            # STEP 3: Get driving signal input
            driving_input = (
                driving_signal_value if i == self.driving_voxel_index else 0.0
            )

            # STEP 4: Construct full input vector for the internal network
            net_input = np.concatenate(
                [
                    local_sensors,
                    comm_inputs_flat,
                    com_velocity,
                    target_orientation_vector,
                    [driving_input],
                    time_inputs,
                ]
            )
            if net_input.shape[0] != self.base_input_size:  # runtime check
                raise RuntimeError(f"Network input size mismatch for voxel {i}.")

            # STEP 5. Run forward pass using the appropriate network
            # pass voxel index 'i' needed for RNN state management
            net_output = self._forward_pass(net_input, i)

            # STEP 6: Parse network output (same structure for all types)
            actuation_signals[i, 0] = net_output[0]

            # Extract communication outputs
            start_idx = 1
            comm_out_N = net_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_E = net_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_S = net_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_W = net_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_U = net_output[start_idx : start_idx + self.n_comm]
            start_idx += self.n_comm
            comm_out_D = net_output[start_idx : start_idx + self.n_comm]  # Last one

            # Store in buffer
            self.current_comm_outputs_buffer[i, self.COMM_IDX_N, :] = comm_out_N
            self.current_comm_outputs_buffer[i, self.COMM_IDX_E, :] = comm_out_E
            self.current_comm_outputs_buffer[i, self.COMM_IDX_S, :] = comm_out_S
            self.current_comm_outputs_buffer[i, self.COMM_IDX_W, :] = comm_out_W
            self.current_comm_outputs_buffer[i, self.COMM_IDX_U, :] = comm_out_U
            self.current_comm_outputs_buffer[i, self.COMM_IDX_D, :] = comm_out_D

        # end of voxel loop

        # update communication state for the next step
        self.previous_comm_outputs = np.copy(self.current_comm_outputs_buffer)

        return actuation_signals

    def get_total_parameter_count(self):
        """
        Returns the total number of parameters for the current network type.

        Args:
            None

        Returns:
            int: Total number of parameters.
        """
        return self.total_params

    def get_parameter_vector(self):
        """
        Flattens all parameters into a single vector based on defined order.

        Args:
            None

        Returns:
            np.ndarray: Flattened parameter vector.
        """
        param_list = []
        # flatten in the canonical order defined by self.param_shapes keys
        for name in self.param_shapes.keys():
            if name in self.params:
                param_list.append(self.params[name].flatten())
            else:
                # this should not happen if setup is correct
                raise KeyError(
                    f"Parameter '{name}' expected but not found in self.params."
                )
        return np.concatenate(param_list)

    def set_parameter_vector(self, param_vector):
        """
        Sets parameters from a flattened vector based on defined order and shapes.

        Args:
            param_vector (np.ndarray): Flattened parameter vector.

        Returns:
            None
        """
        if len(param_vector) != self.total_params:
            raise ValueError(
                f"Parameter vector length mismatch for type '{self.controller_type}'. "
                f"Expected {self.total_params}, got {len(param_vector)}"
            )

        current_idx = 0
        # unflatten in the canonical order defined by self.param_shapes keys
        for name, shape in self.param_shapes.items():
            num_elements = np.prod(shape)
            if current_idx + num_elements > len(param_vector):
                raise ValueError(
                    f"Parameter vector too short when unflattening '{name}'."
                )

            # extract the slice, reshape, and store in the params dictionary
            value = param_vector[current_idx : current_idx + num_elements].reshape(
                shape
            )
            self.params[name] = value
            current_idx += num_elements

        # check if all elements were used
        if current_idx != self.total_params:
            print(
                f"Warning: Parameter vector length mismatch after unflattening. Used {current_idx}/{self.total_params}"
            )

    def get_parameters(self):
        """Returns the dictionary of current parameters (weights/biases)."""
        # return a copy to prevent external modification
        return {k: np.copy(v) for k, v in self.params.items()}

    def load_parameters(self, param_dict):
        """Loads parameters from a dictionary (use with caution, prefers set_parameter_vector)."""
        # basic check to ensure keys match expected shapes
        for name, value in param_dict.items():
            if name in self.param_shapes:
                if value.shape == self.param_shapes[name]:
                    self.params[name] = np.copy(value)
                else:
                    raise ValueError(
                        f"Shape mismatch for parameter '{name}'. Expected {self.param_shapes[name]}, got {value.shape}"
                    )
            else:
                raise ValueError(
                    f"Unexpected parameter name '{name}' for type '{self.controller_type}'."
                )
        # verify all expected params were provided? Optional.
        # print(f"Controller parameters loaded from dictionary for type '{self.controller_type}'.")
