import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MIN_DIM = 1
MAX_DIM = 20


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
        self.grid = np.zeros((x, y, z), dtype=np.uint8)

    def set_val(self, x: int, y: int, z: int, value) -> None:
        """Set the value at position (x, y, z) to 0 or 1."""
        if value not in [0, 1]:
            raise ValueError("VSR's voxel grid value not set as 0 or 1.")
        self.grid[x, y, z] = value

    def visualise_model(self) -> None:
        """Visualise the VSR's structure using matplotlib."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Coordinates of all 1's in the grid
        x, y, z = np.where(self.grid == 1)
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

    def check_contiguous(self) -> bool:
        """Check if the VSR's structure is contiguous."""
        visited = np.zeros_like(self.grid, dtype=bool)
        start_voxel = np.argwhere(self.grid == 1)
        if start_voxel.size == 0:
            return False
        start_voxel = tuple(start_voxel[0])
        self._dfs(start_voxel, visited)
        if not np.array_equal(self.grid == 1, visited):
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
                if 0 <= nx < 10 and 0 <= ny < 10 and 0 <= nz < 10:
                    if self.grid[nx, ny, nz] == 1 and not visited[nx, ny, nz]:
                        stack.append((nx, ny, nz))

    def save_model(self, filename) -> None:
        """Save the current voxel grid to a .csv file of a given name"""
        data = []
        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.grid[x, y, z] == 1:
                        data.append((x, y, z))

        df = pd.DataFrame(data, columns=["x", "y", "z"], dtype=np.uint8)
        df.to_csv(filename, index=False)

    def load_model(self, filename) -> None:
        """Reset the current voxel grid to all zeroes and replace with a .csv file"""
        df = pd.read_csv(filename)
        self.grid = np.zeros((self.max_x, self.max_y, self.max_z), dtype=np.uint8)
        for _, row in df.iterrows():
            self.grid[row["x"], row["y"], row["z"]] = 1

    def generate_model(self) -> tuple:
        """Generate the points and elements for the voxel model."""

        self.check_contiguous()

        points_grid = np.zeros(
            (self.max_x + 1, self.max_y + 1, self.max_z + 1), dtype=np.int8
        )
        points_count = 0
        points_map = {}
        # (0, 1, 5) : 24

        points_string = ""

        for x in range(self.max_x + 1):
            for y in range(self.max_y + 1):
                for z in range(self.max_z + 1):
                    if self._is_point_vertex(x, y, z):
                        points_grid[x, y, z] = 1
                        points_map[(x, y, z)] = points_count
                        points_string += " ".join(map(str, (x, y, z))) + "   "
                        points_count += 1

        element_string = ""

        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.grid[x, y, z] == 1:
                        points = [
                            points_map[(x, y, z)],  # 0
                            points_map[(x + 1, y, z)],  # 1
                            points_map[(x + 1, y + 1, z)],  # 2
                            points_map[(x, y + 1, z)],  # 3
                            points_map[(x, y, z + 1)],  # 4
                            points_map[(x + 1, y, z + 1)],  # 5
                            points_map[(x + 1, y + 1, z + 1)],  # 6
                            points_map[(x, y + 1, z + 1)],  # 7
                        ]

                        elements = [
                            (0, 3, 7, 6),  # (0,0,0), (0,1,0), (0,1,1), (1,1,1)
                            (0, 7, 4, 6),  # (0,0,0), (0,1,1), (0,0,1), (1,1,1)
                            (0, 4, 5, 6),  # (0,0,0), (0,0,1), (1,0,1), (1,1,1)
                            (0, 5, 1, 6),  # (0,0,0), (1,0,1), (1,0,0), (1,1,1)
                            (0, 1, 2, 6),  # (0,0,0), (1,0,0), (1,1,0), (1,1,1)
                            (0, 2, 3, 6),  # (0,0,0), (1,1,0), (0,1,0), (1,1,1)
                        ]

                        for tetrahedron in elements:
                            for vertex in tetrahedron:
                                element_string += " "
                                element_string += str(points[vertex])
                            element_string += "    "

        return points_string.strip(), element_string.strip()

    def _is_point_possible(self, x, y, z) -> bool:
        """Check if (x, y, z) is a valid index in the voxel grid."""
        return 0 <= x < self.max_x and 0 <= y < self.max_y and 0 <= z < self.max_z

    def _is_point_vertex(self, x, y, z) -> bool:
        """Check if (x, y, z) is a vertex of one of the VSR's voxels by checking
        all 8 adjacent possible voxels."""

        return (
            (self._is_point_possible(x, y, z) and self.grid[x, y, z] == 1)
            or (self._is_point_possible(x - 1, y, z) and self.grid[x - 1, y, z] == 1)
            or (
                self._is_point_possible(x - 1, y - 1, z)
                and self.grid[x - 1, y - 1, z] == 1
            )
            or (self._is_point_possible(x, y - 1, z) and self.grid[x, y - 1, z] == 1)
            or (self._is_point_possible(x, y, z - 1) and self.grid[x, y, z - 1] == 1)
            or (
                self._is_point_possible(x - 1, y, z - 1)
                and self.grid[x - 1, y, z - 1] == 1
            )
            or (
                self._is_point_possible(x - 1, y - 1, z - 1)
                and self.grid[x - 1, y - 1, z - 1] == 1
            )
            or (
                self._is_point_possible(x, y - 1, z - 1)
                and self.grid[x, y - 1, z - 1] == 1
            )
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
