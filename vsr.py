import xml.etree.ElementTree as ET
from xml.dom import minidom

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

MIN_DIM = 1
MAX_DIM = 20
TIMESTEP = 0.001
FRICTION_MULT = 1.0
FRICTION_SLIDING = 1.0 * FRICTION_MULT  # default 1
FRICTION_TORSIONAL = 0.005 * FRICTION_MULT  # default 0.005
ROLLING_FRICTION = 0.0001 * FRICTION_MULT  # default 0.0001


class VoxelRobot:
    def __init__(self, x: int, y: int, z: int, gear: float) -> None:
        """
        Initialise an (x, y, z) VSR robot with grid of 0's.

        Args:
            x (int): X dimension of the voxel grid.
            y (int): Y dimension of the voxel grid.
            z (int): Z dimension of the voxel grid.
            gear (float): Gear ratio for the motors.

        Returns:
            None
        """
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
        self.gear = gear

    def set_val(self, x: int, y: int, z: int, value) -> None:
        """
        Set the value at position (x, y, z) to 0 or 1.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            z (int): Z coordinate.
            value (int): Value to set (0 or 1).

        Returns:
            None
        """
        if value not in [0, 1]:
            raise ValueError("VSR's voxel grid value not set as 0 or 1.")
        self.voxel_grid[x, y, z] = value

    def visualise_model(self) -> None:
        """
        Visualise the VSR's structure using matplotlib.

        Args:
            None

        Returns:
            None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # coordinates of all 1's in the grid
        x, y, z = np.where(self.voxel_grid == 1)
        voxel_plot_size = 0.9  # slightly smaller than 1 to avoid overlap

        # plot each voxel as small cube
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
        """
        Check if the VSR's structure is contiguous.

        Args:
            None

        Returns:
            bool: True if the structure is contiguous, False otherwise.
        """
        visited = np.zeros_like(self.voxel_grid, dtype=bool)
        start_voxel = np.argwhere(self.voxel_grid == 1)
        if start_voxel.size == 0:
            return False
        start_voxel = tuple(start_voxel[0])
        self._dfs(start_voxel, visited)
        if not np.array_equal(self.voxel_grid == 1, visited):
            raise ValueError("VSR's structure is not contiguous.")

    def _dfs(self, voxel, visited) -> None:
        """
        Depth first search to check for VSR contiguity.

        Args:
            voxel (tuple): Current voxel coordinates.
            visited (ndarray): Array to keep track of visited voxels.

        Returns:
            None
        """
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
                    if self.voxel_grid[nx, ny, nz] == 1 and not visited[nx, ny, nz]:
                        stack.append((nx, ny, nz))

    def save_model_csv(self, filename) -> None:
        """
        Save the current voxel grid to a .csv file of a given name

        Args:
            filename (str): Name of the file to save the voxel grid.

        Returns:
            None
        """
        data = []
        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.voxel_grid[x, y, z] == 1:
                        data.append((x, y, z))

        df = pd.DataFrame(data, columns=["x", "y", "z"], dtype=np.uint8)
        df.to_csv(filename, index=False)

    def load_model_csv(self, filename) -> None:
        """
        Reset the current voxel grid to all zeroes andload grid from .csv file

        Args:
            filename (str): Name of the file to load the voxel grid from.

        Returns:
            None
        """
        df = pd.read_csv(filename)
        self.voxel_grid = np.zeros((self.max_x, self.max_y, self.max_z), dtype=np.uint8)
        for _, row in df.iterrows():
            self.voxel_grid[row["x"], row["y"], row["z"]] = 1

    def generate_model(self, filepath) -> str:
        """
        Generate the MuJoCo model for the VSR.

        Args:
            filepath (str): File path to save the generated model.

        Returns:
            str: XML string of the generated model.
        """

        self._check_contiguous()
        points_string, elements_string = self._generate_flexcomp_geometry()

        xml_string = f"""
        <mujoco>

            <compiler autolimits="true"/>
            <include file="scene.xml"/>
            <compiler autolimits="true"/>
            <option solver="Newton" tolerance="1e-6" timestep="{str(TIMESTEP)}" integrator="implicitfast"/>
            <size memory="2000M"/>

            <worldbody>

                <body name="target" pos="-300 3 3.5">
                    <geom type="box" size="1 1 3" rgba="1 0 0 0.7"/>
                </body>

                <flexcomp name="vsr" type="direct" dim="3"
                    point="{points_string}"
                    element="{elements_string}"
                    radius="0.005" rgba="0.1 0.9 0.1 1" mass="{self.num_vertex() / 10}">
                    <contact 
                        condim="3" 
                        solref="0.01 1" 
                        solimp="0.95 0.99 0.0001" 
                        selfcollide="none" 
                        friction="{str(FRICTION_SLIDING)} {str(FRICTION_TORSIONAL)} {str(ROLLING_FRICTION)}"
                    />
                    <edge damping="1"/>
                    <elasticity young="250" poisson="0.3"/>
                </flexcomp>

            </worldbody>

        </mujoco>
        """

        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)  # noqa: F841
        mujoco.mj_saveLastXML(filename=filepath + ".xml", m=model)

        # STEP 1: Add sites

        tree = ET.parse(filepath + ".xml")
        root = tree.getroot()

        # add site to each body
        for body in root.findall("./worldbody/body"):
            body_name = body.get("name")
            if not body.get("pos"):
                body.set("pos", "0 0 0")
            if body_name:
                site = ET.SubElement(body, "site")
                site.set("name", f"site_{body.get('pos').replace(' ', '_')}")
                site.set("pos", "0 0 0")
                site.set("size", "0.005")  # small sphere for visualization

        # STEP 2: Add spatial tendons

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
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x}_{y}_{z}_to_{x + 1}_{y + 1}_{z + 1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="1 0 0 1",
                            stiffness="1",
                            damping="0",
                        )
                        # The tendon passes through two sites (vertexes)
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y}_{z}")
                        ET.SubElement(
                            spatial, "site", site=f"site_{x + 1}_{y + 1}_{z + 1}"
                        )
                        motor = ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x}_{y}_{z}_to_{x + 1}_{y + 1}_{z + 1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(self.gear),
                        )

                        # -x -y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x + 1}_{y + 1}_{z}_to_{x}_{y}_{z + 1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="0 1 0 1",
                            stiffness="1",
                            damping="0",
                        )
                        # The tendon passes through two sites (vertexes)
                        ET.SubElement(spatial, "site", site=f"site_{x + 1}_{y + 1}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y}_{z + 1}")
                        motor = ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x + 1}_{y + 1}_{z}_to_{x}_{y}_{z + 1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(self.gear),
                        )

                        # +x -y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x}_{y + 1}_{z}_to_{x + 1}_{y}_{z + 1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="0 0 1 1",
                            stiffness="1",
                            damping="0",
                        )
                        # The tendon passes through two sites (vertexes)
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y + 1}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x + 1}_{y}_{z + 1}")
                        motor = ET.SubElement(
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x}_{y + 1}_{z}_to_{x + 1}_{y}_{z + 1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(self.gear),
                        )

                        # -x +y +z direction
                        t_name = f"voxel_{x}_{y}_{z}_spatial_{x + 1}_{y}_{z}_to_{x}_{y + 1}_{z + 1}"
                        spatial = ET.SubElement(
                            tendon_elem,
                            "spatial",
                            name=t_name,
                            width="0.006",
                            rgba="1 0 1 1",
                            stiffness="1",
                            damping="0",
                        )
                        # The tendon passes through two sites (vertexes)
                        ET.SubElement(spatial, "site", site=f"site_{x + 1}_{y}_{z}")
                        ET.SubElement(spatial, "site", site=f"site_{x}_{y + 1}_{z + 1}")
                        motor = ET.SubElement(  # noqa: F841
                            actuator_elem,
                            "motor",
                            name=f"voxel_{x}_{y}_{z}_motor_{x + 1}_{y}_{z}_to_{x}_{y + 1}_{z + 1}",
                            tendon=t_name,
                            ctrlrange="0 1",
                            gear=str(self.gear),
                        )

        ## STEP 3: Save modded xml model and return

        self._save_modded_xml(tree, filepath)

        return ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")

    def _generate_flexcomp_geometry(self) -> tuple:
        """
        Generate the points and elements for the voxel flexcomp model.

        Args:
            None

        Returns:
            tuple: points_string, elements_string
        """
        self.point_grid = np.zeros(
            (self.max_x + 1, self.max_y + 1, self.max_z + 1), dtype=np.int8
        )
        points_count = 0
        points_map = {}  # e.g (0, 1, 5) : 24
        points_string = ""

        # STEP 1: Iterate through voxel grid and find all vertices
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

        # STEP 2: Iterate through voxel grid and form all tetrahedra
        for x in range(self.max_x):
            for y in range(self.max_y):
                for z in range(self.max_z):
                    if self.voxel_grid[x, y, z] == 1:
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
                                elements_string += " "
                                elements_string += str(points[vertex])
                            elements_string += "    "

        return points_string.strip(), elements_string.strip()

    def _is_point_possible(self, x, y, z) -> bool:
        """
        Check if (x, y, z) is a valid index in the voxel grid.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            z (int): Z coordinate.

        Returns:
            bool: True if the point is within the grid bounds, False otherwise.
        """
        return 0 <= x < self.max_x and 0 <= y < self.max_y and 0 <= z < self.max_z

    def _is_point_vertex(self, x, y, z) -> bool:
        """
        Check if (x, y, z) is a vertex of one of the VSR's voxels by checking
        all 8 adjacent possible voxels.

        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            z (int): Z coordinate.

        Returns:
            bool: True if the point is a vertex, False otherwise.
        """

        return (
            (self._is_point_possible(x, y, z) and self.voxel_grid[x, y, z] == 1)
            or (
                self._is_point_possible(x - 1, y, z)
                and self.voxel_grid[x - 1, y, z] == 1
            )
            or (
                self._is_point_possible(x - 1, y - 1, z)
                and self.voxel_grid[x - 1, y - 1, z] == 1
            )
            or (
                self._is_point_possible(x, y - 1, z)
                and self.voxel_grid[x, y - 1, z] == 1
            )
            or (
                self._is_point_possible(x, y, z - 1)
                and self.voxel_grid[x, y, z - 1] == 1
            )
            or (
                self._is_point_possible(x - 1, y, z - 1)
                and self.voxel_grid[x - 1, y, z - 1] == 1
            )
            or (
                self._is_point_possible(x - 1, y - 1, z - 1)
                and self.voxel_grid[x - 1, y - 1, z - 1] == 1
            )
            or (
                self._is_point_possible(x, y - 1, z - 1)
                and self.voxel_grid[x, y - 1, z - 1] == 1
            )
        )

    def num_vertex(self) -> int:
        """
        Return number of vertexes required to generate the VSR.

        Args:
            None

        Returns:
            int: Number of vertexes.
        """
        points_count = 0

        for x in range(self.max_x + 1):
            for y in range(self.max_y + 1):
                for z in range(self.max_z + 1):
                    if self._is_point_vertex(x, y, z):
                        points_count += 1

        return points_count

    def _save_modded_xml(self, tree, filepath) -> None:
        """
        Formats an XML tree with proper indentation and saves it as filepath + "_modded.xml".

        Args:
            tree (ElementTree): ElementTree object of the XML file.
            filepath (str): Base file path (without extension).
        
        Returns:
            None
        """
        root = tree.getroot()

        # convert tree to a pretty-printed string
        xml_str = ET.tostring(root, encoding="utf-8")

        # use minidom for formatting
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")

        # remove empty lines
        cleaned_xml = "\n".join(
            [line for line in pretty_xml.splitlines() if line.strip()]
        )

        # save the formatted/modded XML to a new file
        new_filepath = filepath + "_modded.xml"
        with open(new_filepath, "w", encoding="utf-8") as f:
            f.write(cleaned_xml)

        print(f"Formatted XML saved to {new_filepath}")
