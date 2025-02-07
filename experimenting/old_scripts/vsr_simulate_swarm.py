import math
import time
import xml.etree.ElementTree as ET
import random
from concurrent.futures import ProcessPoolExecutor

import mujoco
import mujoco.viewer

from vsr import VoxelRobot

# --------------------------
# GLOBAL CONFIG
# --------------------------
MODEL = "quadruped_v3"
FILEPATH = f"vsr_models/{MODEL}/{MODEL}"
ORIGINAL_XML_PATH = FILEPATH + ".xml"
MODIFIED_XML_PATH = FILEPATH + "_modded.xml"
TIMESTEP = "0.001"

# Simulation parameters
DURATION = 30.0        # how long to simulate each particle
NUM_PARALLEL = 10       # how many simulations to run in parallel

# PSO parameters
NUM_PARTICLES = 10     # population size
NUM_ITERATIONS = 200   # how many PSO iterations
w = 0.5                # inertia weight
c1 = 1.5               # cognitive (pbest) coefficient
c2 = 1.5               # social (gbest) coefficient

# Phase range
PHASE_MIN = 0.0
PHASE_MAX = 2 * math.pi

# Velocity range for initialization (you can tune this)
VEL_MIN = -0.5
VEL_MAX = 0.5

# ------------------------------------------------
# Build your VoxelRobot + parse CSV, etc.
# ------------------------------------------------
vsr = VoxelRobot(10, 10, 10)
vsr.load_model_csv(FILEPATH + ".csv")

# Generate the direct flexcomp points/elements
point, element = vsr.generate_model()

# Build a minimal MuJoCo XML for direct flexcomp (for reference)
xml_string = f"""
<mujoco>
    <compiler autolimits="true"/>
    <include file="scene.xml"/>
    <compiler autolimits="true"/>
    <option solver="Newton" tolerance="1e-6" timestep="{TIMESTEP}" integrator="implicitfast"/>
    <size memory="2000M"/>

    <worldbody>
        <flexcomp name="vsr" type="direct" dim="3"
            point="{point}"
            element="{element}"
            radius="0.005" rgba="0.1 0.9 0.1 1" mass="30">
            <contact condim="3" solref="0.01 1" solimp="0.95 0.99 0.0001" selfcollide="none"/>
            <edge damping="1"/>
            <elasticity young="5e2" poisson="0.3"/>
        </flexcomp>
    </worldbody>
</mujoco>
"""

# Create a temporary model just to save as XML
model_tmp = mujoco.MjModel.from_xml_string(xml_string)
data_tmp = mujoco.MjData(model_tmp)
mujoco.mj_saveLastXML(filename=FILEPATH + ".xml", m=model_tmp)

# ------------------------------------------------
# Modify original XML to add motors
# ------------------------------------------------
tree = ET.parse(ORIGINAL_XML_PATH)
root = tree.getroot()

all_joints = []
for body in root.findall("./worldbody/body"):
    body_pos = body.get("pos", "")
    if body_pos:
        pos_str = body_pos.replace(" ", "_")
        joints = body.findall("joint")
        if len(joints) == 3:
            joints[0].set("name", f"vsr_{pos_str}_x")
            joints[1].set("name", f"vsr_{pos_str}_y")
            joints[2].set("name", f"vsr_{pos_str}_z")
            all_joints.append((f"vsr_{pos_str}_x", pos_str))
            all_joints.append((f"vsr_{pos_str}_y", pos_str))
            all_joints.append((f"vsr_{pos_str}_z", pos_str))
        else:
            print(f"Warning: body at position {body_pos} does not have exactly 3 joints.")

actuator = root.find("actuator")
if actuator is None:
    actuator = ET.SubElement(root, "actuator")

for joint_name, pos_str in all_joints:
    motor = ET.SubElement(actuator, "motor")
    motor.set("name", f"{joint_name}_motor")
    motor.set("joint", joint_name)
    motor.set("gear", "1")

tree.write(MODIFIED_XML_PATH, encoding="utf-8", xml_declaration=True)
print(f"Modified XML saved to {MODIFIED_XML_PATH}")

# ------------------------------------------------
# Build simulation function
# ------------------------------------------------
def build_simulation(xml_path=MODIFIED_XML_PATH):
    tmp_model = mujoco.MjModel.from_xml_path(xml_path)
    tmp_data = mujoco.MjData(tmp_model)
    return tmp_model, tmp_data

# ------------------------------------------------
# Simulation function
# ------------------------------------------------
def run_simulation(phases, duration=DURATION, amplitude=0.5, frequency=1.0):
    """Build a fresh model, run the simulation with the given phases, return distance traveled."""
    model, data = build_simulation()
    mujoco.mj_resetData(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vsr_0")
    init_pos = data.xpos[body_id].copy()

    while data.time < duration:
        for key, (phase_x, phase_y, phase_z) in phases.items():
            x_motor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor"
            )
            y_motor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor"
            )
            z_motor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor"
            )

            # Apply sinusoidal control
            if x_motor_id >= 0:
                data.ctrl[x_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data.time + phase_x
                )
            if y_motor_id >= 0:
                data.ctrl[y_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data.time + phase_y
                )
            if z_motor_id >= 0:
                data.ctrl[z_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data.time + phase_z
                )

        mujoco.mj_step(model, data)

    final_pos = data.xpos[body_id]
    dist = math.sqrt(
        (final_pos[0] - init_pos[0])**2 +
        (final_pos[1] - init_pos[1])**2 +
        (final_pos[2] - init_pos[2])**2
    )
    return dist

# ------------------------------------------------
# PSO Helpers
# ------------------------------------------------

def clamp(val, min_val=PHASE_MIN, max_val=PHASE_MAX):
    """Clamp a scalar value to [min_val, max_val]."""
    return max(min_val, min(val, max_val))

T_X = random.uniform(PHASE_MIN, PHASE_MAX)
T_Y = random.uniform(PHASE_MIN, PHASE_MAX)
T_Z = random.uniform(PHASE_MIN, PHASE_MAX)

def random_phases_for_vsr(vsr):
    """
    Returns a dict: { (x,y,z): (px,py,pz) }
    with each px,py,pz in [PHASE_MIN, PHASE_MAX].
    """
    phases = {}
    for key in vsr.point_dict.keys():
        px = T_X
        py = T_Y 
        pz = T_Z
        phases[key] = (px, py, pz)
    return phases

def random_velocity_for_vsr(vsr):
    """
    Returns a dict: { (x,y,z): (vx,vy,vz) }
    with each vx,vy,vz in [VEL_MIN, VEL_MAX].
    """
    velocity = {}
    for key in vsr.point_dict.keys():
        vx = random.uniform(VEL_MIN, VEL_MAX)
        vy = random.uniform(VEL_MIN, VEL_MAX)
        vz = random.uniform(VEL_MIN, VEL_MAX)
        velocity[key] = (vx, vy, vz)
    return velocity

def evaluate_swarm(swarm):
    """
    Evaluate each particle in parallel.
    swarm is a dict of particle_id -> dict with "phases", etc.
    Returns {particle_id: fitness}.
    """
    tasks = [(p_id, swarm[p_id]["phases"]) for p_id in swarm.keys()]
    results = {}

    with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        future_to_pid = {}
        for (p_id, phases) in tasks:
            fut = executor.submit(run_simulation, phases)
            future_to_pid[fut] = p_id

        for fut in future_to_pid:
            p_id = future_to_pid[fut]
            results[p_id] = fut.result()

    return results

# ------------------------------------------------
# Particle Swarm Optimization
# ------------------------------------------------
def particle_swarm_optimization(vsr,
                                num_particles=NUM_PARTICLES,
                                num_iterations=NUM_ITERATIONS):
    """
    Runs a PSO for phase optimization, printing details each iteration.
    """

    # 1) Initialize the swarm
    # For each particle:
    #   - "phases": current position
    #   - "velocity": current velocity
    #   - "pbest_phases": best phases found so far
    #   - "pbest_fitness": best fitness found so far
    swarm = {}
    for i in range(num_particles):
        swarm[i] = {
            "phases": random_phases_for_vsr(vsr),
            "velocity": random_velocity_for_vsr(vsr),
            "pbest_phases": None,
            "pbest_fitness": -1e9
        }

    # 2) Evaluate the swarm initially
    fitnesses = evaluate_swarm(swarm)

    # Update personal best for each particle
    for p_id, fit in fitnesses.items():
        swarm[p_id]["pbest_phases"] = swarm[p_id]["phases"]
        swarm[p_id]["pbest_fitness"] = fit

    # Track global best
    gbest_id = max(fitnesses, key=fitnesses.get)
    gbest_fit = fitnesses[gbest_id]
    gbest_phases = swarm[gbest_id]["phases"]

    print(f"Initial global best: {gbest_fit:.4f} (particle={gbest_id})")

    # 3) Main PSO loop
    for it in range(num_iterations):
        print(f"\n=== PSO Iteration {it+1}/{num_iterations} ===")

        # For debugging: print top few from last iteration
        sorted_ids = sorted(fitnesses, key=fitnesses.get, reverse=True)
        top3 = [(pid, fitnesses[pid]) for pid in sorted_ids]
        print(" Top from previous iteration:")
        for pid, fval in top3:
            print(f"   Particle {pid} => dist = {fval:.4f}")

        # (a) Update velocity & position
        for p_id in swarm:
            # Current position & velocity
            phases = swarm[p_id]["phases"]
            velocity = swarm[p_id]["velocity"]
            pbest = swarm[p_id]["pbest_phases"]

            new_phases = {}
            new_velocity = {}

            # For each voxel (x,y,z)
            for key in phases.keys():
                (px, py, pz) = phases[key]
                (vx, vy, vz) = velocity[key]
                (pbest_x, pbest_y, pbest_z) = pbest[key]
                (gbest_x, gbest_y, gbest_z) = gbest_phases[key]

                # Random factors
                r1 = random.random()
                r2 = random.random()

                # Update velocity component-wise
                vx_new = (w * vx
                          + c1 * r1 * (pbest_x - px)
                          + c2 * r2 * (gbest_x - px))
                vy_new = (w * vy
                          + c1 * r1 * (pbest_y - py)
                          + c2 * r2 * (gbest_y - py))
                vz_new = (w * vz
                          + c1 * r1 * (pbest_z - pz)
                          + c2 * r2 * (gbest_z - pz))

                # Update position
                px_new = px + vx_new
                py_new = py + vy_new
                pz_new = pz + vz_new

                # Clamp
                px_new = clamp(px_new)
                py_new = clamp(py_new)
                pz_new = clamp(pz_new)

                # Store
                new_phases[key] = (px_new, py_new, pz_new)
                new_velocity[key] = (vx_new, vy_new, vz_new)

            # Overwrite swarm with updates
            swarm[p_id]["phases"] = new_phases
            swarm[p_id]["velocity"] = new_velocity

        # (b) Evaluate swarm after move
        fitnesses = evaluate_swarm(swarm)

        # (c) Update personal best & global best
        for p_id, fit in fitnesses.items():
            # update personal best
            if fit > swarm[p_id]["pbest_fitness"]:
                swarm[p_id]["pbest_fitness"] = fit
                swarm[p_id]["pbest_phases"] = swarm[p_id]["phases"]

            # update global best
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_id = p_id
                gbest_phases = swarm[p_id]["phases"]

        print(f"Iteration {it+1} complete. Best so far: {gbest_fit:.4f} (particle={gbest_id})")

    print("\nPSO complete!")
    print(f"Final global best distance: {gbest_fit:.4f}")
    return gbest_phases, gbest_fit


# ------------------------------------------------
# Run PSO & visualize
# ------------------------------------------------
if __name__ == "__main__":
    best_phases, best_fitness = particle_swarm_optimization(
        vsr,
        num_particles=NUM_PARTICLES,
        num_iterations=NUM_ITERATIONS
    )
    print(f"\nBest solution traveled: {best_fitness:.4f}")

    # Visualization
    model_view, data_view = build_simulation()
    viewer = mujoco.viewer.launch_passive(model_view, data_view)

    body_id = mujoco.mj_name2id(model_view, mujoco.mjtObj.mjOBJ_BODY, "vsr_0")
    init_pos = data_view.xpos[body_id].copy()

    sim_start = time.time()
    VIS_DURATION = 5.0
    amplitude = 0.5
    frequency = 1.0

    while (time.time() - sim_start) < VIS_DURATION:
        for key, (phase_x, phase_y, phase_z) in best_phases.items():
            x_motor_id = mujoco.mj_name2id(
                model_view, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor"
            )
            y_motor_id = mujoco.mj_name2id(
                model_view, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor"
            )
            z_motor_id = mujoco.mj_name2id(
                model_view, mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor"
            )

            if x_motor_id >= 0:
                data_view.ctrl[x_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data_view.time + phase_x
                )
            if y_motor_id >= 0:
                data_view.ctrl[y_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data_view.time + phase_y
                )
            if z_motor_id >= 0:
                data_view.ctrl[z_motor_id] = amplitude * math.sin(
                    2*math.pi*frequency*data_view.time + phase_z
                )

        mujoco.mj_step(model_view, data_view)
        viewer.sync()

    final_pos = data_view.xpos[body_id]
    dist = math.sqrt(
        (final_pos[0] - init_pos[0])**2 +
        (final_pos[1] - init_pos[1])**2 +
        (final_pos[2] - init_pos[2])**2
    )
    print(f"Final visualization traveled: {dist:.4f}")
    print("Done watching final best policy.")
