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

# GA / Simulation
POP_SIZE = 10          # how many individuals in the population
NUM_GENERATIONS = 200    # how many generations to run
TOURNAMENT_SIZE = 3    # for selection
ELITISM = 1            # how many top individuals to carry over unchanged
DURATION = 30.0         # seconds to simulate each individual
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
MUTATION_STD = 0.3     # standard deviation of mutation for phases in radians
NUM_PARALLEL = 5       # how many simulations to run in parallel

# GA Range for phases
PHASE_MIN = 0.0
PHASE_MAX = 2 * math.pi

# ------------------------------------------------
# Build your VoxelRobot + parse CSV, etc.
# ------------------------------------------------
vsr = VoxelRobot(10, 10, 10)
vsr.load_model(FILEPATH + ".csv")

# Create a 10*10*10 empty vsr
vsr = VoxelRobot(10, 10, 10)
vsr.load_model(FILEPATH + ".csv")

# Generate the direct flexcomp points/elements
point, element = vsr.generate_model()

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

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

mujoco.mj_saveLastXML(filename=FILEPATH + ".xml", m=model)

# ------------------------------------------------
# Modify original XML to have motors
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
            x_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor")
            y_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor")
            z_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor")

            if x_motor_id >= 0:
                data.ctrl[x_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_x)
            if y_motor_id >= 0:
                data.ctrl[y_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_y)
            if z_motor_id >= 0:
                data.ctrl[z_motor_id] = amplitude * math.sin(2*math.pi*frequency*data.time + phase_z)

        mujoco.mj_step(model, data)

    final_pos = data.xpos[body_id]
    dist = math.sqrt((final_pos[0] - init_pos[0])**2 + (final_pos[1] - init_pos[1])**2 + (final_pos[2] - init_pos[2])**2)
    return dist

T_X = random.uniform(PHASE_MIN, PHASE_MAX)
T_Y = random.uniform(PHASE_MIN, PHASE_MAX)
T_Z = random.uniform(PHASE_MIN, PHASE_MAX)

# ------------------------------------------------
# GA Helpers
# ------------------------------------------------
def random_phases_for_vsr(vsr):
    """Generate random phases in [PHASE_MIN, PHASE_MAX] for each voxel vertex."""
    phases = {}
    for key in vsr.point_dict.keys():
        phases[key] = (
            T_X,
            T_Y,
            T_Z,
        )
    return phases

def clamp(val, min_val=PHASE_MIN, max_val=PHASE_MAX):
    return max(min_val, min(val, max_val))

def mutate(phases):
    """Mutate an individual's phases with some probability, using Gaussian noise."""
    mutated = {}
    for key, (px, py, pz) in phases.items():
        new_px, new_py, new_pz = px, py, pz
        if random.random() < MUTATION_PROB:
            new_px += random.gauss(0, MUTATION_STD)
        if random.random() < MUTATION_PROB:
            new_py += random.gauss(0, MUTATION_STD)
        if random.random() < MUTATION_PROB:
            new_pz += random.gauss(0, MUTATION_STD)

        mutated[key] = (clamp(new_px), clamp(new_py), clamp(new_pz))
    return mutated

def crossover(parent1, parent2):
    """
    Uniform crossover at voxel level:
    for each voxel, with CROSSOVER_PROB chance swap phases between parents.
    """
    child1 = {}
    child2 = {}
    for key in parent1.keys():
        if random.random() < CROSSOVER_PROB:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
        else:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
    return child1, child2

def tournament_selection(pop, fitnesses, k=TOURNAMENT_SIZE):
    """Pick best among k random individuals (by ID)."""
    selected = random.sample(list(pop.keys()), k)
    best_id = selected[0]
    best_fit = fitnesses[best_id]
    for s in selected[1:]:
        if fitnesses[s] > best_fit:
            best_id = s
            best_fit = fitnesses[s]
    return best_id

def evaluate_population(pop):
    """
    Evaluate each individual in parallel.
    Return {pop_id: fitness}.
    """
    results = {}
    tasks = [(pop_id, phases) for pop_id, phases in pop.items()]

    with ProcessPoolExecutor(max_workers=NUM_PARALLEL) as executor:
        future_to_idx = {}
        for pop_id, phases in tasks:
            future = executor.submit(run_simulation, phases)
            future_to_idx[future] = pop_id

        for fut in future_to_idx:
            pop_id = future_to_idx[fut]
            results[pop_id] = fut.result()

    return results

# ------------------------------------------------
# Main GA routine (with extra print statements)
# ------------------------------------------------
def genetic_algorithm(vsr,
                      pop_size=POP_SIZE,
                      num_generations=NUM_GENERATIONS):
    """
    Runs a simple GA for phase optimization, printing details each generation.
    """
    # 1) Initialize population
    population = {}
    for i in range(pop_size):
        population[i] = random_phases_for_vsr(vsr)

    # 2) Evaluate population
    fitnesses = evaluate_population(population)

    # Track best
    best_id = max(fitnesses, key=fitnesses.get)
    best_score = fitnesses[best_id]
    print(f"Initial best: {best_score:.4f} (ID={best_id})")

    next_pop_id = pop_size

    for gen in range(num_generations):
        # Print all individuals' distances for debugging:
        print(f"\n=== Generation {gen+1}/{num_generations} - Detailed Fitness ===")
        for pid in sorted(fitnesses.keys()):
            print(f"  Individual {pid}: distance = {fitnesses[pid]:.4f}")

        # Sort by fitness descending
        sorted_ids = sorted(fitnesses, key=fitnesses.get, reverse=True)
        # Print best few
        top_fitnesses = [(pid, fitnesses[pid]) for pid in sorted_ids[:3]]
        print("  --> Top 3:")
        for pid, fit_val in top_fitnesses:
            print(f"      ID={pid}, dist={fit_val:.4f}")
        
        # Elitism
        new_population = {}
        for e in range(ELITISM):
            elite_id = sorted_ids[e]
            new_population[e] = population[elite_id]

        # Fill rest
        while len(new_population) < pop_size:
            parent_id1 = tournament_selection(population, fitnesses)
            parent_id2 = tournament_selection(population, fitnesses)
            p1 = population[parent_id1]
            p2 = population[parent_id2]

            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)

            new_population[next_pop_id] = c1
            next_pop_id += 1
            if len(new_population) < pop_size:
                new_population[next_pop_id] = c2
                next_pop_id += 1

        population = new_population
        fitnesses = evaluate_population(population)

        # track best
        best_id_this_gen = max(fitnesses, key=fitnesses.get)
        best_score_this_gen = fitnesses[best_id_this_gen]
        if best_score_this_gen > best_score:
            best_score = best_score_this_gen
            best_id = best_id_this_gen

        print(f"Generation {gen+1} complete. Best so far: {best_score:.4f} (ID={best_id})")

    print("\nGA complete!")
    print(f"Final best distance: {best_score:.4f}")
    return population[best_id], best_score


# ------------------------------------------------
# Run GA & visualize
# ------------------------------------------------
if __name__ == "__main__":
    best_phases, best_fitness = genetic_algorithm(vsr, pop_size=POP_SIZE, num_generations=NUM_GENERATIONS)
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
            x_motor_id = mujoco.mj_name2id(model_view, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_x_motor")
            y_motor_id = mujoco.mj_name2id(model_view, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_y_motor")
            z_motor_id = mujoco.mj_name2id(model_view, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vsr_{key[0]}_{key[1]}_{key[2]}_z_motor")

            if x_motor_id >= 0:
                data_view.ctrl[x_motor_id] = amplitude * math.sin(2*math.pi*frequency*data_view.time + phase_x)
            if y_motor_id >= 0:
                data_view.ctrl[y_motor_id] = amplitude * math.sin(2*math.pi*frequency*data_view.time + phase_y)
            if z_motor_id >= 0:
                data_view.ctrl[z_motor_id] = amplitude * math.sin(2*math.pi*frequency*data_view.time + phase_z)

        mujoco.mj_step(model_view, data_view)
        viewer.sync()

    final_pos = data_view.xpos[body_id]
    dist = math.sqrt((final_pos[0] - init_pos[0])**2 +
                     (final_pos[1] - init_pos[1])**2 +
                     (final_pos[2] - init_pos[2])**2)
    print(f"Final visualization traveled: {dist:.4f}")
    print("Done watching final best policy.")
