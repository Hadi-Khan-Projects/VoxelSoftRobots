import numpy as np
from vsr import VoxelRobot

# Create a 10*10*10 numpy grid, populate it with 0's
vsr_grid = np.zeros((10, 10, 10))

# 4*3*1 robot body. 
for i in range(0, 4):
    for j in range(0, 3):
        vsr_grid[i, j, 1] = 1

# Add 1*1*1 legs in each corner of body
vsr_grid[0, 0, 0] = 1
vsr_grid[3, 0, 0] = 1
vsr_grid[0, 2, 0] = 1
vsr_grid[3, 2, 0] = 1

vsr = VoxelRobot(vsr_grid)
vsr.visualise_model()
vsr.generate_model()

# 4*3*1 robot body. 
for i in range(0, 8):
    for j in range(0, 6):
        vsr.set_val(i, j, 2, 1)
        vsr.set_val(i, j, 3, 1)

# Add 1*1*1 legs in each corner of body
for i in range(0, 2):
    vsr.set_val(0, 0, i, 1)
    vsr.set_val(0, 1, i, 1)
    vsr.set_val(1, 0, i, 1)
    vsr.set_val(1, 1, i, 1)

    vsr.set_val(6, 0, i, 1)
    vsr.set_val(6, 1, i, 1)
    vsr.set_val(7, 0, i, 1)
    vsr.set_val(7, 1, i, 1)

    vsr.set_val(0, 4, i, 1)
    vsr.set_val(0, 5, i, 1)
    vsr.set_val(1, 4, i, 1)
    vsr.set_val(1, 5, i, 1)

    vsr.set_val(6, 4, i, 1)
    vsr.set_val(6, 5, i, 1)
    vsr.set_val(7, 4, i, 1)
    vsr.set_val(7, 5, i, 1)