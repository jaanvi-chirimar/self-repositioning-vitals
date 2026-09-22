import pybullet as p
import pybullet_data
import time

# 1. Connect to the physics server (headless mode)
p.connect(p.GUI)

# 2. Tell PyBullet where its built-in models live
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set gravity (x, y, z) — gravity points down in z
p.setGravity(0,0,-9.8)

# 4. Load the ground plane and a robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])

import time
# inside the loop, after stepSimulation:

# 5. Step the simulation and print the robot's position
# for i in range(100):
#     p.stepSimulation()
#     time.sleep(1/60)
#     position = p.getBasePositionAndOrientation(robot_id)[0]
#     print(i, position)

while True:
    p.stepSimulation()
    time.sleep(1/60)

# 6. Clean up
p.disconnect()
print("done")