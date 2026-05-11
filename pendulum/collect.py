import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mj_models/pendulum.xml")
data = mujoco.MjData(model)

data.qpos[0] = np.pi / 4
data.qvel[0] = 0.0

trajectory = []
while data.time < 10.0:
    mujoco.mj_step(model, data)
    trajectory.append([data.qpos[0], data.qvel[0], data.qacc[0]])

arr = np.array(trajectory)
np.save("pendulum_trajectory.npy", arr)