import mujoco
import numpy as np
import time

num_runs = 10
total_time = 0

model = mujoco.MjModel.from_xml_path("mj_models/pendulum.xml")
data = mujoco.MjData(model)

def single_run():
    mujoco.mj_resetData(model, data)
    data.qpos[0] = np.pi / 4
    data.qvel[0] = 0.0

    trajectory = []

    while data.time < 1000.0:
        mujoco.mj_step(model, data)
        trajectory.append([data.qpos[0], data.qvel[0], data.qacc[0]])
    

for _ in range(num_runs):
    start = time.time()
    single_run()
    end = time.time()
    
    total_time += (end - start)
    
    print(f"Total time: {end - start}")

print(f"Average time over {num_runs} runs: {total_time / num_runs:6f}")