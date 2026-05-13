import mujoco
import numpy as np
import tracemalloc

num_runs = 10
total_memory = 0

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
        
tracemalloc.start()
for _ in range(num_runs):
    tracemalloc.clear_traces()
    single_run()    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory Usage Peak: {peak} | Memory Usage Current: {current}")
    total_memory += peak
tracemalloc.stop()

print(f"Average peak memory over {num_runs} runs: {total_memory / num_runs / 1024:.2f} KB")