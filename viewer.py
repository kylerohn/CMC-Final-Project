# visualize.py — load saved trajectory, replay in viewer
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "x11"

args = sys.argv

if len(args) < 3:
    print("Usage: viewer.py <mujoco_model_path> <trajectory_path>")
    exit(1)

model = mujoco.MjModel.from_xml_path(args[1])
data = mujoco.MjData(model)

arr = np.load(args[2])

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    for row in arr:
        data.qpos[0] = row[0]
        data.qvel[0] = row[1]
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    viewer.close()