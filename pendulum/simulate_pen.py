import mujoco
import numpy as np
import pandas as pd

def collect_trajectory(model, theta0, omega0, t_end=5.0):
    data = mujoco.MjData(model)
    data.qpos[0] = theta0
    data.qvel[0] = omega0

    rows = []
    while data.time < t_end:
        theta = data.qpos[0]
        omega = data.qvel[0]
        alpha = data.qacc[0]

        mujoco.mj_step(model, data)

        rows.append({
            "theta":      theta,
            "omega":      omega,
            "alpha":      alpha,
            "theta_next": data.qpos[0],
            "omega_next": data.qvel[0],
        })

    return pd.DataFrame(rows)

model = mujoco.MjModel.from_xml_path("mj_models/pendulum.xml")

# Sample initial conditions
np.random.seed(42)
n_trajectories = 20
theta0s = np.random.uniform(-np.pi, np.pi, n_trajectories)
omega0s = np.random.uniform(-2.0, 2.0, n_trajectories)

# Split into train/test initial conditions
split = int(0.8 * n_trajectories)
dfs = []
for i, (theta0, omega0) in enumerate(zip(theta0s, omega0s)):
    df = collect_trajectory(model, theta0, omega0)
    df["trajectory_id"] = i
    df["split"] = "train" if i < split else "test"
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.to_csv("pendulum_trajectories.csv", index=False)

print(f"Total samples: {len(df)}")
print(f"Train: {len(df[df.split == 'train'])}, Test: {len(df[df.split == 'test'])}")
print(df.head())