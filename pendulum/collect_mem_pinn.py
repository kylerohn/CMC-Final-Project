import torch
import tracemalloc
import numpy as np
from torch import nn

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

model = PINN(2, 16, 2)

weights = torch.load("pinn_models/pinn-05-11-12:26:22-16hdim.pth")
model.load_state_dict(weights)
model.eval()
model = torch.compile(model)

n_steps = 100000
num_runs = 10
total_memory = 0

def single_run():
    theta0 = np.pi/4  # radians
    omega0 = 0.0  # rad/s
    rows = []
    state = torch.tensor([[theta0, omega0]], dtype=torch.float32)
    
    with torch.no_grad():
        for _ in range(n_steps):
            next_state = model(state)

            theta_next = (next_state[0, 0].item() + np.pi) % (2 * np.pi) - np.pi
            omega_next = next_state[0, 1].item()

            rows.append([theta_next, omega_next])
            state[0, 0] = theta_next
            state[0, 1] = omega_next

single_run() # warmup for lazy loading
tracemalloc.start()
for _ in range(num_runs):
    tracemalloc.clear_traces()
    single_run()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory Usage Peak: {peak} | Memory Usage Current: {current}")
    total_memory += peak
tracemalloc.stop()

print(f"Average peak memory over {num_runs} runs: {total_memory / num_runs / 1024:.2f} KB")
