import ray
import time
import copy
import torch
from client import Client

# Initialize Ray
ray.init(ignore_reinit_error=True)

NUM_CLIENTS = 3
ROUNDS = 5

# Create clients
clients = [Client.remote(i) for i in range(NUM_CLIENTS)]

# Initialize global model from first client
state = ray.get(clients[0].train.remote(local_epochs=0))

# Training loop
for r in range(ROUNDS):
    # Start local training concurrently
    futures = [c.train.remote(local_epochs=1) for c in clients]
    state_dicts = ray.get(futures)

    # Simple FedAvg
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        agg[k] = torch.zeros_like(agg[k])

    for sd in state_dicts:
        for k in agg:
            agg[k] += sd[k]

    for k in agg:
        agg[k] /= len(state_dicts)

    # Broadcast aggregated model to all clients
    [c.set_global.remote(agg) for c in clients]

    print(f"Round {r+1} done")

# Shutdown Ray
ray.shutdown()
