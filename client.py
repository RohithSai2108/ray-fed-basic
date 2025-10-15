# client.py
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_partitioned_mnist
from model import SimpleCNN

@ray.remote
class Client:
    def __init__(self, cid, num_clients=3, in_channels=1, batch_size=32):
        self.cid = cid
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SimpleCNN(in_channels=in_channels).to(self.device)

        # Get non-IID MNIST for this client
        self.train_loader, self.test_loader = get_partitioned_mnist(cid, num_clients, batch_size)

    def set_global(self, state_dict):
        """Receive and load global model weights."""
        self.model.load_state_dict(state_dict)

    def train(self, local_epochs=1, lr=0.01):
        """Local training on client's data."""
        opt = optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for e in range(local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

        # Return updated model parameters to the server
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
