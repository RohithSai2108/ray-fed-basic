import ray
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import get_mnist          # uses FashionMNIST under the hood (same shape)
from model import SimpleCNN
from utils import (
    compute_sigma_from_score,
    add_gaussian_noise_to_state,
    flatten_state_dict,
)


@ray.remote
class Client:
    """
    Ray actor representing one FL client.
    - Trains locally
    - (Optionally) adds personalized DP noise based on `score`
    - Returns (state_dict, delta_vector, metrics)
    """

    def __init__(self, cid: int, dataset: str = "mnist", in_channels: int = 1, malicious_id: int = -1):
        self.cid = cid
        self.malicious_id = malicious_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Dataset + Model ----
        ds = (dataset or "mnist").lower()
        if ds == "mnist":
            self.train_loader, self.test_loader = get_mnist(batch_size=64)
            self.model = SimpleCNN(in_channels=in_channels, num_classes=10).to(self.device)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Keep previous global weights to compute deltas each round
        self.prev_global = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ---------------- Helpers ----------------

    def set_global(self, state_dict):
        """Receive and load global model from server."""
        self.model.load_state_dict(state_dict)
        self.prev_global = {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def get_state(self):
        """Return current clean weights (used for bootstrap if desired)."""
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    # ---------------- Train ------------------

    def train(self, local_epochs: int = 1, score: float = 1.0, lr: float = 0.01, dp: bool = True):
        """
        Train locally for `local_epochs`.
        - If local_epochs == 0: return current weights (bootstrap), no DP.
        - score in [0,1] controls the DP noise scale (if dp=True).
        Returns: (state_dict, delta_vec, {'loss': avg_loss or None})
        """
        loss_fn = nn.CrossEntropyLoss()
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # ---- Bootstrap: no training, no DP ----
        if local_epochs == 0:
            current = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            delta_vec = flatten_state_dict({k: current[k] - self.prev_global[k] for k in current})
            return current, delta_vec, {"loss": None}

        # ---- Normal training ----
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for e in range(local_epochs):
            running = 0.0
            batches_this_epoch = 0

            for b, (x, y) in enumerate(self.train_loader):
                # optional malicious behavior for robustness experiments
                if self.cid == self.malicious_id:
                    y = (y + 1) % 10

                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

                running += float(loss.item())
                total_loss += float(loss.item())
                batches_this_epoch += 1
                total_batches += 1

            avg_epoch_loss = running / max(1, batches_this_epoch)
            print(f"[Client {self.cid}] Epoch {e+1} avg loss: {avg_epoch_loss:.4f}")

        # ---- Prepare return payload ----
        current = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        delta_vec = flatten_state_dict({k: current[k] - self.prev_global[k] for k in current})

        # DP noise (only if enabled)
        if dp:
            sigma = compute_sigma_from_score(score)
            noisy_state = add_gaussian_noise_to_state(current, sigma, device="cpu")
        else:
            noisy_state = current

        avg_loss_overall = total_loss / max(1, total_batches)
        return noisy_state, delta_vec, {"loss": avg_loss_overall}

    # ---------------- Eval -------------------

    def eval_global(self) -> float:
        """Evaluate current model on local test set. Returns accuracy in [0,1]."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / max(1, total)
