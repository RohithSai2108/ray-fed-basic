# datasets.py
import os
import gzip
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# ---------------------------
# Helpers
# ---------------------------

def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)

def _mnist_raw_available(root: str = "./data"):
    """Return True if the 4 MNIST .gz raw files are present under root/MNIST/raw."""
    raw = os.path.join(root, "MNIST", "raw")
    required = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    if not os.path.isdir(raw):
        return False
    for f in required:
        if not os.path.isfile(os.path.join(raw, f)):
            return False
    return True

# ---------------------------
# Transforms
# ---------------------------

MNIST_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

MNIST_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

CIFAR_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

SVHN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ---------------------------
# MNIST raw reader (directly from .gz IDX files)
# ---------------------------

def _read_idx_images(gz_path):
    """Read IDX image file (u8) from .gz and return numpy array shape (N, H, W)."""
    with gzip.open(gz_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, rows, cols)
    return data

def _read_idx_labels(gz_path):
    """Read IDX label file (u8) from .gz and return numpy array shape (N,)."""
    with gzip.open(gz_path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def load_mnist_from_raw(root: str = "./data"):
    """
    Load MNIST from ./data/MNIST/raw/*.gz into PyTorch TensorDataset
    Returns (train_dataset, test_dataset) as TensorDataset with float tensors in [0,1].
    Raises FileNotFoundError if files are missing.
    """
    raw_dir = os.path.join(root, "MNIST", "raw")
    train_images_gz = os.path.join(raw_dir, "train-images-idx3-ubyte.gz")
    train_labels_gz = os.path.join(raw_dir, "train-labels-idx1-ubyte.gz")
    test_images_gz  = os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_gz  = os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz")

    for p in (train_images_gz, train_labels_gz, test_images_gz, test_labels_gz):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MNIST raw file missing: {p}")

    X_train = _read_idx_images(train_images_gz).astype(np.float32) / 255.0
    y_train = _read_idx_labels(train_labels_gz).astype(np.int64)
    X_test  = _read_idx_images(test_images_gz).astype(np.float32) / 255.0
    y_test  = _read_idx_labels(test_labels_gz).astype(np.int64)

    X_train_t = torch.from_numpy(X_train).unsqueeze(1)  # (N,1,H,W)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test).unsqueeze(1)
    y_test_t  = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)
    return train_ds, test_ds

# ---------------------------
# Public dataset functions
# ---------------------------

def get_mnist(batch_size: int = 64, root: str = "./data"):
    """
    Returns (train_loader, test_loader) for MNIST.
    Prefers local raw files (load_mnist_from_raw). If missing, falls back to EMNIST('mnist').
    """
    if _mnist_raw_available(root):
        train_ds, test_ds = load_mnist_from_raw(root=root)
        train_loader = _make_loader(train_ds, batch_size, shuffle=True)
        test_loader  = _make_loader(test_ds, batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        # fallback to EMNIST split='mnist' which is shape/label compatible
        train_ds = datasets.EMNIST(root=root, split="mnist", train=True, download=True, transform=MNIST_TRAIN_TRANSFORM)
        test_ds  = datasets.EMNIST(root=root, split="mnist", train=False, download=True, transform=MNIST_TEST_TRANSFORM)
        return _make_loader(train_ds, batch_size, shuffle=True), _make_loader(test_ds, batch_size, shuffle=False)

def get_cifar10(batch_size: int = 64, root: str = "./data"):
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=CIFAR_TRAIN_TRANSFORM)
    test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=CIFAR_TEST_TRANSFORM)
    return _make_loader(train_ds, batch_size, shuffle=True), _make_loader(test_ds, batch_size, shuffle=False)

def get_svhn(batch_size: int = 64, root: str = "./data"):
    train_ds = datasets.SVHN(root=root, split="train", download=True, transform=SVHN_TRANSFORM)
    test_ds  = datasets.SVHN(root=root, split="test",  download=True, transform=SVHN_TRANSFORM)
    return _make_loader(train_ds, batch_size, shuffle=True), _make_loader(test_ds, batch_size, shuffle=False)

# ---------------------------
# Partitioned MNIST for federated experiments
# ---------------------------

def get_partitioned_mnist(cid: int, num_clients: int, batch_size: int = 64,
                          root: str = "./data", noniid: bool = False, alpha: float = 0.5):
    """
    Return (train_loader, test_loader) for client `cid` out of `num_clients`.
    Uses raw MNIST if available (no torchvision download). Otherwise falls back to EMNIST('mnist').
    Produces IID equal-sized shards by default; set noniid=True for a simple Dirichlet split.
    """
    if _mnist_raw_available(root):
        full_train, test_ds = load_mnist_from_raw(root=root)
    else:
        full_train = datasets.EMNIST(root=root, split="mnist", train=True, download=True, transform=MNIST_TRAIN_TRANSFORM)
        test_ds    = datasets.EMNIST(root=root, split="mnist", train=False, download=True, transform=MNIST_TEST_TRANSFORM)

    n = len(full_train)

    if not noniid:
        sizes = [n // num_clients] * num_clients
        for i in range(n % num_clients):
            sizes[i] += 1
        idx = np.random.permutation(n)
        parts = []
        start = 0
        for s in sizes:
            parts.append(idx[start:start+s].tolist())
            start += s
    else:
        props = np.random.dirichlet([alpha] * num_clients)
        counts = (props * n).astype(int)
        while counts.sum() < n:
            counts[np.argmax(props - counts / n)] += 1
        idx = np.random.permutation(n)
        parts = []
        start = 0
        for c in counts:
            parts.append(idx[start:start+c].tolist())
            start += c

    my_idx = parts[cid]
    sub_train = Subset(full_train, my_idx)
    train_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader

# ---------------------------
# Small validation loader (from test)
# ---------------------------

def small_val_loader_from_test(test_loader, max_samples: int = 2000):
    ds = test_loader.dataset
    n = min(len(ds), max_samples)
    idx = list(range(n))
    sub = Subset(ds, idx)
    return _make_loader(sub, test_loader.batch_size, shuffle=False, num_workers=0)
