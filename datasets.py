# datasets.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---- Common helpers --------------------------------------------------------

def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 0):
    # num_workers=0 is safest on Windows/VS Code
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)

# ---- MNIST (using Fashion-MNIST as drop-in replacement to avoid broken URLs) ----

def get_mnist(batch_size: int = 64):
    """
    Returns (train_loader, test_loader) for Fashion-MNIST (28x28, 1 channel, 10 classes),
    used as a drop-in replacement for MNIST due to legacy URL issues.
    """
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tf_train)
    test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tf_test)

    return (
        _make_loader(train_ds, batch_size, shuffle=True),
        _make_loader(test_ds,  batch_size, shuffle=False),
    )

# ---- CIFAR-10 (32x32 RGB, 10 classes) -------------------------------------

def get_cifar10(batch_size: int = 64):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        ),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tf_test)

    return (
        _make_loader(train_ds, batch_size, shuffle=True),
        _make_loader(test_ds,  batch_size, shuffle=False),
    )

# ---- SVHN (32x32 RGB, 10 classes; labels are 1..10 where '10' means digit 0) ----

def get_svhn(batch_size: int = 64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        ),
    ])
    train_ds = datasets.SVHN(root="./data", split="train", download=True, transform=tf)
    test_ds  = datasets.SVHN(root="./data", split="test",  download=True, transform=tf)

    return (
        _make_loader(train_ds, batch_size, shuffle=True),
        _make_loader(test_ds,  batch_size, shuffle=False),
    )

# ---- (Optional) tiny validation sampler for expensive contribution methods ----

def small_val_loader_from_test(test_loader, max_samples: int = 2000):
    """
    Build a small validation DataLoader (e.g., for least-core) from the test set iterator.
    Works only if test_loader.dataset supports __getitem__/__len__.
    """
    ds = test_loader.dataset
    n = min(len(ds), max_samples)
    idx = list(range(n))
    sub = Subset(ds, idx)
    return _make_loader(sub, test_loader.batch_size, shuffle=False, num_workers=0)
