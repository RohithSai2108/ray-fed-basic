# datasets.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_partitioned_mnist(client_id, num_clients=3, batch_size=32):
    """
    Returns a DataLoader for a specific client with a non-IID MNIST partition.
    Each client gets mostly certain digits.
    """
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Sort by label to simulate non-IID distribution
    sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset[i][1])

    total_per_client = len(dataset) // num_clients
    start = client_id * total_per_client
    end = start + total_per_client
    indices = sorted_indices[start:end]

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader, None  # return None for test loader for simplicity
