import torch

def compute_sigma_from_score(score, eps_min=0.05, eps_max=0.6, base=1.0):
    eps = eps_min + (eps_max - eps_min) * float(score)
    return base / max(eps, 1e-6)

def add_gaussian_noise_to_state(state_dict, sigma, device='cpu'):
    noisy = {}
    for k, v in state_dict.items():
        noise = torch.normal(0, sigma, size=v.shape, device=device)
        noisy[k] = (v.to(device) + noise).detach().cpu()
    return noisy

def flatten_state_dict(state_dict):
    parts = [p.detach().view(-1).cpu() for p in state_dict.values()]
    return torch.cat(parts)
