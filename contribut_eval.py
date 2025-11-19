# contrib_eval.py
import random
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# -------- Fast proxy: gradient/weight-change similarity (default) --------
def compute_contribs_by_similarity(deltas):
    """
    deltas: list[1D torch.Tensor on CPU]
    Returns a list of non-negative weights that sum to 1.
    """
    mats = [d.numpy().reshape(1, -1) for d in deltas]
    agg = (sum(deltas)).numpy().reshape(1, -1)
    sims = [max(cosine_similarity(m, agg)[0, 0], 0.0) for m in mats]
    arr = np.array(sims, dtype=float)
    if arr.sum() <= 0:
        return [1.0 / len(deltas)] * len(deltas)
    return (arr / arr.sum()).tolist()

# -------- Paper-faithful (slower): Monte-Carlo least-core --------
def _aggregate_subset(state_dicts, idxs):
    import copy
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        agg[k] = torch.zeros_like(agg[k])
    for i in idxs:
        for k in agg:
            agg[k] += state_dicts[i][k]
    for k in agg:
        agg[k] /= len(idxs)
    return agg

def _eval_state(model_class, state_dict, val_loader, device='cpu'):
    m = model_class().to(device)
    m.load_state_dict(state_dict)
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            p = m(x).argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)

def monte_carlo_least_core(state_dicts, model_class, val_loader, M=100):
    """
    Approximate least-core via M random coalition samples.
    Returns a list of non-negative weights that sum to 1.
    """
    try:
        import cvxpy as cp
    except Exception:
        # Fallback if cvxpy not installed
        # Use similarity proxy to avoid breaking the pipeline
        deltas = []
        # Build deltas from state_dicts by flattening relative to their mean
        with torch.no_grad():
            mean = {}
            for k in state_dicts[0]:  # average state
                mean[k] = sum(sd[k] for sd in state_dicts) / len(state_dicts)
            for sd in state_dicts:
                flat = torch.cat([(sd[k] - mean[k]).view(-1).cpu() for k in sd])
                deltas.append(flat)
        return compute_contribs_by_similarity(deltas)

    n = len(state_dicts)
    vN = _eval_state(model_class, _aggregate_subset(state_dicts, list(range(n))), val_loader)

    samples = []
    for _ in range(M):
        s = random.randint(1, n - 1)
        idxs = random.sample(range(n), s)
        vS = _eval_state(model_class, _aggregate_subset(state_dicts, idxs), val_loader)
        samples.append((idxs, vS))

    x = cp.Variable(n)
    eps = cp.Variable()
    cons = [cp.sum(x[idxs]) >= vS - eps for idxs, vS in samples]
    cons += [cp.sum(x) == vN, x >= 0, eps >= 0]
    prob = cp.Problem(cp.Minimize(eps), cons)
    prob.solve(solver=cp.SCS, verbose=False)
    xs = np.maximum(np.array(x.value).flatten(), 0)
    s = xs.sum()
    return (xs / s).tolist() if s > 0 else [1.0 / n] * n
