# run_experiment.py — Fed-RDP complete (Steps 9.2, 9.3, 9.4)

import argparse
import copy
import json
import os
import time

import ray
import torch
import matplotlib.pyplot as plt

from client import Client
from contribut_eval import compute_contribs_by_similarity, monte_carlo_least_core
from reputation import update_score
from ledger import log as ledger_log


# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10', 'svhn'])
parser.add_argument('--clients', type=int, default=3)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--local-epochs', type=int, default=1)
parser.add_argument('--malicious', type=int, default=-1)
parser.add_argument('--dp', action='store_true', help='Enable personalized DP noise on clients')
parser.add_argument('--contrib', default='similarity', choices=['similarity', 'leastcore'],
                    help='Contribution estimator')
parser.add_argument('--outdir', default='outputs', help='Directory to save plots & summary')
args = parser.parse_args()


# ---------------------------
# Ray init
# ---------------------------
ray.init(ignore_reinit_error=True)


# ---------------------------
# Helper: reputation-weighted aggregation
# ---------------------------
def weighted_aggregate(state_dicts, scores):
    """
    Combine client model states using weights proportional to reputation scores.
    When scores are equal, this reduces to FedAvg.
    """
    agg = copy.deepcopy(state_dicts[0])
    for k in agg:
        agg[k] = torch.zeros_like(agg[k])

    total = float(sum(scores)) + 1e-12
    for i, sd in enumerate(state_dicts):
        w = scores[i] / total
        for k in agg:
            agg[k] += sd[k].float() * w
    return agg


# ---------------------------
# Spawn clients
# ---------------------------
clients = [Client.remote(i, dataset=args.dataset, malicious_id=args.malicious)
           for i in range(args.clients)]

# Bootstrap global weights from client 0 (no training)
_ = ray.get(clients[0].train.remote(local_epochs=0))

# ---------------------------
# Reputation init
# ---------------------------
scores = [1.0 for _ in range(args.clients)]

# ---------------------------
# Tracking (for plots & summary)
# ---------------------------
acc_hist = []
scores_hist = []
contribs_hist = []


# ---------------------------
# Federated rounds
# ---------------------------
for r in range(args.rounds):
    print(f"\n[Server] Round {r+1} starting ...")
    t0 = time.time()

    # Local training in parallel (pass current score & DP flag)
    futures = [
        clients[i].train.remote(
            local_epochs=args.local_epochs,
            score=scores[i],
            dp=args.dp
        ) for i in range(args.clients)
    ]
    results = ray.get(futures)  # list of (noisy_state, delta_vec, metrics)

    # Split results
    state_dicts = [res[0] for res in results]
    deltas      = [res[1] for res in results]
    metrics     = [res[2] for res in results]

    # ---- Log initial entries to ledger (pre-contrib; contrib=0 placeholder) ----
    for i, (sd, dv, met) in enumerate(results):
        ledger_log(round_no=r, client_id=i, sd=sd, score=scores[i], contrib=0.0, loss=met['loss'])

    # Aggregation (reputation-weighted)
    global_state = weighted_aggregate(state_dicts, scores)

    # Broadcast global model to all clients
    _ = [c.set_global.remote(global_state) for c in clients]

    # ---- Contribution computation ----
    if args.contrib == 'leastcore':
        try:
            # Least-core needs model_class + val_loader; contrib_eval has a fallback if not provided.
            contribs = monte_carlo_least_core(state_dicts, model_class=None, val_loader=None,
                                              M=args.clients * args.clients)
        except Exception as e:
            print(f"[Server] Least-core not available ({e}); falling back to similarity.")
            contribs = compute_contribs_by_similarity(deltas)
    else:
        contribs = compute_contribs_by_similarity(deltas)

    # ---- Reputation update ----
    scores = [update_score(scores[i], contribs[i], args.clients) for i in range(args.clients)]

    # ---- Ledger (with actual contribs) ----
    for i, sd in enumerate(state_dicts):
        ledger_log(round_no=r, client_id=i, sd=sd, score=scores[i], contrib=contribs[i], loss=metrics[i]['loss'])

    # ---- Accuracy (quick proxy: mean across clients) ----
    accs = ray.get([c.eval_global.remote() for c in clients])
    avg_acc = sum(accs) / len(accs)

    # ---- Track histories for plots ----
    acc_hist.append(avg_acc)
    scores_hist.append(scores[:])
    contribs_hist.append(contribs[:])

    # ---- Round logs ----
    norm_w = [round(s / (sum(scores) + 1e-12), 3) for s in scores]
    print(f"[Server] Round {r+1} contribs: { [round(c,3) for c in contribs] }")
    print(f"[Server] Round {r+1} scores:   { [round(s,3) for s in scores] }")
    print(f"[Server] Round {r+1} weights:  { norm_w }")
    print(f"[Server] Round {r+1} avg client acc: {avg_acc:.4f}")
    print(f"[Server] Round {r+1} done in {time.time() - t0:.2f}s")


# ---------------------------
# Save plots & summary
# ---------------------------
os.makedirs(args.outdir, exist_ok=True)

# Accuracy vs Round
plt.figure()
plt.plot(acc_hist)
plt.xlabel("Round"); plt.ylabel("Avg client accuracy")
plt.title(f"Accuracy vs Round ({args.dataset}, clients={args.clients}, dp={args.dp}, contrib={args.contrib})")
plt.savefig(os.path.join(args.outdir, "acc_vs_round.png"))
plt.close()

# Reputation scores per client
for i in range(args.clients):
    plt.figure()
    plt.plot([s[i] for s in scores_hist])
    plt.xlabel("Round"); plt.ylabel(f"Score (client {i})")
    plt.title(f"Reputation vs Round — client {i}")
    plt.savefig(os.path.join(args.outdir, f"score_client_{i}.png"))
    plt.close()

# Contributions per client
for i in range(args.clients):
    plt.figure()
    plt.plot([c[i] for c in contribs_hist])
    plt.xlabel("Round"); plt.ylabel(f"Contribution (client {i})")
    plt.title(f"Contribution vs Round — client {i}")
    plt.savefig(os.path.join(args.outdir, f"contrib_client_{i}.png"))
    plt.close()

# Summary JSON
with open(os.path.join(args.outdir, "summary.json"), "w") as f:
    json.dump(
        {
            "args": vars(args),
            "acc_hist": acc_hist,
            "scores_hist": scores_hist,
            "contribs_hist": contribs_hist
        },
        f, indent=2
    )

print(f"\n[Server] Saved plots & summary to: {args.outdir}/ and ledger.json")


# ---------------------------
# Shutdown
# ---------------------------
ray.shutdown()
print("\n[Server] Training complete! Ray shutdown successful.")
