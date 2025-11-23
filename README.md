
# ray-fed-basic-implementation

Lightweight reference implementation for reputation-weighted federated learning using Ray + PyTorch.

## Quick overview

- The main entrypoint is `run_experiment.py`.
- Clients and training logic live in `client.py`. Dataset utilities are in `datasets.py`.
- A small smoke test for Ray is provided in `test_ray.py`.
- The repository will save plots and a `summary.json` into the `outputs/` (or the `--outdir`) directory and append model/round entries to `ledger.json`.

## Requirements

- Python 3.11 (Dockerfile uses 3.11-slim)
- `pip` and virtualenv (for local runs)
- Docker (optional, recommended for reproducible runs)
- At least ~2GB RAM for CPU experiments (more is better for faster training)

See `requirements.txt` for the Python package versions used in the environment.

## Install & run locally (PowerShell)

1. Create and activate a virtual environment (PowerShell on Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the federated experiment (example):

```powershell
python run_experiment.py --dataset mnist --clients 3 --rounds 10 --local-epochs 1
```

Useful flags:
- `--dp` : enable personalized DP noise on clients
- `--contrib <similarity|leastcore>` : contribution estimator (default: similarity)
- `--outdir <dir>` : directory to save plots & summary (default `outputs`)

Example with DP and least-core contribution estimator:

```powershell
python run_experiment.py --dataset mnist --clients 5 --rounds 20 --local-epochs 2 --dp --contrib leastcore --outdir outputs_rdp_mnist
```

If you don't want to use Ray actors and want a quick smoke-run, you can open `test_ray.py`:

```powershell
python test_ray.py
```

## Data

- By default the code prefers raw MNIST files placed under `./data/MNIST/raw` as the 4 IDX `.gz` files. If those files are present the repository will load them directly (no torchvision download required).
- If the raw files are missing, `datasets` falls back to `torchvision.datasets.EMNIST(split='mnist')` and will download the data into `./data` automatically.

To run with local raw MNIST files, place the following files under `data/MNIST/raw`:

- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

When those are present the code uses them; otherwise it will download EMNIST automatically at runtime.

## Docker

The `Dockerfile` sets up a minimal Python 3.11 image and installs the CPU PyTorch wheel, then installs `requirements.txt` and copies the repo. The default Docker command runs `run_experiment.py` with a small example configuration.

Build the image (from the repo root):

```powershell
docker build -t ray-fed-basic:local .
```

Run the container (example):

```powershell
docker run --rm --shm-size=5.07gb ray-fed-basic:local
```

Notes:
- `--shm-size=5.07gb` is used in the Docker example to provide additional shared memory for matplotlib/parallelism. Adjust as needed for your environment.
- The Docker `CMD` (see `Dockerfile`) uses `python run_experiment.py --dataset mnist --clients 3 --rounds 10 --local-epochs 1` by default.

## Outputs

- Plots (accuracy, reputation, contribution) and a `summary.json` are saved to the directory given by `--outdir` (default `outputs`).
- The run also writes entries to `ledger.json` (in repo root) to track per-round client contributions/scores.

Example: after running the default example you should see files under `outputs/` such as `acc_vs_round.png` and `summary.json`.

## Project structure

- `run_experiment.py` : main experiment runner (CLI args documented in the file)
- `client.py` : Ray actor implementing local training and optional DP noise
- `datasets.py` : dataset helpers (raw MNIST reader + EMNIST fallback + partitioning)
- `model.py` : neural network architecture used by clients
- `ledger.py` : simple ledger logging helper (writes `ledger.json`)
- `contribut_eval.py` : contribution estimation utilities (similarity / least-core)
- `reputation.py` : reputation update logic
- `test_ray.py` : tiny Ray smoke-test

## Troubleshooting

- If you see dataset download errors, ensure your machine has internet access or place the MNIST raw files in `data/MNIST/raw`.
- If running out of memory, reduce `--clients` or `--local-epochs` or run on a machine with more RAM.

## Want me to tailor this README to your exact terminal history?

I updated the README to match typical commands used with this repository (PowerShell examples included). If you want me to reflect specific terminal commands you ran earlier, paste those commands here and I'll incorporate them verbatim into a "Recent commands" or "Examples" section.
