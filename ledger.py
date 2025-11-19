# ledger.py
from tinydb import TinyDB
import pickle, hashlib, time, os

_db_path = "ledger.json"
# ensure DB directory exists
_db = TinyDB(_db_path)

def _hash_state(sd):
    """
    Create a short deterministic hash from a model state-dict-like mapping.
    sd: dict mapping param_name -> torch.Tensor OR numpy array
    We only sample the first few elements to bound size.
    """
    try:
        small = {}
        for k, v in sd.items():
            # try to convert to numpy safely
            try:
                arr = v.cpu().numpy().ravel()[:16]  # first 16 elements
            except Exception:
                # fallback: pickle the object (safe small)
                arr = pickle.dumps(v)[:64]
            small[k] = arr
        return hashlib.sha256(pickle.dumps(small)).hexdigest()
    except Exception as e:
        # last-resort hash
        return hashlib.sha256(str(type(sd)).encode() + str(time.time()).encode()).hexdigest()

def log(round_no, client_id, sd, score, contrib, loss):
    """
    Save a ledger entry.
    round_no: int
    client_id: int
    sd: state dict (will be hashed)
    score: float
    contrib: float
    loss: float or None
    """
    try:
        entry = {
            "round": int(round_no),
            "client": int(client_id),
            "hash": _hash_state(sd) if sd is not None else None,
            "score": float(score) if score is not None else None,
            "contrib": float(contrib) if contrib is not None else None,
            "loss": None if loss is None else float(loss),
            "ts": time.time(),
        }
        _db.insert(entry)
    except Exception as e:
        # do not raise to avoid killing training; print for debugging
        print(f"[ledger.log] failed to write entry: {e}")
