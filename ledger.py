# ledger.py
from tinydb import TinyDB
import pickle, hashlib, time

_db = TinyDB("ledger.json")

def _hash_state(sd):
    # lightweight state hash (privacy-preserving: state only hashed)
    return hashlib.sha256(pickle.dumps({k: v.cpu().numpy()[:10] for k, v in sd.items()})).hexdigest()

def log(round_no:int, client_id:int, state_dict, score:float, contrib:float, loss):
    _db.insert({
        "round": round_no,
        "client": client_id,
        "hash": _hash_state(state_dict),
        "score": float(score),
        "contrib": float(contrib),
        "loss": None if loss is None else float(loss),
        "ts": time.time()
    })
