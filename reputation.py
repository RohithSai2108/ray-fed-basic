# reputation.py
import math

def update_score(s, contrib_ratio, n, k=10.0, s_min=0.0, s_max=1.0):
    """
    Logistic-shaped update around uniform contribution 1/n.
    s in [0,1] -> s_new in [0,1].
    """
    alpha = k * (contrib_ratio - 1.0 / n)
    inc = (s_max - s) / (1.0 + math.exp(-alpha))
    dec = (s - s_min) / (1.0 + math.exp(alpha))
    s_new = s + inc - dec
    if s_new < s_min: s_new = s_min
    if s_new > s_max: s_new = s_max
    return s_new
