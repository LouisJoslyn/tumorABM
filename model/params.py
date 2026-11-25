"""Parameter loading. Numba kernels receive a flat float64 array P; indices are module constants."""
import os
import numpy as np
import yaml

NAMES = [
    "dt", "T_div", "T_death", "O_q", "O_n", "mu", "mut_sd", "vessel_regress", "T_clear",
    "k_o_tissue", "k_o_tumor", "k_c", "chemo_secrete", "chemo_apop", "k_s", "supp_secrete",
    "T_life", "T_life_ex", "move_prob", "tumor_drag", "s_motility", "chi", "k_kill",
    "p_block", "s_block", "e_kill", "e_pdl1", "p_prolif", "max_div", "t_prime", "r_recruit",
    "T_necro_clear",
]
for _i, _n in enumerate(NAMES):
    globals()["P_" + _n.upper()] = _i


def load_config(path):
    """Load YAML; a `base:` key names a parent config (relative path) to inherit from."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(os.path.join(os.path.dirname(path), base))
        for sec, vals in cfg.items():
            if isinstance(vals, dict) and isinstance(parent.get(sec), dict):
                parent[sec].update(vals)
            else:
                parent[sec] = vals
        cfg = parent
    return cfg


def param_array(cfg):
    p = cfg["params"]
    missing = [n for n in NAMES if n not in p]
    if missing:
        raise KeyError(f"missing params: {missing}")
    return np.array([float(p[n]) for n in NAMES], dtype=np.float64)
