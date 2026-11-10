"""Worker functions for running the reservoir trajectories on several GPUs.

They live in a module (not in the notebook) so that worker processes started with the
"spawn" method can import them. Spawn gives every worker a clean CUDA state, which a
Jupyter kernel that has already used CUDA cannot guarantee with "fork".
"""
from collections import Counter

import numpy as np
import torch
from emu_sv import SVBackend

_state = {}  # per-process copy of the simulation inputs, set by init_worker
_built = {}  # per-process cache: window index -> sequence built for that window


def init_worker(gpu_ids, sequence, config, windows, n_trajectories, seed, cpu_threads=4):
    """Pin this worker process to one GPU and keep the simulation inputs.

    Several workers can share a GPU (the queue holds one entry per worker), so each one
    gets only a few CPU threads to avoid oversubscribing the cores.
    """
    torch.cuda.set_device(gpu_ids.get())
    torch.set_num_threads(cpu_threads)
    _state.update(sequence=sequence, config=config, windows=windows,
                  n_trajectories=n_trajectories, seed=seed)


def run_trajectory(job):
    """Simulate one noise realisation of one window; returns (window index, bitstring counts)."""
    w, _ = divmod(job, _state["n_trajectories"])
    seed = _state["seed"] + job  # distinct, reproducible noise for every job
    np.random.seed(seed)         # Pulser draws its noise and shots from NumPy's global generator
    torch.manual_seed(seed)
    if w not in _built:  # all trajectories of a window share the same built sequence
        _built[w] = _state["sequence"].build(x=_state["windows"][w].tolist())  # fill "x" with this window
    built = _built[w]
    return w, Counter(SVBackend(built, config=_state["config"]).run().final_bitstrings)
