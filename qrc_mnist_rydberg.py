"""Quantum reservoir computing (QRC) for MNIST denoising with Rydberg atoms.

Reproduces the setup of "Image Denoising via Quantum Reservoir Computing"
(Das, Antoncich and Wang, arXiv:2512.18612) with one change: the readout is
LINEAR (ridge regression) instead of the paper's neural network. With a linear
readout the reservoir is the only source of nonlinearity, so any improvement
over the PCA baseline comes from the reservoir and not from the readout.

Steps (one section of this file each, run in this order by main()):

    1. Data       The first POOL_TRAIN / POOL_TEST MNIST images, corrupted with
                  speckle noise.
    2. Encoding   PCA of the noisy images: one coordinate x_i in [0, 1] per atom.
    3. Reservoir  Atoms on a rows x cols array (a chain if rows = 1) under a
                  constant drive, with x_i setting the local detuning of atom i
                  (emu-sv on GPU). <Z_i> and <Z_i Z_j> are estimated from SHOTS
                  measured bitstrings at N_MEASUREMENTS times of the evolution.
    4. Readout    Ridge regression from the features to the clean image, for
                  the PCA coordinates (baseline) and for the reservoir features.
    5. Metrics    MSE, SSIM and Tenegrad (sharpness) on the test images,
                  reported as mean +- std over the images, as in the paper.

Steps 3-5 are repeated for every reservoir size in ATOM_GRIDS.

Outputs are named after the parameters, so changing any parameter never
overwrites or mixes results:

    results/<RUN_NAME>/features/<rows>x<cols>_<spacing>um_<id>/
        Cached reservoir features, one folder per reservoir configuration:
        train.npy / test.npy (one row per pool image, NaN until simulated) and
        encoding.npz with their inputs and targets (PCA encoding, clean and
        noisy images, labels), so any readout can be trained from this folder
        alone. An interrupted run resumes where it stopped, and raising
        N_TRAIN / N_TEST only simulates the new images.
    results/<RUN_NAME>/runs/<grids>_<spacing>um_pool<P>-<Q>_train<n>_test<m>_<id>/
        log.txt, metrics/ (config.json, results.csv, per_image.csv), figures/.
        <grids> lists the arrays of ATOM_GRIDS, e.g. 3x6 or 3x6+4x6.

<id> is a short hash of every parameter the folder depends on; the full list
is in the config.json inside it.
"""

import contextlib
import csv
import hashlib
import io
import json
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from emu_sv import BitStrings, SVBackend, SVConfig
from pulser.waveforms import ConstantWaveform
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.datasets import MNIST

from rydberg_reservoir import build_constant_sequence, build_register


# ============================================================
# Configuration
# ============================================================

SEED = 42

# Step 1. Data, as in the paper: the first 1000 train / 200 test MNIST images.
# The pools fix the noise, the PCA basis and the rescaling: changing them
# changes the encoding of every image. For a quick run, lower N_TRAIN / N_TEST
# instead and the features can be reused when you scale up.
POOL_TRAIN = 1000  # images used for the noise, PCA and rescaling
POOL_TEST = 200
N_TRAIN = 1000  # images simulated and used to train the readout (limited to POOL_TRAIN)
N_TEST = 200  # images simulated and used for the metrics (limited to POOL_TEST)
NOISE_SIGMA = 0.7  # strength of the multiplicative speckle noise

# Steps 2-3. Reservoir. The paper uses a chain of 18 atoms, (1, 18).
# Number of atoms -> (rows, cols) of the array; (1, n) is a chain of n atoms.
# The encoding uses as many PCA coordinates as there are atoms.
ATOM_GRIDS = {18: (3, 6)}
SPACING = 10.0  # um; nearest-neighbour interaction C6 / SPACING^6: 5.4 rad/us at 10 um, 20.7 at 8 um

# Hamiltonian in the paper's convention (its Eq. 3), constant in time:
#     H = sum_i [RABI sigma_x^i + Delta_i n_i] + sum_{i<j} V_ij n_i n_j,
#     Delta_i = GLOBAL_DETUNING - LOCAL_DETUNING * x_i.
RABI = 2 * np.pi  # rad/us
GLOBAL_DETUNING = 4.5  # rad/us
LOCAL_DETUNING = 9.0  # rad/us
N_MEASUREMENTS = 8  # readout times, one every READOUT_INTERVAL
READOUT_INTERVAL = 500  # ns; the evolution lasts N_MEASUREMENTS * READOUT_INTERVAL
SHOTS = 1000  # measured bitstrings per readout time

# Simulation of the quantum reservoir. The drive is constant, so the time step
# only sets the speed and the GPU memory: the evolution is exact up to
# KRYLOV_TOLERANCE (checked against a 10 ns step). A longer step needs more
# Krylov vectors of 2^N amplitudes each, so larger arrays use a shorter step
# (24 atoms with 100 ns peak at ~9.5 GB of a 16 GB GPU).
TIME_STEPS = {20: 250, 24: 100}  # up to this many atoms -> emu-sv time step in ns
KRYLOV_TOLERANCE = 1e-7
GPUS = (0, 1, 2, 3)  # one worker process per GPU

# Step 4. Linear readout
VALIDATION_FRACTION = 0.10  # training images held out to choose the ridge penalty
RIDGE_PENALTIES = np.logspace(-3, 5, 17)

# Step 5. Metrics. The paper gives neither the Sobel normalisation nor the
# threshold T of its Tenegrad. Its Table 1 values are ~64 times smaller than
# with the plain Sobel kernel used in qrc_mnist, which matches a normalised
# kernel (divided by 8, so G^2 / 64). The factor is the same for every method.
TENG_SOBEL_NORMALIZED = True  # False gives the Tenegrad scale of qrc_mnist
TENG_THRESHOLD = 0.0  # gradient threshold T (not given in the paper)

# Outputs: results/<RUN_NAME>/ (see the module docstring).
ROOT = Path(__file__).resolve().parent
RUN_NAME = "qrc_mnist_rydberg"
RUN_DIR = ROOT / "results" / RUN_NAME

# Readout times as fractions of the sequence duration (1.0 = the end).
READOUT_TIMES = [(k + 1) / N_MEASUREMENTS for k in range(N_MEASUREMENTS)]


# ============================================================
# Helpers: configuration ids and logging
# ============================================================

def config_id(config):
    """Short id of a configuration dict: first 8 hex digits of its SHA-1."""
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


def run_config(n_train, n_test):
    """Every parameter of a run; saved as metrics/config.json in its folder."""
    return {
        "n_train": n_train, "n_test": n_test,
        "pool_train": POOL_TRAIN, "pool_test": POOL_TEST,
        "noise_sigma": NOISE_SIGMA, "seed": SEED,
        "atom_grids": {str(n): list(grid) for n, grid in ATOM_GRIDS.items()},
        "spacing": SPACING, "rabi": RABI, "global_detuning": GLOBAL_DETUNING,
        "local_detuning": LOCAL_DETUNING, "n_measurements": N_MEASUREMENTS,
        "readout_interval": READOUT_INTERVAL, "shots": SHOTS,
        "dt": {str(n): time_step(n) for n in ATOM_GRIDS},
        "krylov_tolerance": KRYLOV_TOLERANCE,
        "validation_fraction": VALIDATION_FRACTION,
        "ridge_penalties": [float(p) for p in RIDGE_PENALTIES],
        "teng_sobel_normalized": TENG_SOBEL_NORMALIZED, "teng_threshold": TENG_THRESHOLD,
    }


class Tee:
    """Stream that writes to several streams at once (screen and log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()  # keep the log file up to date while the run goes on

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name):
        # Anything else (encoding, fileno, isatty...) comes from the screen stream.
        return getattr(self.streams[0], name)


def print_step(title):
    """Print a step header with the current time."""
    print(f"\n[{datetime.now():%H:%M:%S}] === {title} ===")


def format_duration(seconds):
    """Human-readable duration: seconds, minutes or hours."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# ============================================================
# Step 1. Data: MNIST + speckle noise
# ============================================================

def load_data():
    """Load the MNIST pools and corrupt them with speckle noise.

    The noise is multiplicative, x * (1 + NOISE_SIGMA * eps) with
    eps ~ N(0, 1), and pixels are clipped back to [0, 1].

    Returns:
        A dict of CPU tensors: clean_train and noisy_train (POOL_TRAIN rows),
        clean_test and noisy_test (POOL_TEST rows), one image of 784 pixels in
        [0, 1] per row, and test_labels (the digit of every test image).
    """
    generator = torch.Generator().manual_seed(SEED)
    data = {}
    for split, train, n in (("train", True, POOL_TRAIN), ("test", False, POOL_TEST)):
        # The first run downloads MNIST into data/. Its progress indicator
        # rewrites one line with "\r", which becomes a long garbage line in a
        # log file, so it is silenced.
        with contextlib.redirect_stderr(io.StringIO()):
            mnist = MNIST(ROOT / "data", train=train, download=True)
        clean = mnist.data[:n].reshape(n, -1).float() / 255.0
        eps = torch.randn(clean.shape, generator=generator)
        data[f"clean_{split}"] = clean
        data[f"noisy_{split}"] = (clean * (1.0 + NOISE_SIGMA * eps)).clamp(0.0, 1.0)
        data[f"{split}_labels"] = mnist.targets[:n]
    return data


# ============================================================
# Step 2. Encoding: PCA, one coordinate per atom
# ============================================================

def pca_encode(data, n_components):
    """Encode every noisy image as n_components numbers in [0, 1].

    As in the paper: the PCA basis is fitted on the clean training images, the
    noisy images are projected onto it, and every coordinate is rescaled to
    [0, 1] with the min / max of the noisy training projections. Coordinate i
    of an image later drives atom i of the reservoir.

    Returns:
        x_train [POOL_TRAIN, n_components] and x_test [POOL_TEST, n_components].
    """
    mean = data["clean_train"].mean(dim=0)

    # The rows of vh are the principal axes, sorted by explained variance.
    _, _, vh = torch.linalg.svd(data["clean_train"] - mean, full_matrices=False)
    axes = vh[:n_components].T  # [784, n_components]

    x_train = (data["noisy_train"] - mean) @ axes
    x_test = (data["noisy_test"] - mean) @ axes

    x_min = x_train.min(dim=0).values
    x_range = (x_train.max(dim=0).values - x_min).clamp_min(1e-8)
    return (
        ((x_train - x_min) / x_range).clamp(0.0, 1.0),
        ((x_test - x_min) / x_range).clamp(0.0, 1.0),
    )


# ============================================================
# Step 3. Quantum reservoir (emu-sv, one simulation per image)
# ============================================================

def encoded_sequence(x, rows, cols):
    """Pulser sequence of the paper's reservoir with one image written into it.

    The paper writes the Hamiltonian as
        H = sum_i [RABI sigma_x^i + Delta_i n_i] + sum_{i<j} V_ij n_i n_j,
        Delta_i = GLOBAL_DETUNING - LOCAL_DETUNING * x_i,
    while Pulser uses sum_i [Omega/2 sigma_x^i - delta_i n_i] + (same V term).
    Hence Omega = 2 RABI and delta_i = -Delta_i = LOCAL_DETUNING x_i - GLOBAL_DETUNING.

    Pulser adds local detunings with a detuning map modulator (DMM), which only
    takes negative values, so delta_i is written as
        delta_i = (LOCAL_DETUNING - GLOBAL_DETUNING) - LOCAL_DETUNING (1 - x_i):
    a constant global detuning plus a DMM detuning -LOCAL_DETUNING applied to
    atom i with weight 1 - x_i.
    """
    reg = build_register(rows, cols, spacing=SPACING)
    duration = N_MEASUREMENTS * READOUT_INTERVAL
    seq = build_constant_sequence(
        reg, duration=duration, rabi=2 * RABI, detuning=LOCAL_DETUNING - GLOBAL_DETUNING
    )
    weights = {q: 1.0 - float(xi) for q, xi in zip(reg.qubit_ids, x)}
    seq.config_detuning_map(reg.define_detuning_map(weights), "dmm_0")
    seq.add_dmm_detuning(ConstantWaveform(duration, -LOCAL_DETUNING), "dmm_0")
    return seq


def z_features_from_bitstrings(counts, n_atoms):
    """<Z_i> and <Z_i Z_j> (i < j) estimated from measured bitstrings.

    As in the paper, every measured bit b gives Z = (-1)^b ("1" = Rydberg).

    Args:
        counts: {"0110...": number of shots}; character i is atom i.

    Returns:
        n_atoms + n_atoms (n_atoms - 1) / 2 values: the <Z_i>, then the
        <Z_i Z_j> of every pair.
    """
    z = np.array([[1.0 - 2.0 * (c == "1") for c in bitstring] for bitstring in counts])
    shots = np.array(list(counts.values()), dtype=float)
    mean_z = shots @ z / shots.sum()
    mean_zz = (z * shots[:, None]).T @ z / shots.sum()
    i, j = np.triu_indices(n_atoms, k=1)
    return np.concatenate([mean_z, mean_zz[i, j]])


def time_step(n_atoms):
    """emu-sv time step in ns for an array of n_atoms atoms (see TIME_STEPS)."""
    for max_atoms, dt in sorted(TIME_STEPS.items()):
        if n_atoms <= max_atoms:
            return dt
    raise ValueError(f"No time step for {n_atoms} atoms: add one to TIME_STEPS.")


def quantum_features(x, rows, cols, seed):
    """Simulate the reservoir for one encoded image and measure it.

    The state evolves continuously from all atoms in the ground state; at each
    readout time SHOTS bitstrings are sampled from the state, as SHOTS
    repetitions of the experiment measured at that time would give.

    Args:
        x: N_ATOMS values in [0, 1], the PCA encoding of one image.
        seed: random seed of the shot sampling, fixed per image so the
            features are reproducible.

    Returns:
        A float32 array with N_MEASUREMENTS * (N + N (N - 1) / 2) features:
        <Z_i> and <Z_i Z_j> at every readout time.
    """
    torch.manual_seed(seed)
    config = SVConfig(
        observables=[BitStrings(evaluation_times=READOUT_TIMES, num_shots=SHOTS)],
        dt=time_step(rows * cols),
        krylov_tolerance=KRYLOV_TOLERANCE,
        gpu=True,
        log_level=logging.WARN,
    )
    results = SVBackend(encoded_sequence(x, rows, cols), config=config).run()
    return np.concatenate([
        z_features_from_bitstrings(counts, rows * cols) for counts in results.bitstrings
    ]).astype(np.float32)


# Module-level parameters used inside the worker processes. Workers start as
# fresh processes (spawn) and would read the values written in this file, so
# they receive the values of the parent process instead: a script that imports
# this module and changes them (e.g. test12.py) simulates what it asked for.
WORKER_SETTINGS = (
    "SEED", "SPACING", "RABI", "GLOBAL_DETUNING", "LOCAL_DETUNING", "N_MEASUREMENTS",
    "READOUT_INTERVAL", "SHOTS", "READOUT_TIMES", "TIME_STEPS", "KRYLOV_TOLERANCE",
)


def worker_settings():
    """Current values of WORKER_SETTINGS, to hand to the worker processes."""
    return {name: globals()[name] for name in WORKER_SETTINGS}


def _init_worker(gpu_queue, settings):
    """Pin a worker process to one GPU (emu-sv uses the current CUDA device).

    settings (see worker_settings) replace the module parameters of the worker.

    The simulation runs on the GPU and the CPU only drives it, so every worker
    uses one CPU thread: with torch's default (one per core) the 4 workers
    fight over the cores, measured 3x slower while keeping ~24 cores busy each.
    """
    globals().update(settings)
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_queue.get())


def _simulate(job):
    """Worker task: (split, index, x, rows, cols) -> (split, index, features)."""
    split, index, x, rows, cols = job
    seed = SEED + 2 * index + (split == "test")  # different for every image
    return split, index, quantum_features(x, rows, cols, seed)


def reservoir_config(rows, cols):
    """Every setting the cached features of a rows x cols array depend on."""
    return {
        "pool_train": POOL_TRAIN, "pool_test": POOL_TEST, "noise_sigma": NOISE_SIGMA,
        "seed": SEED, "rows": rows, "cols": cols, "spacing": SPACING,
        "rabi": RABI, "global_detuning": GLOBAL_DETUNING,
        "local_detuning": LOCAL_DETUNING, "n_measurements": N_MEASUREMENTS,
        "readout_interval": READOUT_INTERVAL, "shots": SHOTS, "dt": time_step(rows * cols),
        "krylov_tolerance": KRYLOV_TOLERANCE,
    }


def check_cache_config(directory, config):
    """Store the configuration of a cache, and refuse a cache that differs."""
    path = directory / "config.json"
    if not path.exists():
        path.write_text(json.dumps(config, indent=4))
    elif json.loads(path.read_text()) != config:
        raise ValueError(f"{directory} holds features computed with another configuration.")


def check_encoding(directory, encoding):
    """Store the inputs and targets of the cached features, or check them.

    encoding.npz holds the reservoir input of every pool image (x_train,
    x_test), the clean and noisy images and the labels, so that the cache
    folder alone is enough to train any readout (ridge, MLP...). If the file
    already exists, the current encoding must match it, so that features of
    different inputs are never mixed (another torch version could, for
    example, draw different noise).
    """
    path = directory / "encoding.npz"
    arrays = {name: value.numpy() for name, value in encoding.items()}
    if not path.exists():
        np.savez_compressed(path, **arrays)
        return
    with np.load(path) as saved:
        for name, value in arrays.items():
            if not np.allclose(saved[name], value, atol=1e-5):
                raise ValueError(f"{path} differs from the current encoding ({name}).")


def open_cache(path, shape):
    """Open (or create) a float32 .npy cache; rows not computed yet are NaN."""
    if path.exists():
        return np.lib.format.open_memmap(path, mode="r+")
    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    cache[:] = np.nan
    return cache


def run_quantum_reservoir(x_train, x_test, rows, cols, encoding):
    """Reservoir features of every image, simulated in parallel on the GPUs.

    The cache folder is named after the reservoir configuration
    (results/<RUN_NAME>/features/<rows>x<cols>_<spacing>um_<id>/). Images missing from it
    are distributed over GPUS, one worker per GPU, and every result is written
    to disk as soon as it arrives.

    Args:
        x_train, x_test: PCA encodings of the images to simulate.
        encoding: arrays stored in the cache next to the features (see
            check_encoding).

    Returns:
        (train, test) float32 tensors of N_MEASUREMENTS * (N + N (N - 1) / 2)
        features.
    """
    config = reservoir_config(rows, cols)
    directory = RUN_DIR / "features" / f"{rows}x{cols}_{SPACING:g}um_{config_id(config)}"
    directory.mkdir(parents=True, exist_ok=True)
    check_cache_config(directory, config)
    check_encoding(directory, encoding)

    n_atoms = rows * cols
    n_features = N_MEASUREMENTS * (n_atoms + n_atoms * (n_atoms - 1) // 2)
    cache = {
        "train": open_cache(directory / "train.npy", (POOL_TRAIN, n_features)),
        "test": open_cache(directory / "test.npy", (POOL_TEST, n_features)),
    }

    jobs = [
        (split, i, x[i].numpy(), rows, cols)
        for split, x in (("train", x_train), ("test", x_test))
        for i in range(len(x))
        if np.isnan(cache[split][i, 0])
    ]
    n_images = len(x_train) + len(x_test)
    print(f"  Cache {directory.relative_to(ROOT)}: "
          f"{n_images - len(jobs)}/{n_images} images already simulated")

    if jobs:
        print(f"  Simulating {len(jobs)} images on GPUs {GPUS}, one image per GPU at a time")
        # "spawn" gives every worker a clean CUDA context.
        context = mp.get_context("spawn")
        gpu_queue = context.Queue()
        for gpu in GPUS:
            gpu_queue.put(gpu)

        start = time.time()
        report_every = max(1, len(jobs) // 20)  # about 20 progress lines
        with context.Pool(len(GPUS), _init_worker, (gpu_queue, worker_settings())) as pool:
            for done, (split, i, features) in enumerate(
                pool.imap_unordered(_simulate, jobs), start=1
            ):
                cache[split][i] = features
                cache[split].flush()
                if done % report_every == 0 or done == len(jobs):
                    elapsed = time.time() - start
                    left = elapsed / done * (len(jobs) - done)
                    print(f"  {done:>{len(str(len(jobs)))}}/{len(jobs)} images | "
                          f"{format_duration(elapsed)} elapsed | ~{format_duration(left)} left")

    return (
        torch.from_numpy(np.array(cache["train"][: len(x_train)])),
        torch.from_numpy(np.array(cache["test"][: len(x_test)])),
    )


# ============================================================
# Step 4. Linear readout: ridge regression
# ============================================================

def validation_split(n):
    """Fixed random split of n training images into (fit, validation) indices."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_val = max(1, int(n * VALIDATION_FRACTION))
    return order[n_val:], order[:n_val]


def ridge_fit(x, y, penalties):
    """Closed-form ridge regression, one model per penalty.

    Features are standardised and an intercept is fitted. All penalties share
    one eigendecomposition of the Gram matrix G = Xs^T Xs, since
    W(p) = Q diag(1 / (e + p)) Q^T Xs^T (y - mean(y)) with G = Q diag(e) Q^T.

    Returns:
        A list with one prediction function per penalty.
    """
    mean, std = x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)
    xs = (x - mean) / std
    y_mean = y.mean(dim=0)

    eigenvalues, q = torch.linalg.eigh(xs.T @ xs)
    projected = q.T @ (xs.T @ (y - y_mean))

    models = []
    for p in penalties:
        w = q @ (projected / (eigenvalues + p)[:, None])
        models.append(lambda x_new, w=w: ((x_new - mean) / std) @ w + y_mean)
    return models


def ridge_readout(x_train, y_train, x_test):
    """Train the linear readout and predict the test images.

    The penalty is the one in RIDGE_PENALTIES with the lowest MSE on the
    validation images (see validation_split); the model is then refitted on all
    training images with that penalty.

    Returns:
        (predicted test images clipped to [0, 1], chosen penalty, its
        validation MSE)
    """
    x_train, y_train, x_test = x_train.double(), y_train.double(), x_test.double()
    fit, val = validation_split(len(x_train))

    candidates = ridge_fit(x_train[fit], y_train[fit], RIDGE_PENALTIES)
    val_mse = [
        (model(x_train[val]).clamp(0.0, 1.0) - y_train[val]).square().mean().item()
        for model in candidates
    ]
    best = int(np.argmin(val_mse))
    penalty = float(RIDGE_PENALTIES[best])

    (model,) = ridge_fit(x_train, y_train, [penalty])
    return model(x_test).clamp(0.0, 1.0).float(), penalty, val_mse[best]


# ============================================================
# Step 5. Metrics (as in the paper)
# ============================================================

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def tenegrad(images):
    """Tenegrad sharpness of every [784] image (the paper's Eq. 11).

    Mean over the pixels of G^2 = Gx^2 + Gy^2 where G > TENG_THRESHOLD, with
    Gx, Gy the Sobel gradients of the image on a 0-255 scale (as in
    qrc_mnist), using the normalised Sobel kernel if TENG_SOBEL_NORMALIZED.
    """
    sobel = SOBEL_X / 8.0 if TENG_SOBEL_NORMALIZED else SOBEL_X
    x = F.pad(images.reshape(-1, 1, 28, 28) * 255.0, (1, 1, 1, 1), mode="replicate")
    gradients = F.conv2d(x, torch.stack([sobel, sobel.T])[:, None])
    g2 = gradients.square().sum(dim=1)
    return torch.where(g2 > TENG_THRESHOLD**2, g2, 0.0).mean(dim=(1, 2))


def image_metrics(images, clean):
    """MSE, SSIM and Tenegrad of every [784] image (one value per image)."""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
    return {
        "mse": (images - clean).square().mean(dim=1),
        "ssim": ssim(images.reshape(-1, 1, 28, 28), clean.reshape(-1, 1, 28, 28)),
        "teng": tenegrad(images),
    }


def summary_row(n_atoms, method, metrics, **extra):
    """One row of the results table: mean and std of every metric."""
    row = {"n_atoms": n_atoms, "method": method, "n_features": None,
           "penalty": None, "val_mse": None, **extra}
    for name, values in metrics.items():
        row[f"{name}_mean"] = values.mean().item()
        row[f"{name}_std"] = values.std().item()
    return row


# ============================================================
# Outputs: tables and figures
# ============================================================

def print_metrics(rows):
    """Print mean +- std of MSE, SSIM and Tenegrad, one line per row."""
    print(f"  {'atoms':>5s}  {'method':6s} {'MSE':>17s}   {'SSIM':>13s}   {'TENG':>16s}")
    for row in rows:
        atoms = "-" if row["n_atoms"] is None else row["n_atoms"]
        print(f"  {atoms:>5}  {row['method']:6s} "
              f"{row['mse_mean']:7.4f} ± {row['mse_std']:6.4f}   "
              f"{row['ssim_mean']:5.3f} ± {row['ssim_std']:5.3f}   "
              f"{row['teng_mean']:7.1f} ± {row['teng_std']:6.1f}")


def plot_scaling(table, path, title):
    """Mean test MSE, SSIM and Tenegrad versus the number of atoms."""
    atoms = sorted(ATOM_GRIDS)
    value = {(row["method"], row["n_atoms"]): row for row in table}
    reference = {row["method"]: row for row in table if row["n_atoms"] is None}
    styles = {"PCA": {"color": "#52514e", "marker": "s"}, "QRC": {"color": "#2a78d6", "marker": "o"}}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    panels = (("mse", "Test MSE (lower is better)"), ("ssim", "Test SSIM (higher is better)"),
              ("teng", "Test Tenegrad (sharpness)"))
    for ax, (metric, panel_title) in zip(axes, panels):
        for method, style in styles.items():
            ax.plot(atoms, [value[(method, n)][f"{metric}_mean"] for n in atoms],
                    label=method, linewidth=1.8, markersize=6, **style)
        ax.axhline(reference["Noisy"][f"{metric}_mean"], color="#9a9893", linewidth=1.2,
                   linestyle=(0, (4, 3)), label="Noisy input")
        if metric == "teng":
            ax.axhline(reference["Clean"]["teng_mean"], color="#9a9893", linewidth=1.2,
                       linestyle=(0, (1, 2)), label="Clean image")
        ax.set_title(panel_title, loc="left", fontsize=11)
        ax.set_xlabel("Number of atoms")
        ax.set_xticks(atoms)
        ax.grid(color="#e8e7e4", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_examples(images_by_label, path, title, n_show=10):
    """Save the first test images as a grid, one row per label."""
    n_show = min(n_show, len(next(iter(images_by_label.values()))))
    fig, axes = plt.subplots(
        len(images_by_label), n_show,
        figsize=(1.4 * n_show + 1.2, 1.5 * len(images_by_label) + 0.4), squeeze=False,
    )
    for r, (label, images) in enumerate(images_by_label.items()):
        for c in range(n_show):
            axes[r, c].imshow(images[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(table, per_image, examples, run_dir, n_train, n_test):
    """Write the results tables and the figures into the run folder."""
    write_csv(table, run_dir / "metrics" / "results.csv")
    write_csv(per_image, run_dir / "metrics" / "per_image.csv")

    description = (f"{n_train} train / {n_test} test images, PCA pool {POOL_TRAIN} / "
                   f"{POOL_TEST}, speckle sigma = {NOISE_SIGMA}, {SHOTS} shots, ridge readout")
    n_atoms = max(ATOM_GRIDS)
    rows, cols = ATOM_GRIDS[n_atoms]
    save_examples(
        examples, run_dir / "figures" / f"examples_{n_atoms}atoms.png",
        f"First test images, {n_atoms} atoms ({rows} x {cols}, {SPACING:g} um); {description}",
    )
    if len(ATOM_GRIDS) > 1:
        plot_scaling(table, run_dir / "figures" / "scaling.png", f"MNIST denoising: {description}")
    print(f"  Saved in {run_dir.relative_to(ROOT)}: log.txt, metrics/, figures/")


# ============================================================
# Main: the study, step by step
# ============================================================

def run_folder(n_train, n_test):
    """Folder of a run: runs/<grids>_<spacing>um_pool<P>-<Q>_train<n>_test<m>_<id>/."""
    grids = "+".join(f"{rows}x{cols}" for rows, cols in ATOM_GRIDS.values())
    return (RUN_DIR / "runs" / f"{grids}_{SPACING:g}um_pool{POOL_TRAIN}-{POOL_TEST}"
            f"_train{n_train}_test{n_test}_{config_id(run_config(n_train, n_test))}")


def main():
    """Run the whole study with the current parameters.

    Returns:
        (run folder, results table: one dict per row of results.csv)
    """
    start = time.time()
    torch.manual_seed(SEED)

    # Images actually used: N_TRAIN / N_TEST, limited by the size of the pools.
    n_train, n_test = min(N_TRAIN, POOL_TRAIN), min(N_TEST, POOL_TEST)

    # Every run gets its own folder, named after its parameters, with its log,
    # configuration, results tables and figures.
    config = run_config(n_train, n_test)
    run_dir = run_folder(n_train, n_test)
    for folder in ("metrics", "figures"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics" / "config.json").write_text(json.dumps(config, indent=4))

    # Everything printed also goes to run_dir/log.txt, line by line.
    log_file = open(run_dir / "log.txt", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f"Run folder: {run_dir.relative_to(ROOT)}")
    if (n_train, n_test) != (N_TRAIN, N_TEST):
        print(f"Note: N_TRAIN / N_TEST = {N_TRAIN} / {N_TEST} exceed the pools "
              f"({POOL_TRAIN} / {POOL_TEST}); using {n_train} / {n_test} images")

    # Step 1. Data: clean and noisy MNIST images.
    print_step("Step 1. Data: MNIST + speckle noise")
    data = load_data()
    y_train = data["clean_train"][:n_train]  # what the readout must reproduce
    clean_test = data["clean_test"][:n_test]
    noisy_test = data["noisy_test"][:n_test]
    labels = data["test_labels"][:n_test]
    print(f"  Pools: {POOL_TRAIN} train / {POOL_TEST} test images, "
          f"speckle noise sigma = {NOISE_SIGMA}")
    print(f"  Used:  {n_train} train / {n_test} test images")

    # Step 2. Encoding. The first n PCA coordinates do not depend on how many
    # are computed, so the encoding is done once for the largest reservoir and
    # every smaller reservoir uses its first n coordinates.
    print_step("Step 2. Encoding: PCA of the noisy images, one coordinate per atom")
    x_pool_train, x_pool_test = pca_encode(data, max(ATOM_GRIDS))
    x_train, x_test = x_pool_train[:n_train], x_pool_test[:n_test]
    print(f"  {max(ATOM_GRIDS)} coordinates per image, PCA fitted on the {POOL_TRAIN} "
          "clean training images, rescaled to [0, 1]")

    # References: the clean images (their sharpness) and the noisy input.
    table, per_image = [], []
    examples = {"Clean": clean_test, "Noisy": noisy_test}
    for method, images in examples.items():
        metrics = image_metrics(images, clean_test)
        table.append(summary_row(None, method, metrics))
        per_image += [{"n_atoms": None, "method": method, "image": k, "digit": int(labels[k]),
                       **{name: values[k].item() for name, values in metrics.items()}}
                      for k in range(n_test)]

    for n_atoms, (rows, cols) in ATOM_GRIDS.items():
        x_train_n, x_test_n = x_train[:, :n_atoms], x_test[:, :n_atoms]

        # Step 3. Quantum reservoir: <Z_i>, <Z_i Z_j> from SHOTS bitstrings.
        print_step(f"Step 3. Quantum reservoir: {n_atoms} atoms ({rows} x {cols}), "
                   f"{SHOTS} shots at {N_MEASUREMENTS} times")
        # Stored in the cache with the features: the encoding of every pool
        # image, the clean and noisy images and the labels.
        encoding = {"x_train": x_pool_train[:, :n_atoms], "x_test": x_pool_test[:, :n_atoms], **data}
        feature_sets = {
            "PCA": (x_train_n, x_test_n),
            "QRC": run_quantum_reservoir(x_train_n, x_test_n, rows, cols, encoding),
        }
        for method, (features, _) in feature_sets.items():
            print(f"  {method}: {features.shape[1]} features per image")

        # Step 4. Linear readout (ridge) for the PCA baseline and the reservoir.
        print_step(f"Step 4. Ridge readout: {n_atoms} atoms")
        print(f"  Penalty chosen among {len(RIDGE_PENALTIES)} values "
              f"({RIDGE_PENALTIES[0]:.0e} to {RIDGE_PENALTIES[-1]:.0e}) by the MSE on "
              f"{len(validation_split(n_train)[1])} held-out training images")
        predictions = {}
        for method, (features_train, features_test) in feature_sets.items():
            predictions[method] = ridge_readout(features_train, y_train, features_test)
            _, penalty, val_mse = predictions[method]
            edge = "  (edge of the grid)" if penalty in (RIDGE_PENALTIES[0], RIDGE_PENALTIES[-1]) else ""
            print(f"  {method}: penalty {penalty:.0e} | validation MSE {val_mse:.5f}{edge}")

        # Step 5. Test metrics.
        print_step(f"Step 5. Test metrics: {n_atoms} atoms ({n_test} test images)")
        rows_n = []
        for method, (prediction, penalty, val_mse) in predictions.items():
            metrics = image_metrics(prediction, clean_test)
            rows_n.append(summary_row(
                n_atoms, method, metrics, n_features=feature_sets[method][0].shape[1],
                penalty=penalty, val_mse=val_mse,
            ))
            per_image += [{"n_atoms": n_atoms, "method": method, "image": k,
                           "digit": int(labels[k]),
                           **{name: values[k].item() for name, values in metrics.items()}}
                          for k in range(n_test)]
            if n_atoms == max(ATOM_GRIDS):
                examples[method] = prediction
        print_metrics(rows_n)
        table += rows_n

    print_step("Summary: mean ± std over the test images")
    print_metrics(table)
    print()
    save_results(table, per_image, examples, run_dir, n_train, n_test)
    print(f"  Total time: {format_duration(time.time() - start)}")

    # Back to the screen only, so that a script calling main() several times
    # (e.g. test12.py) does not keep writing into this run's log.
    sys.stdout, sys.stderr = sys.stdout.streams[0], sys.stderr.streams[0]
    log_file.close()
    return run_dir, table


if __name__ == "__main__":
    main()
