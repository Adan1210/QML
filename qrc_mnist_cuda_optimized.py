#!/usr/bin/env python3

# ============================================================
# OPTIMIZED QRC MNIST DENOISING - SINGLE NVIDIA GPU
#
# GPU stack:
#
#   - RAPIDS cuML + CuPy/DLPack:
#       PCA entirely on CUDA
#
#   - PyTorch CUDA:
#       Batched statevector simulation
#       Batched diagonal Hamiltonian evaluation
#       Exact expectation values
#
#   - PyTorch CUDA + AMP:
#       MLP readout training
#
#   - TorchMetrics:
#       SSIM
#
#   - Kornia:
#       Sobel gradients / Tenegrad
#
# Main optimization:
#
#   state.shape = [batch_size, 2**N_QUBITS]
#
# Therefore Q_BATCH really controls the number of quantum
# states simulated simultaneously on the GPU.
#
# Quantum evolution is incremental:
#
#   t = 0
#     -> 0.5 us
#     -> 1.0 us
#     -> ...
#     -> 4.0 us
#
# We DO NOT restart the simulation from |0...0> at every
# measurement time.
#
# With:
#
#   N_MEASUREMENTS = 8
#   TROTTER_STEPS_PER_WINDOW = 8
#
# total Trotter steps per image = 64.
# ============================================================


# ============================================================
# 0. SELECT GPU
# ============================================================

import os

# Change this to "1", "2", or "3" to use another A16.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================
# 1. IMPORTS
# ============================================================

import copy
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import cupy as cp

import torch
import torch.nn as nn

from torchvision.datasets import MNIST

from cuml.decomposition import PCA as cuPCA

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
)

import kornia

import matplotlib.pyplot as plt


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

N_TRAIN = 9000
N_TEST = 1000


# ------------------------------------------------------------
# Start here.
#
# For the scaling experiment change ONLY:
#
#   12 -> 15 -> 18
#
# initially.
# ------------------------------------------------------------

N_QUBITS = 18


# ------------------------------------------------------------
# Noise
# ------------------------------------------------------------

NOISE_SIGMA = 0.7


# ------------------------------------------------------------
# Quantum reservoir
# ------------------------------------------------------------

N_MEASUREMENTS = 8

MEASUREMENT_DT_US = 0.5

TROTTER_STEPS_PER_WINDOW = 8


# ------------------------------------------------------------
# Measurement mode
#
# None:
#     exact <Zi> and <ZiZj>
#
# 1000:
#     finite-shot measurements
#
# Start with exact values for the scaling study.
# ------------------------------------------------------------

N_SHOTS = None


# ------------------------------------------------------------
# Rydberg-like Hamiltonian parameters
#
# Units:
#
#   Omega, Delta, V : rad / us
#   distance         : um
# ------------------------------------------------------------

OMEGA = (
    2.0
    *
    math.pi
)

DELTA_GLOBAL = 4.5

DELTA_SCALE = 9.0

ATOM_SPACING_UM = 10.0

C6 = 5.42e6


# ============================================================
# 3. QUANTUM BATCH CONFIGURATION
# ============================================================

# If True, the program benchmarks multiple batch sizes using
# one measurement window and selects the highest throughput.
#
# This is useful because the optimal batch depends strongly
# on N_QUBITS.

AUTO_TUNE_Q_BATCH = True


# Used only when AUTO_TUNE_Q_BATCH = False.

Q_BATCH = 512


# Candidate batches for automatic tuning.
#
# Values larger than N_TRAIN are automatically ignored.

Q_BATCH_CANDIDATES = [
    64,
    128,
    256,
    512,
    1000,
]


# ============================================================
# 4. MLP CONFIGURATION
# ============================================================

MLP_BATCH_SIZE = 64

LEARNING_RATE = 1e-3

MAX_EPOCHS = 500

PATIENCE = 20

VALIDATION_FRACTION = 0.10

DROPOUT = 0.30

USE_AMP = True


# ============================================================
# 5. CACHE
# ============================================================

# Set to False whenever you want to force a new quantum
# feature extraction.

USE_QRC_CACHE = False


# ============================================================
# 6. CUDA DEVICE
# ============================================================

assert torch.cuda.is_available(), (
    "CUDA is not available."
)

DEVICE = torch.device(
    "cuda:0"
)

torch.cuda.set_device(
    DEVICE
)


torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision(
    "high"
)


# ============================================================
# 7. OUTPUT PATHS
# ============================================================

if N_SHOTS is None:

    SHOTS_LABEL = "exact"

else:

    SHOTS_LABEL = (
        f"{N_SHOTS}shots"
    )


EXPERIMENT_NAME = (
    f"qrc_mnist_"
    f"{N_QUBITS}q_"
    f"{N_MEASUREMENTS}times_"
    f"{SHOTS_LABEL}_"
    f"vectorized_cuda"
)


BASE_DIR = (
    Path("./results")
    /
    EXPERIMENT_NAME
)


FEATURE_DIR = (
    BASE_DIR
    /
    "features"
)

MODEL_DIR = (
    BASE_DIR
    /
    "models"
)

METRIC_DIR = (
    BASE_DIR
    /
    "metrics"
)

FIGURE_DIR = (
    BASE_DIR
    /
    "figures"
)


for directory in [

    FEATURE_DIR,
    MODEL_DIR,
    METRIC_DIR,
    FIGURE_DIR,

]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


QRC_CACHE_FILE = (
    FEATURE_DIR
    /
    "quantum_features.pt"
)


PREDICTION_FILE = (
    FEATURE_DIR
    /
    "predictions.pt"
)


GLOBAL_SCALING_FILE = (
    Path("./results")
    /
    "qrc_scaling_summary.csv"
)


# ============================================================
# 8. GENERAL UTILITIES
# ============================================================

def print_header(
    title,
):

    print()

    print(
        "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


def seed_everything(
    seed,
):

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )


def gpu_memory_gb():

    return (
        torch.cuda.memory_allocated()
        /
        1024**3
    )


def gpu_peak_memory_gb():

    return (
        torch.cuda.max_memory_allocated()
        /
        1024**3
    )


seed_everything(
    SEED
)


# ============================================================
# 9. CUDA INFORMATION
# ============================================================

print_header(
    "CUDA DEVICE"
)


gpu_properties = (
    torch.cuda.get_device_properties(
        0
    )
)


TOTAL_VRAM_GB = (
    gpu_properties.total_memory
    /
    1024**3
)


print(
    "GPU:",
    torch.cuda.get_device_name(
        0
    )
)


print(
    f"VRAM: "
    f"{TOTAL_VRAM_GB:.2f} GB"
)


print(
    "CUDA version:",
    torch.version.cuda
)


print(
    "N qubits:",
    N_QUBITS
)


print(
    "Hilbert dimension:",
    2 ** N_QUBITS
)


# ============================================================
# 10. LOAD MNIST
# ============================================================

def load_mnist():

    print_header(
        "LOADING MNIST"
    )


    train_dataset = MNIST(

        root="./data",

        train=True,

        download=True,

    )


    test_dataset = MNIST(

        root="./data",

        train=False,

        download=True,

    )


    clean_train = (

        train_dataset.data[
            :N_TRAIN
        ]

        .reshape(
            N_TRAIN,
            -1,
        )

        .to(
            DEVICE,
            dtype=torch.float32,
        )

        /
        255.0
    )


    clean_test = (

        test_dataset.data[
            :N_TEST
        ]

        .reshape(
            N_TEST,
            -1,
        )

        .to(
            DEVICE,
            dtype=torch.float32,
        )

        /
        255.0
    )


    test_labels = (

        test_dataset.targets[
            :N_TEST
        ]

        .clone()
    )


    print(
        "clean_train:",
        clean_train.shape,
        clean_train.device,
    )


    print(
        "clean_test :",
        clean_test.shape,
        clean_test.device,
    )


    return (
        clean_train,
        clean_test,
        test_labels,
    )


# ============================================================
# 11. SPECKLE NOISE
# ============================================================

def add_speckle_noise(
    images,
    seed,
):

    generator = torch.Generator(
        device=DEVICE
    )


    generator.manual_seed(
        seed
    )


    epsilon = torch.randn(

        images.shape,

        generator=generator,

        device=DEVICE,

        dtype=images.dtype,

    )


    noisy = (

        images

        *

        (
            1.0

            +

            NOISE_SIGMA
            *
            epsilon
        )

    )


    return noisy.clamp(
        0.0,
        1.0,
    )


# ============================================================
# 12. PCA ON CUDA
#
# PyTorch CUDA
#      |
#      | DLPack zero-copy
#      v
# CuPy CUDA
#      |
#      v
# RAPIDS cuML
#      |
#      | DLPack zero-copy
#      v
# PyTorch CUDA
# ============================================================

def run_cuda_pca(
    clean_train,
    noisy_train,
    noisy_test,
):

    print_header(
        "PCA ON CUDA"
    )


    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # PyTorch CUDA -> CuPy CUDA.
    # --------------------------------------------------------

    clean_train_cp = cp.from_dlpack(
        clean_train.contiguous()
    )


    noisy_train_cp = cp.from_dlpack(
        noisy_train.contiguous()
    )


    noisy_test_cp = cp.from_dlpack(
        noisy_test.contiguous()
    )


    # --------------------------------------------------------
    # RAPIDS cuML PCA.
    #
    # The PCA basis is fitted using CLEAN training images.
    # --------------------------------------------------------

    pca = cuPCA(

        n_components=N_QUBITS,

        whiten=False,

        output_type="cupy",

    )


    pca.fit(
        clean_train_cp
    )


    # --------------------------------------------------------
    # Project NOISY images.
    # --------------------------------------------------------

    Z_train_cp = pca.transform(
        noisy_train_cp
    )


    Z_test_cp = pca.transform(
        noisy_test_cp
    )


    # --------------------------------------------------------
    # CuPy CUDA -> PyTorch CUDA.
    # --------------------------------------------------------

    Z_train = torch.from_dlpack(
        Z_train_cp
    ).float()


    Z_test = torch.from_dlpack(
        Z_test_cp
    ).float()


    torch.cuda.synchronize()


    elapsed = (
        time.perf_counter()
        -
        start
    )


    # --------------------------------------------------------
    # Normalize every PCA coordinate using training statistics.
    # --------------------------------------------------------

    z_min = Z_train.min(
        dim=0
    ).values


    z_max = Z_train.max(
        dim=0
    ).values


    z_range = (

        z_max

        -

        z_min

    ).clamp_min(
        1e-8
    )


    Z_train = (

        Z_train

        -

        z_min

    ) / z_range


    Z_test = (

        Z_test

        -

        z_min

    ) / z_range


    Z_train = Z_train.clamp(
        0.0,
        1.0,
    )


    Z_test = Z_test.clamp(
        0.0,
        1.0,
    )


    print(
        f"PCA time: {elapsed:.3f} s"
    )


    print(
        "PCA basis dimension:",
        N_QUBITS
    )


    print(
        "Z_train:",
        Z_train.shape,
        Z_train.device,
    )


    print(
        "Z_test :",
        Z_test.shape,
        Z_test.device,
    )


    return (
        Z_train,
        Z_test,
    )


# ============================================================
# 13. BUILD QUANTUM CONSTANTS
# ============================================================

@torch.inference_mode()
def build_quantum_constants():
    """
    Build all quantities that are independent of the input
    image.

    All tensors remain on the GPU.
    """

    print_header(
        "BUILDING QUANTUM CONSTANTS"
    )


    dim = (
        1
        <<
        N_QUBITS
    )


    basis_indices = torch.arange(

        dim,

        device=DEVICE,

        dtype=torch.long,

    )


    qubit_indices = torch.arange(

        N_QUBITS,

        device=DEVICE,

        dtype=torch.long,

    )


    # --------------------------------------------------------
    # occupation[state, qubit]
    #
    # = 0 or 1
    # --------------------------------------------------------

    occupation = (

        (
            basis_indices[:, None]

            >>

            qubit_indices[None, :]
        )

        &

        1

    ).to(
        torch.float32
    )


    # ========================================================
    # RYDBERG INTERACTION MATRIX
    # ========================================================

    positions = (

        torch.arange(

            N_QUBITS,

            device=DEVICE,

            dtype=torch.float32,

        )

        *

        ATOM_SPACING_UM

    )


    distances = torch.abs(

        positions[:, None]

        -

        positions[None, :]

    )


    V = torch.zeros(

        (
            N_QUBITS,
            N_QUBITS,
        ),

        device=DEVICE,

        dtype=torch.float32,

    )


    interaction_mask = (
        distances > 0
    )


    V[
        interaction_mask
    ] = (

        C6

        /

        distances[
            interaction_mask
        ].pow(
            6
        )

    )


    # ========================================================
    # Interaction energy:
    #
    # sum_{i<j} V_ij n_i n_j
    #
    # The factor 1/2 removes double counting.
    # ========================================================

    interaction_energy = (

        0.5

        *

        torch.sum(

            (
                occupation
                @
                V
            )

            *

            occupation,

            dim=1,

        )

    )


    # ========================================================
    # Z eigenvalues
    #
    # occupation 0 -> Z = +1
    # occupation 1 -> Z = -1
    # ========================================================

    z_basis = (

        1.0

        -

        2.0
        *
        occupation

    )


    # --------------------------------------------------------
    # All unique qubit pairs.
    # --------------------------------------------------------

    pair_indices = torch.triu_indices(

        N_QUBITS,

        N_QUBITS,

        offset=1,

        device=DEVICE,

    )


    pair_i = pair_indices[
        0
    ]


    pair_j = pair_indices[
        1
    ]


    # --------------------------------------------------------
    # <Zi Zj> eigenvalues.
    # --------------------------------------------------------

    zz_basis = (

        z_basis[
            :,
            pair_i
        ]

        *

        z_basis[
            :,
            pair_j
        ]

    )


    # --------------------------------------------------------
    # Complete observable matrix:
    #
    # [Z1, ..., Zn, Z1Z2, Z1Z3, ...]
    #
    # shape:
    #
    # [2**N_QUBITS, FEATURES_PER_TIME]
    # --------------------------------------------------------

    observables = torch.cat(

        [
            z_basis,
            zz_basis,
        ],

        dim=1,

    ).contiguous()


    features_per_time = (
        observables.shape[
            1
        ]
    )


    total_features = (

        features_per_time

        *

        N_MEASUREMENTS

    )


    nearest_neighbor_v = (

        C6

        /

        ATOM_SPACING_UM**6

    )


    print(
        "Hilbert dimension:",
        dim
    )


    print(
        "Nearest-neighbour V:",
        nearest_neighbor_v,
        "rad/us"
    )


    print(
        "Features/time:",
        features_per_time
    )


    print(
        "Total features:",
        total_features
    )


    print(
        f"Observable matrix VRAM: "
        f"{observables.numel() * 4 / 1024**2:.2f} MiB"
    )


    return {

        "dim":
            dim,

        "occupation":
            occupation,

        "interaction_energy":
            interaction_energy,

        "observables":
            observables,

        "features_per_time":
            features_per_time,

        "total_features":
            total_features,

    }


# ============================================================
# 14. APPLY THE RABI DRIVE
# ============================================================

@torch.inference_mode()
def apply_rx(
    state,
    qubit,
    cosine,
    sine_complex,
):
    """
    Apply RX(theta) to one qubit for every statevector in
    the batch simultaneously.

    state shape:

        [batch, 2**N_QUBITS]
    """

    batch_size = (
        state.shape[
            0
        ]
    )


    stride = (
        1
        <<
        qubit
    )


    block = (
        2
        *
        stride
    )


    view = state.view(

        batch_size,

        -1,

        2,

        stride,

    )


    # Copies are required because both halves are overwritten.

    amplitude_zero = (
        view[
            :,
            :,
            0,
            :
        ]
        .clone()
    )


    amplitude_one = (
        view[
            :,
            :,
            1,
            :
        ]
        .clone()
    )


    view[
        :,
        :,
        0,
        :
    ] = (

        cosine
        *
        amplitude_zero

        +

        sine_complex
        *
        amplitude_one

    )


    view[
        :,
        :,
        1,
        :
    ] = (

        sine_complex
        *
        amplitude_zero

        +

        cosine
        *
        amplitude_one

    )


# ============================================================
# 15. MEASURE QRC FEATURES
# ============================================================

@torch.inference_mode()
def measure_quantum_features(
    state,
    observables,
):
    """
    Measure all <Zi> and <Zi Zj> observables.

    For exact mode this becomes one CUDA matrix
    multiplication:

        probabilities @ observables
    """

    probabilities = (

        state.real.square()

        +

        state.imag.square()

    )


    if N_SHOTS is None:

        # ----------------------------------------------------
        # Exact expectation values.
        # ----------------------------------------------------

        return (

            probabilities

            @

            observables

        )


    # ========================================================
    # FINITE SHOTS
    # ========================================================

    samples = torch.multinomial(

        probabilities,

        num_samples=N_SHOTS,

        replacement=True,

    )


    sampled_observables = observables[
        samples
    ]


    return sampled_observables.mean(
        dim=1
    )


# ============================================================
# 16. SIMULATE ONE QRC BATCH
# ============================================================

@torch.inference_mode()
def simulate_qrc_batch(
    z_batch,
    constants,
    n_measurements=N_MEASUREMENTS,
):
    """
    Simulate a complete batch of input images.

    Crucially, the quantum state is evolved continuously.

    For every measurement window:

        8 Trotter steps
        -> measure
        -> continue from the SAME state
    """

    occupation = constants[
        "occupation"
    ]


    interaction_energy = constants[
        "interaction_energy"
    ]


    observables = constants[
        "observables"
    ]


    dim = constants[
        "dim"
    ]


    batch_size = (
        z_batch.shape[
            0
        ]
    )


    # ========================================================
    # INPUT-DEPENDENT DETUNINGS
    # ========================================================

    delta = (

        DELTA_GLOBAL

        -

        DELTA_SCALE
        *
        z_batch

    )


    # ========================================================
    # DIAGONAL ENERGY
    #
    # E(s) =
    #     sum_i Delta_i n_i
    #     + sum_i<j V_ij n_i n_j
    #
    # The matrix multiplication is executed by CUDA/cuBLAS.
    # ========================================================

    local_energy = (

        delta

        @

        occupation.T

    )


    diagonal_energy = (

        local_energy

        +

        interaction_energy[
            None,
            :
        ]

    )


    del local_energy


    # ========================================================
    # TROTTTER PARAMETERS
    # ========================================================

    dt = (

        MEASUREMENT_DT_US

        /

        TROTTER_STEPS_PER_WINDOW

    )


    rx_angle = (

        2.0

        *

        OMEGA

        *

        dt

    )


    cosine = math.cos(
        rx_angle
        /
        2.0
    )


    sine_complex = (

        -1j

        *

        math.sin(
            rx_angle
            /
            2.0
        )

    )


    # ========================================================
    # DIAGONAL HALF-STEP
    #
    # exp(-i H_diag dt / 2)
    # ========================================================

    phase_half = torch.exp(

        (
            -0.5j

            *

            dt

        )

        *

        diagonal_energy

    ).to(
        torch.complex64
    )


    del diagonal_energy


    # ========================================================
    # INITIAL STATE |000...0>
    # ========================================================

    state = torch.zeros(

        (
            batch_size,
            dim,
        ),

        device=DEVICE,

        dtype=torch.complex64,

    )


    state[
        :,
        0
    ] = (
        1.0
        +
        0.0j
    )


    features_over_time = []


    # ========================================================
    # CONTINUOUS EVOLUTION
    # ========================================================

    for measurement_index in range(
        n_measurements
    ):

        # ----------------------------------------------------
        # Advance by one 0.5 us measurement window.
        # ----------------------------------------------------

        for _ in range(
            TROTTER_STEPS_PER_WINDOW
        ):

            # -----------------------------------------------
            # Half diagonal step.
            # -----------------------------------------------

            state.mul_(
                phase_half
            )


            # -----------------------------------------------
            # exp(-i Omega sum X_i dt)
            #
            # All X_i commute, therefore this is a product
            # of independent RX gates.
            # -----------------------------------------------

            for qubit in range(
                N_QUBITS
            ):

                apply_rx(

                    state,

                    qubit,

                    cosine,

                    sine_complex,

                )


            # -----------------------------------------------
            # Second diagonal half-step.
            # -----------------------------------------------

            state.mul_(
                phase_half
            )


        # ----------------------------------------------------
        # Read observables without collapsing the state.
        # ----------------------------------------------------

        features = measure_quantum_features(

            state,

            observables,

        )


        features_over_time.append(
            features
        )


    return torch.cat(

        features_over_time,

        dim=1,

    )


# ============================================================
# 17. AUTOMATIC Q_BATCH BENCHMARK
# ============================================================

@torch.inference_mode()
def select_best_q_batch(
    Z_train,
    constants,
):

    if not AUTO_TUNE_Q_BATCH:

        return min(
            Q_BATCH,
            len(
                Z_train
            ),
        )


    print_header(
        "CUDA Q_BATCH AUTOTUNE"
    )


    valid_candidates = [

        batch_size

        for batch_size
        in Q_BATCH_CANDIDATES

        if batch_size
        <=
        len(
            Z_train
        )

    ]


    if len(
        valid_candidates
    ) == 0:

        return len(
            Z_train
        )


    # ========================================================
    # SMALL WARM-UP
    # ========================================================

    warmup_size = min(

        16,

        len(
            Z_train
        ),

    )


    _ = simulate_qrc_batch(

        Z_train[
            :warmup_size
        ],

        constants,

        n_measurements=1,

    )


    torch.cuda.synchronize()


    best_batch = None

    best_throughput = -1.0


    print()

    print(
        f"{'Batch':>8s} "
        f"{'Time [s]':>12s} "
        f"{'images/s':>12s} "
        f"{'Peak VRAM':>12s}"
    )


    print(
        "-" * 50
    )


    for batch_size in valid_candidates:

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()


        subset = Z_train[
            :batch_size
        ]


        try:

            torch.cuda.synchronize()


            start = (
                time.perf_counter()
            )


            features = simulate_qrc_batch(

                subset,

                constants,

                # Benchmark only one 0.5 us window.
                n_measurements=1,

            )


            torch.cuda.synchronize()


            elapsed = (

                time.perf_counter()

                -

                start

            )


            throughput = (

                batch_size

                /

                elapsed

            )


            peak_vram = (
                gpu_peak_memory_gb()
            )


            print(

                f"{batch_size:8d} "
                f"{elapsed:12.4f} "
                f"{throughput:12.2f} "
                f"{peak_vram:10.2f} GB"

            )


            if throughput > best_throughput:

                best_throughput = (
                    throughput
                )

                best_batch = (
                    batch_size
                )


            del features


        except torch.OutOfMemoryError:

            print(

                f"{batch_size:8d} "
                f"{'OOM':>12s}"

            )


            torch.cuda.empty_cache()


    if best_batch is None:

        raise RuntimeError(
            "No Q_BATCH candidate fits in GPU memory."
        )


    print()

    print(
        "Selected Q_BATCH:",
        best_batch
    )


    print(
        f"Benchmark throughput: "
        f"{best_throughput:.2f} images/s"
    )


    return best_batch


# ============================================================
# 18. FULL QRC FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def extract_qrc_features(
    Z,
    constants,
    batch_size,
    name,
):

    print_header(
        f"QRC {name.upper()} FEATURES"
    )


    print(
        "Q_BATCH:",
        batch_size
    )


    print(
        "Internal Trotter dt:",
        MEASUREMENT_DT_US
        /
        TROTTER_STEPS_PER_WINDOW,
        "us"
    )


    print(
        "Total Trotter steps/image:",
        N_MEASUREMENTS
        *
        TROTTER_STEPS_PER_WINDOW
    )


    all_features = []


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()


    total_start = (
        time.perf_counter()
    )


    processed = 0


    for batch_start in range(

        0,

        len(
            Z
        ),

        batch_size,

    ):

        batch_end = min(

            batch_start
            +
            batch_size,

            len(
                Z
            ),

        )


        z_batch = Z[
            batch_start:
            batch_end
        ]


        batch_start_time = (
            time.perf_counter()
        )


        batch_features = simulate_qrc_batch(

            z_batch,

            constants,

            n_measurements=N_MEASUREMENTS,

        )


        torch.cuda.synchronize()


        batch_elapsed = (

            time.perf_counter()

            -

            batch_start_time

        )


        # Move the finished embedding to CPU.
        #
        # This prevents the output cache from occupying GPU
        # memory while later quantum batches run.

        all_features.append(

            batch_features.cpu()

        )


        processed += len(
            z_batch
        )


        total_elapsed = (

            time.perf_counter()

            -

            total_start

        )


        average_per_image = (

            total_elapsed

            /

            processed

        )


        remaining_seconds = (

            (
                len(
                    Z
                )

                -

                processed
            )

            *

            average_per_image

        )


        print(

            f"images "
            f"{processed:4d}/"
            f"{len(Z):4d} | "

            f"batch "
            f"{batch_elapsed:8.3f} s | "

            f"{len(z_batch) / batch_elapsed:7.2f} img/s | "

            f"remaining ~"
            f"{remaining_seconds / 60:7.2f} min",

            flush=True,

        )


        del batch_features


    torch.cuda.synchronize()


    total_elapsed = (

        time.perf_counter()

        -

        total_start

    )


    peak_vram = (
        gpu_peak_memory_gb()
    )


    output = torch.cat(

        all_features,

        dim=0,

    )


    print()

    print(
        f"{name} extraction time: "
        f"{total_elapsed:.3f} s "
        f"({total_elapsed / 60:.2f} min)"
    )


    print(
        f"Time/image: "
        f"{total_elapsed / len(Z):.5f} s"
    )


    print(
        f"Throughput: "
        f"{len(Z) / total_elapsed:.2f} images/s"
    )


    print(
        f"Peak VRAM: "
        f"{peak_vram:.2f} / "
        f"{TOTAL_VRAM_GB:.2f} GB"
    )


    return (

        output,

        total_elapsed,

        peak_vram,

    )


# ============================================================
# 19. DENOISING MLP
# ============================================================

class DenoisingMLP(
    nn.Module
):

    def __init__(
        self,
        input_dim,
    ):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                1024,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),


            nn.Linear(
                1024,
                512,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),


            nn.Linear(
                512,
                784,
            ),

            nn.Sigmoid(),

        )


    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 20. TRAIN / VALIDATION SPLIT
# ============================================================

def create_split():

    generator = torch.Generator(
        device="cpu"
    )


    generator.manual_seed(
        SEED
    )


    permutation = torch.randperm(

        N_TRAIN,

        generator=generator,

    )


    n_validation = int(

        N_TRAIN

        *

        VALIDATION_FRACTION

    )


    validation_indices = (

        permutation[
            :n_validation
        ]

        .to(
            DEVICE
        )

    )


    training_indices = (

        permutation[
            n_validation:
        ]

        .to(
            DEVICE
        )

    )


    return (
        training_indices,
        validation_indices,
    )


# ============================================================
# 21. TRAIN MLP
# ============================================================

def train_model(
    X,
    Y,
    training_indices,
    validation_indices,
    name,
):

    X_fit = X[
        training_indices
    ]


    Y_fit = Y[
        training_indices
    ]


    X_validation = X[
        validation_indices
    ]


    Y_validation = Y[
        validation_indices
    ]


    model = DenoisingMLP(

        X.shape[
            1
        ]

    ).to(
        DEVICE
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

    )


    criterion = (
        nn.MSELoss()
    )


    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=USE_AMP,

    )


    best_validation_loss = float(
        "inf"
    )


    best_state = None

    patience_counter = 0


    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )


    for epoch in range(
        1,
        MAX_EPOCHS
        +
        1
    ):

        model.train()


        permutation = torch.randperm(

            len(
                X_fit
            ),

            device=DEVICE,

        )


        training_loss = 0.0


        for batch_start in range(

            0,

            len(
                X_fit
            ),

            MLP_BATCH_SIZE,

        ):

            batch_indices = permutation[

                batch_start:
                batch_start
                +
                MLP_BATCH_SIZE

            ]


            xb = X_fit[
                batch_indices
            ]


            yb = Y_fit[
                batch_indices
            ]


            optimizer.zero_grad(
                set_to_none=True
            )


            with torch.amp.autocast(

                device_type="cuda",

                dtype=torch.float16,

                enabled=USE_AMP,

            ):

                prediction = model(
                    xb
                )


                loss = criterion(

                    prediction,

                    yb,

                )


            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            training_loss += (

                loss.item()

                *

                len(
                    xb
                )

            )


        training_loss /= len(
            X_fit
        )


        model.eval()


        with torch.inference_mode():

            validation_prediction = model(
                X_validation
            )


            validation_loss = criterion(

                validation_prediction,

                Y_validation,

            ).item()


        if (

            epoch == 1

            or

            epoch % 10 == 0

        ):

            print(

                f"{name:8s} | "
                f"epoch {epoch:3d} | "
                f"train {training_loss:.6f} | "
                f"val {validation_loss:.6f}"

            )


        if (

            validation_loss

            <

            best_validation_loss

            -

            1e-7

        ):

            best_validation_loss = (
                validation_loss
            )


            best_state = copy.deepcopy(
                model.state_dict()
            )


            patience_counter = 0


        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            print(

                f"{name}: early stopping "
                f"at epoch {epoch}"

            )

            break


    torch.cuda.synchronize()


    elapsed = (

        time.perf_counter()

        -

        start

    )


    model.load_state_dict(
        best_state
    )


    print(
        f"{name} training time: "
        f"{elapsed:.2f} s"
    )


    return model


# ============================================================
# 22. IMAGE METRICS
# ============================================================

def mse_per_image(
    prediction,
    target,
):

    return (

        prediction

        .sub(
            target
        )

        .square()

        .mean(
            dim=1
        )

    )


def ssim_per_image(
    prediction,
    target,
):

    metric = (

        StructuralSimilarityIndexMeasure(

            data_range=1.0,

            reduction="none",

        )

        .to(
            DEVICE
        )

    )


    return metric(

        prediction,

        target,

    )


def tenegrad_per_image(
    images,
):

    # Use an 8-bit-like intensity scale.

    images_255 = (

        images

        *

        255.0

    )


    gradients = (
        kornia.filters
        .spatial_gradient(

            images_255,

            mode="sobel",

            order=1,

            normalized=False,

        )
    )


    gx = gradients[
        :,
        :,
        0,
    ]


    gy = gradients[
        :,
        :,
        1,
    ]


    return (

        gx.square()

        +

        gy.square()

    ).mean(

        dim=(
            1,
            2,
            3,
        )

    )


# ============================================================
# 23. METRIC REPORTING
# ============================================================

def metric_summary(
    values,
):

    return {

        "mean":
            values.mean().item(),

        "std":
            values.std(
                unbiased=True
            ).item(),

    }


def print_metric(
    name,
    values,
):

    summary = metric_summary(
        values
    )


    print(

        f"{name:22s}: "
        f"{summary['mean']:.6f} "
        f"+- "
        f"{summary['std']:.6f}"

    )


# ============================================================
# 24. APPEND SCALING RESULTS
# ============================================================

def append_scaling_result(
    selected_q_batch,
    train_time,
    test_time,
    peak_vram,
    mse_noisy,
    mse_pca,
    mse_qrc,
    ssim_noisy,
    ssim_pca,
    ssim_qrc,
    teng_clean,
    teng_noisy,
    teng_pca,
    teng_qrc,
):

    GLOBAL_SCALING_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,

    )


    file_exists = (
        GLOBAL_SCALING_FILE.exists()
    )


    row = {

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "n_qubits":
            N_QUBITS,

        "hilbert_dimension":
            2 ** N_QUBITS,

        "q_batch":
            selected_q_batch,

        "features_per_time":
            N_QUBITS
            +
            N_QUBITS
            *
            (
                N_QUBITS - 1
            )
            //
            2,

        "total_features":
            (
                N_QUBITS
                +
                N_QUBITS
                *
                (
                    N_QUBITS - 1
                )
                //
                2
            )
            *
            N_MEASUREMENTS,

        "train_qrc_time_s":
            train_time,

        "test_qrc_time_s":
            test_time,

        "total_qrc_time_s":
            train_time
            +
            test_time,

        "peak_vram_gb":
            peak_vram,

        "mse_noisy":
            mse_noisy.mean().item(),

        "mse_pca":
            mse_pca.mean().item(),

        "mse_qrc":
            mse_qrc.mean().item(),

        "ssim_noisy":
            ssim_noisy.mean().item(),

        "ssim_pca":
            ssim_pca.mean().item(),

        "ssim_qrc":
            ssim_qrc.mean().item(),

        "tenegrad_clean":
            teng_clean.mean().item(),

        "tenegrad_noisy":
            teng_noisy.mean().item(),

        "tenegrad_pca":
            teng_pca.mean().item(),

        "tenegrad_qrc":
            teng_qrc.mean().item(),

    }


    with open(

        GLOBAL_SCALING_FILE,

        "a",

        newline="",

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(
                row.keys()
            ),

        )


        if not file_exists:

            writer.writeheader()


        writer.writerow(
            row
        )


# ============================================================
# 25. MAIN EXPERIMENT
# ============================================================

def main():

    # ========================================================
    # DATA
    # ========================================================

    (
        clean_train,
        clean_test,
        test_labels,
    ) = load_mnist()


    noisy_train = add_speckle_noise(

        clean_train,

        seed=SEED,

    )


    noisy_test = add_speckle_noise(

        clean_test,

        seed=SEED + 1,

    )


    # ========================================================
    # PCA
    # ========================================================

    (
        Z_train,
        Z_test,
    ) = run_cuda_pca(

        clean_train,

        noisy_train,

        noisy_test,

    )


    # ========================================================
    # QUANTUM CONSTANTS
    # ========================================================

    constants = (
        build_quantum_constants()
    )


    # ========================================================
    # CHOOSE QUANTUM BATCH SIZE
    # ========================================================

    selected_q_batch = (
        select_best_q_batch(

            Z_train,

            constants,

        )
    )


    # ========================================================
    # QRC FEATURES
    # ========================================================

    if (

        USE_QRC_CACHE

        and

        QRC_CACHE_FILE.exists()

    ):

        print_header(
            "LOADING CACHED QRC FEATURES"
        )


        cache = torch.load(

            QRC_CACHE_FILE,

            map_location="cpu",

            weights_only=False,

        )


        Q_train_cpu = cache[
            "Q_train"
        ]


        Q_test_cpu = cache[
            "Q_test"
        ]


        train_qrc_time = cache[
            "train_time"
        ]


        test_qrc_time = cache[
            "test_time"
        ]


        peak_vram = cache[
            "peak_vram"
        ]


    else:

        (
            Q_train_cpu,
            train_qrc_time,
            train_peak_vram,
        ) = extract_qrc_features(

            Z_train,

            constants,

            selected_q_batch,

            "train",

        )


        (
            Q_test_cpu,
            test_qrc_time,
            test_peak_vram,
        ) = extract_qrc_features(

            Z_test,

            constants,

            selected_q_batch,

            "test",

        )


        peak_vram = max(

            train_peak_vram,

            test_peak_vram,

        )


        torch.save(

            {

                "Q_train":
                    Q_train_cpu,

                "Q_test":
                    Q_test_cpu,

                "train_time":
                    train_qrc_time,

                "test_time":
                    test_qrc_time,

                "peak_vram":
                    peak_vram,

                "q_batch":
                    selected_q_batch,

                "n_qubits":
                    N_QUBITS,

                "n_shots":
                    N_SHOTS,

            },

            QRC_CACHE_FILE,

        )


    # ========================================================
    # MOVE QRC EMBEDDINGS BACK TO CUDA
    # ========================================================

    Q_train = Q_train_cpu.to(

        DEVICE,

        dtype=torch.float32,

        non_blocking=True,

    )


    Q_test = Q_test_cpu.to(

        DEVICE,

        dtype=torch.float32,

        non_blocking=True,

    )


    print_header(
        "QRC EXTRACTION SUMMARY"
    )


    print(
        "Q_train:",
        Q_train.shape
    )


    print(
        "Q_test:",
        Q_test.shape
    )


    print(
        f"Train: "
        f"{train_qrc_time:.3f} s"
    )


    print(
        f"Test : "
        f"{test_qrc_time:.3f} s"
    )


    print(
        f"Total: "
        f"{train_qrc_time + test_qrc_time:.3f} s"
    )


    print(
        f"Peak VRAM: "
        f"{peak_vram:.2f} GB"
    )


    # ========================================================
    # FIXED VALIDATION SPLIT
    # ========================================================

    (
        training_indices,
        validation_indices,
    ) = create_split()


    # ========================================================
    # TRAIN QRC READOUT
    # ========================================================

    print_header(
        "TRAINING QRC READOUT"
    )


    qrc_model = train_model(

        Q_train,

        clean_train,

        training_indices,

        validation_indices,

        "QRC",

    )


    # ========================================================
    # TRAIN PCA BASELINE
    # ========================================================

    print_header(
        "TRAINING PCA BASELINE"
    )


    pca_model = train_model(

        Z_train,

        clean_train,

        training_indices,

        validation_indices,

        "PCA",

    )


    # ========================================================
    # SAVE MODELS
    # ========================================================

    torch.save(

        qrc_model.state_dict(),

        MODEL_DIR
        /
        "qrc_mlp.pt",

    )


    torch.save(

        pca_model.state_dict(),

        MODEL_DIR
        /
        "pca_mlp.pt",

    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    qrc_model.eval()

    pca_model.eval()


    with torch.inference_mode():

        pred_qrc = (

            qrc_model(
                Q_test
            )

            .clamp(
                0.0,
                1.0,
            )

        )


        pred_pca = (

            pca_model(
                Z_test
            )

            .clamp(
                0.0,
                1.0,
            )

        )


    # ========================================================
    # IMAGE SHAPES
    # ========================================================

    clean_images = clean_test.reshape(

        -1,
        1,
        28,
        28,

    )


    noisy_images = noisy_test.reshape(

        -1,
        1,
        28,
        28,

    )


    pca_images = pred_pca.reshape(

        -1,
        1,
        28,
        28,

    )


    qrc_images = pred_qrc.reshape(

        -1,
        1,
        28,
        28,

    )


    # ========================================================
    # MSE
    # ========================================================

    mse_noisy = mse_per_image(

        noisy_test,

        clean_test,

    )


    mse_pca = mse_per_image(

        pred_pca,

        clean_test,

    )


    mse_qrc = mse_per_image(

        pred_qrc,

        clean_test,

    )


    # ========================================================
    # SSIM
    # ========================================================

    with torch.inference_mode():

        ssim_noisy = ssim_per_image(

            noisy_images,

            clean_images,

        )


        ssim_pca = ssim_per_image(

            pca_images,

            clean_images,

        )


        ssim_qrc = ssim_per_image(

            qrc_images,

            clean_images,

        )


    # ========================================================
    # TENEGRAD
    # ========================================================

    with torch.inference_mode():

        teng_clean = tenegrad_per_image(
            clean_images
        )


        teng_noisy = tenegrad_per_image(
            noisy_images
        )


        teng_pca = tenegrad_per_image(
            pca_images
        )


        teng_qrc = tenegrad_per_image(
            qrc_images
        )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print_header(
        "FINAL RESULTS"
    )


    print()

    print(
        "MSE -- lower is better"
    )

    print(
        "-" * 70
    )


    print_metric(
        "Noisy MSE",
        mse_noisy,
    )


    print_metric(
        "PCA MSE",
        mse_pca,
    )


    print_metric(
        "QRC MSE",
        mse_qrc,
    )


    print()

    print(
        "SSIM -- higher is better"
    )

    print(
        "-" * 70
    )


    print_metric(
        "Noisy SSIM",
        ssim_noisy,
    )


    print_metric(
        "PCA SSIM",
        ssim_pca,
    )


    print_metric(
        "QRC SSIM",
        ssim_qrc,
    )


    print()

    print(
        "Tenegrad -- higher means stronger gradients"
    )

    print(
        "-" * 70
    )


    print_metric(
        "Clean Tenegrad",
        teng_clean,
    )


    print_metric(
        "Noisy Tenegrad",
        teng_noisy,
    )


    print_metric(
        "PCA Tenegrad",
        teng_pca,
    )


    print_metric(
        "QRC Tenegrad",
        teng_qrc,
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    torch.save(

        {

            "labels":
                test_labels,

            "clean":
                clean_test.cpu(),

            "noisy":
                noisy_test.cpu(),

            "pca":
                pred_pca.cpu(),

            "qrc":
                pred_qrc.cpu(),

        },

        PREDICTION_FILE,

    )


    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {

        "configuration": {

            "seed":
                SEED,

            "gpu":
                torch.cuda.get_device_name(
                    0
                ),

            "n_qubits":
                N_QUBITS,

            "hilbert_dimension":
                2 ** N_QUBITS,

            "n_train":
                N_TRAIN,

            "n_test":
                N_TEST,

            "q_batch":
                selected_q_batch,

            "n_measurements":
                N_MEASUREMENTS,

            "measurement_dt_us":
                MEASUREMENT_DT_US,

            "trotter_steps_per_window":
                TROTTER_STEPS_PER_WINDOW,

            "total_trotter_steps":
                N_MEASUREMENTS
                *
                TROTTER_STEPS_PER_WINDOW,

            "n_shots":
                N_SHOTS,

            "noise_sigma":
                NOISE_SIGMA,

            "omega":
                OMEGA,

            "delta_global":
                DELTA_GLOBAL,

            "delta_scale":
                DELTA_SCALE,

            "atom_spacing_um":
                ATOM_SPACING_UM,

            "c6":
                C6,

            "qrc_train_time_s":
                train_qrc_time,

            "qrc_test_time_s":
                test_qrc_time,

            "qrc_total_time_s":
                train_qrc_time
                +
                test_qrc_time,

            "peak_vram_gb":
                peak_vram,

        },


        "mse": {

            "noisy":
                metric_summary(
                    mse_noisy
                ),

            "pca":
                metric_summary(
                    mse_pca
                ),

            "qrc":
                metric_summary(
                    mse_qrc
                ),

        },


        "ssim": {

            "noisy":
                metric_summary(
                    ssim_noisy
                ),

            "pca":
                metric_summary(
                    ssim_pca
                ),

            "qrc":
                metric_summary(
                    ssim_qrc
                ),

        },


        "tenegrad": {

            "clean":
                metric_summary(
                    teng_clean
                ),

            "noisy":
                metric_summary(
                    teng_noisy
                ),

            "pca":
                metric_summary(
                    teng_pca
                ),

            "qrc":
                metric_summary(
                    teng_qrc
                ),

        },

    }


    summary_file = (

        METRIC_DIR

        /

        "metrics_summary.json"

    )


    with open(

        summary_file,

        "w",

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,

        )


    # ========================================================
    # SAVE PER-IMAGE METRICS
    # ========================================================

    per_image_file = (

        METRIC_DIR

        /

        "metrics_per_image.csv"

    )


    with open(

        per_image_file,

        "w",

        newline="",

    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [

                "image_index",

                "digit",

                "mse_noisy",
                "mse_pca",
                "mse_qrc",

                "ssim_noisy",
                "ssim_pca",
                "ssim_qrc",

                "tenegrad_clean",
                "tenegrad_noisy",
                "tenegrad_pca",
                "tenegrad_qrc",

            ]
        )


        for i in range(
            N_TEST
        ):

            writer.writerow(
                [

                    i,

                    int(
                        test_labels[
                            i
                        ]
                    ),

                    mse_noisy[
                        i
                    ].item(),

                    mse_pca[
                        i
                    ].item(),

                    mse_qrc[
                        i
                    ].item(),

                    ssim_noisy[
                        i
                    ].item(),

                    ssim_pca[
                        i
                    ].item(),

                    ssim_qrc[
                        i
                    ].item(),

                    teng_clean[
                        i
                    ].item(),

                    teng_noisy[
                        i
                    ].item(),

                    teng_pca[
                        i
                    ].item(),

                    teng_qrc[
                        i
                    ].item(),

                ]
            )


    # ========================================================
    # SAVE EXAMPLE IMAGES
    # ========================================================

    n_show = min(
        10,
        N_TEST,
    )


    image_rows = [

        (
            clean_images[
                :n_show,
                0
            ].cpu(),
            "Clean",
        ),

        (
            noisy_images[
                :n_show,
                0
            ].cpu(),
            "Noisy",
        ),

        (
            pca_images[
                :n_show,
                0
            ].cpu(),
            "PCA",
        ),

        (
            qrc_images[
                :n_show,
                0
            ].cpu(),
            "QRC",
        ),

    ]


    figure, axes = plt.subplots(

        4,

        n_show,

        figsize=(
            2 * n_show,
            8,
        ),

    )


    for row_index, (
        images,
        label,
    ) in enumerate(
        image_rows
    ):

        for column_index in range(
            n_show
        ):

            axes[
                row_index,
                column_index
            ].imshow(

                images[
                    column_index
                ],

                cmap="gray",

                vmin=0,

                vmax=1,

            )


            axes[
                row_index,
                column_index
            ].axis(
                "off"
            )


        axes[
            row_index,
            0
        ].set_ylabel(
            label
        )


    plt.tight_layout()


    figure_file = (

        FIGURE_DIR

        /

        "denoising_comparison.png"

    )


    plt.savefig(

        figure_file,

        dpi=200,

        bbox_inches="tight",

    )


    plt.close(
        figure
    )


    # ========================================================
    # APPEND GLOBAL SCALING TABLE
    # ========================================================

    append_scaling_result(

        selected_q_batch,

        train_qrc_time,

        test_qrc_time,

        peak_vram,

        mse_noisy,
        mse_pca,
        mse_qrc,

        ssim_noisy,
        ssim_pca,
        ssim_qrc,

        teng_clean,
        teng_noisy,
        teng_pca,
        teng_qrc,

    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print_header(
        "SAVED OUTPUTS"
    )


    print(
        "Experiment:",
        BASE_DIR
    )


    print(
        "Summary:",
        summary_file
    )


    print(
        "Scaling table:",
        GLOBAL_SCALING_FILE
    )


    print(
        "Figure:",
        figure_file
    )


# ============================================================
# 26. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()