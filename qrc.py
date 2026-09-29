#!/usr/bin/env python3
"""Generate fixed QRC features from PCA data. No training is performed.

Example: python qrc_features.py --input pca.npy --output qrc_features.npz
Default output: one row per sample and 1368 columns (18 qubits, 8 times).
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pennylane as qml


# 1. Settings: edit the defaults here or pass options from the terminal.

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=Path("pca.npy"))
parser.add_argument("--input-key", default=None)
parser.add_argument("--output", type=Path, default=Path("qrc_features.npz"))
parser.add_argument("--n-qubits", type=int, default=18)
parser.add_argument("--windows", "--measurement-windows", type=int, default=8)
parser.add_argument("--steps-per-window", type=int, default=10)
parser.add_argument("--total-time", type=float, default=1.0)
parser.add_argument("--omega", type=float, default=1.0)
parser.add_argument("--delta-global", type=float, default=0.0)
parser.add_argument("--delta-scale", type=float, default=1.0)
parser.add_argument("--c6", type=float, default=1.0)
parser.add_argument("--spacing", type=float, default=1.0)
parser.add_argument("--device", "--backend", default="auto")
parser.add_argument("--no-fallback", action="store_true")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--batch-mode", choices=["auto", "broadcast", "sequential"], default="auto")
parser.add_argument("--output-dtype", choices=["float64", "float32"], default="float64")
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

n_qubits = args.n_qubits
windows = args.windows
steps_per_window = args.steps_per_window
total_time = args.total_time
omega = args.omega
delta_global = args.delta_global
delta_scale = args.delta_scale
c6 = args.c6
spacing = args.spacing
batch_size = args.batch_size


# 2. Load the PCA matrix, assuming valid data and settings.

input_key = args.input_key
if args.input.suffix.lower() == ".npy":
    pca_data = np.load(args.input, allow_pickle=False, mmap_mode="r")
else:
    with np.load(args.input, allow_pickle=False) as archive:
        if input_key is None:
            input_key = archive.files[0]
        pca_data = archive[input_key]


# 3. Set the times and all pair interactions in the 1D chain.
# Units: hbar=1, angular frequencies, no extra factor 1/2 on omega.
# The default physical values are illustrative, not an experimental calibration.

window_duration = total_time / windows
dt = window_duration / steps_per_window
times = np.arange(1, windows + 1) * window_duration
interactions = []

for i in range(n_qubits):
    for j in range(i + 1, n_qubits):
        distance = spacing * (j - i)
        coupling = c6 / distance**6
        interactions.append((i, j, coupling))

observables_per_window = n_qubits + len(interactions)
n_features = windows * observables_per_window


# 4. The only function describes the quantum circuit for PennyLane.
# Each execution starts in |00...0>. The input changes only the detunings.
# Second-order Strang splitting approximates exp(-i H t).


def quantum_circuit(z, steps):
    delta = delta_global - delta_scale * z

    for step in range(steps):
        # Half of the X evolution.
        for i in range(n_qubits):
            qml.RX(omega * dt, wires=i)

        # Detuning: exp(-i * delta_i * n_i * dt), with n_i = (I-Z_i)/2.
        for i in range(n_qubits):
            if z.ndim == 1:
                local_delta = delta[i]
            else:
                local_delta = delta[:, i]
            qml.PhaseShift(-dt * local_delta, wires=i)

        # Interaction: exp(-i * V_ij * n_i * n_j * dt).
        for i, j, coupling in interactions:
            qml.ControlledPhaseShift(-dt * coupling, wires=[i, j])

        # The other half of the X evolution.
        for i in range(n_qubits):
            qml.RX(omega * dt, wires=i)

    measurements = []
    for i in range(n_qubits):
        measurements.append(qml.expval(qml.PauliZ(i)))
    for i, j, coupling in interactions:
        measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
    return tuple(measurements)


# 5. Try GPU first, then CPU. Only the selected device is printed.

started = time.perf_counter()
if args.device == "auto":
    candidates = ["lightning.gpu", "lightning.qubit", "default.qubit"]
else:
    candidates = [args.device]
    if not args.no_fallback:
        for name in ["lightning.qubit", "default.qubit"]:
            if name not in candidates:
                candidates.append(name)

selected_device = None
backend_failures = []
for name in candidates:
    try:
        device = qml.device(name, wires=n_qubits, shots=None)
        circuit = qml.QNode(quantum_circuit, device, interface=None, diff_method=None)
        circuit(np.zeros(n_qubits), steps_per_window)
    except Exception as error:
        backend_failures.append(f"{name}: {error}")
        continue
    selected_device = name
    break

if selected_device is None:
    raise RuntimeError("No usable device. " + " | ".join(backend_failures))

use_broadcast = False
if args.batch_mode == "broadcast":
    use_broadcast = True
elif args.batch_mode == "auto" and selected_device == "default.qubit":
    use_broadcast = True

if not args.quiet:
    print(f"Device: {selected_device}")
    print(f"Output shape: ({len(pca_data)}, {n_features})")


# 6. Compute the features in batches and save the result.
# Each time is measured by replaying the evolution from zero to that time.
# These are instantaneous measurements, not averages over a time interval.

args.output.parent.mkdir(parents=True, exist_ok=True)
broadcast_failed = False

with tempfile.TemporaryDirectory(prefix="qrc-", dir=args.output.parent) as tempdir:
    # Keep the growing output on disk instead of filling RAM.
    features = np.lib.format.open_memmap(
        Path(tempdir) / "features.npy",
        mode="w+",
        dtype=args.output_dtype,
        shape=(len(pca_data), n_features),
    )

    for start in range(0, len(pca_data), batch_size):
        stop = min(start + batch_size, len(pca_data))
        batch = np.asarray(pca_data[start:stop], dtype=np.float64)

        for window in range(windows):
            steps = (window + 1) * steps_per_window
            values = None

            if use_broadcast and len(batch) > 1:
                try:
                    # One returned array per observable; stack them as columns.
                    values = np.stack(circuit(batch, steps), axis=1)
                except Exception:
                    use_broadcast = False
                    broadcast_failed = True
                    values = None

            if values is None:
                rows = []
                for sample in batch:
                    measurements = circuit(sample, steps)
                    rows.append(np.asarray(measurements, dtype=np.float64))
                values = np.stack(rows)

            first_column = window * observables_per_window
            last_column = first_column + observables_per_window
            features[start:stop, first_column:last_column] = values

        if not args.quiet:
            print(f"Processed {stop}/{len(pca_data)} samples", flush=True)

    features.flush()

    # Describe the order of columns for Python and Julia.
    observable_i = list(range(n_qubits))
    observable_j = [-1] * n_qubits
    for i, j, coupling in interactions:
        observable_i.append(i)
        observable_j.append(j)

    execution_mode = "sequential"
    if use_broadcast:
        execution_mode = "broadcast"

    metadata = {
        "config": {
            "n_qubits": n_qubits,
            "windows": windows,
            "steps_per_window": steps_per_window,
            "total_time": total_time,
            "omega": omega,
            "delta_global": delta_global,
            "delta_scale": delta_scale,
            "c6": c6,
            "spacing": spacing,
        },
        "input_file": str(args.input.resolve()),
        "input_key": input_key,
        "device_requested": args.device,
        "device_used": selected_device,
        "backend_failures": backend_failures,
        "batch_mode_requested": args.batch_mode,
        "batch_size": batch_size,
        "execution_mode": execution_mode,
        "broadcast_failed": broadcast_failed,
        "output_dtype": args.output_dtype,
        "pennylane_version": qml.__version__,
        "numpy_version": np.__version__,
        "scheme": "second_order_strang",
        "initial_state": "all_zero",
        "units": "hbar=1; angular frequencies; no extra omega/2",
        "layout": "rows=samples; columns=time-major then singles then lexicographic pairs",
        "wire_index_base": 0,
        "single_observable_j_sentinel": -1,
        "time_semantics": "instantaneous endpoints; independent replay from zero",
        "shots": None,
        "training": False,
        "elapsed_compute_seconds": time.perf_counter() - started,
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")

    # Replace the destination only after the archive has been written successfully.
    temporary_output = Path(tempdir) / "output.npz"
    np.savez_compressed(
        temporary_output,
        features=features,
        times=times,
        observable_i=np.asarray(observable_i, dtype=np.int64),
        observable_j=np.asarray(observable_j, dtype=np.int64),
        feature_time_index=np.repeat(np.arange(windows), observables_per_window),
        feature_i=np.tile(observable_i, windows),
        feature_j=np.tile(observable_j, windows),
        metadata_json_utf8=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    temporary_output.replace(args.output)
    del features

if not args.quiet:
    print(f"Saved: {args.output.resolve()}")


