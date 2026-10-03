"""Spacing scan of the 3 x 6 reservoir, then the 4 x 6 reservoir at the best spacing.

Uses the study of qrc_mnist_rydberg.py (every run keeps its usual folder and
feature cache, nothing is overwritten) in three steps:

    1. Scan    The 3 x 6 array (18 atoms) at every spacing in SPACINGS. Runs
               that already exist are read from their results.csv instead of
               being repeated.
    2. Choice  The spacing with the lowest validation MSE of the QRC readout
               (training images held out to choose the ridge penalty). The
               test images take no part in the choice, so their metrics stay
               an honest estimate.
    3. Final   The 3 x 6 and 4 x 6 arrays at that spacing in one run (the
               3 x 6 features come from the cache of step 1). One 4 x 6 image
               is simulated first to check that it fits in GPU memory, with a
               shorter time step if it does not.

The scan table is saved as results/qrc_mnist_rydberg/spacing_scan_3x6.csv.
Run with: python test12.py (the scan takes ~30 min, the 4 x 6 run ~1 day).
"""

import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from pulser.devices import MockDevice

import qrc_mnist_rydberg as q


# ============================================================
# Configuration
# ============================================================

SPACINGS = (8.0, 9.0, 10.0, 11.0)  # um; the 3 x 6 runs at 8 and 10 um already exist
SCAN_GRID = (3, 6)
FINAL_GRIDS = {18: (3, 6), 24: (4, 6)}

# Time steps tried for the largest final array until one fits in GPU memory.
# Stronger interactions (smaller spacing) need more Krylov vectors per step,
# and every vector holds 2^N amplitudes. Each step must divide READOUT_INTERVAL.
PROBE_TIME_STEPS = (100, 50, 25)  # ns
MEMORY_FRACTION = 0.85  # largest share of the GPU memory a worker may use


# ============================================================
# Step 1. Scan of the spacing
# ============================================================

def read_results(run_dir):
    """Rows of a run's results.csv, keyed by (n_atoms, method)."""
    with open(run_dir / "metrics" / "results.csv") as file:
        return {(row["n_atoms"], row["method"]): row for row in csv.DictReader(file)}


def scan(n_train, n_test):
    """Run (or read) the SCAN_GRID study at every spacing; one summary row each."""
    n_atoms = SCAN_GRID[0] * SCAN_GRID[1]
    summary = []
    for spacing in SPACINGS:
        q.SPACING, q.ATOM_GRIDS = spacing, {n_atoms: SCAN_GRID}
        run_dir = q.run_folder(n_train, n_test)
        if (run_dir / "metrics" / "results.csv").exists():
            print(f"  {spacing:g} um: already done, {run_dir.relative_to(q.ROOT)}")
        else:
            print(f"  {spacing:g} um: running")
            q.main()
        results = read_results(run_dir)
        qrc, pca = results[(str(n_atoms), "QRC")], results[(str(n_atoms), "PCA")]
        summary.append({
            "spacing_um": spacing,
            "nn_interaction_rad_us": MockDevice.interaction_coeff / spacing**6,
            "qrc_val_mse": float(qrc["val_mse"]),
            **{f"qrc_{m}": float(qrc[f"{m}_mean"]) for m in ("mse", "ssim", "teng")},
            **{f"pca_{m}": float(pca[f"{m}_mean"]) for m in ("mse", "ssim", "teng")},
            "run": run_dir.name,
        })
    return summary


def print_scan(summary):
    """Scan table: validation MSE (the choice) and test metrics of every spacing."""
    print(f"  {'spacing':>7s} {'V_nn':>9s} {'val MSE':>8s}   "
          f"{'test MSE':>8s} {'SSIM':>6s} {'TENG':>7s}   (QRC; PCA: test MSE / SSIM / TENG)")
    for row in summary:
        print(f"  {row['spacing_um']:5g} um {row['nn_interaction_rad_us']:5.1f} r/us "
              f"{row['qrc_val_mse']:8.5f}   {row['qrc_mse']:8.4f} {row['qrc_ssim']:6.3f} "
              f"{row['qrc_teng']:7.1f}   ({row['pca_mse']:.4f} / {row['pca_ssim']:.3f} / "
              f"{row['pca_teng']:.1f})")


# ============================================================
# Step 3. Time step of the largest array (GPU memory check)
# ============================================================

def _probe(rows, cols):
    """Worker: simulate one random image of a rows x cols array.

    Returns:
        (seconds, peak share of the GPU memory), or None if out of memory.
    """
    x = np.random.default_rng(0).uniform(0.0, 1.0, rows * cols)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        q.quantum_features(x, rows, cols, seed=0)
    except torch.cuda.OutOfMemoryError:
        return None
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return time.time() - start, torch.cuda.max_memory_allocated() / total


def choose_time_step(rows, cols, n_images):
    """Set the time step of a rows x cols array to the first one that fits.

    Every candidate of PROBE_TIME_STEPS is tried on one image in a worker
    process like the ones of the real run (same settings, one GPU).
    """
    n_atoms = rows * cols
    context = mp.get_context("spawn")
    for dt in PROBE_TIME_STEPS:
        q.TIME_STEPS = {**q.TIME_STEPS, n_atoms: dt}
        gpu_queue = context.Queue()
        gpu_queue.put(q.GPUS[0])
        with ProcessPoolExecutor(1, mp_context=context, initializer=q._init_worker,
                                 initargs=(gpu_queue, q.worker_settings())) as pool:
            result = pool.submit(_probe, rows, cols).result()
        if result is None:
            print(f"  time step {dt} ns: out of GPU memory")
            continue
        seconds, memory = result
        print(f"  time step {dt} ns: {seconds:.0f} s per image, "
              f"{100 * memory:.0f}% of the GPU memory")
        if memory <= MEMORY_FRACTION:
            hours = seconds * n_images / len(q.GPUS) / 3600
            print(f"  -> {dt} ns; {n_images} images on {len(q.GPUS)} GPUs: ~{hours:.0f} h")
            return dt
    raise RuntimeError(f"No time step in {PROBE_TIME_STEPS} fits {n_atoms} atoms in GPU memory.")


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()
    n_train, n_test = min(q.N_TRAIN, q.POOL_TRAIN), min(q.N_TEST, q.POOL_TEST)
    rows, cols = SCAN_GRID

    # Step 1. The 3 x 6 study at every spacing.
    q.print_step(f"test12, step 1. Spacing scan, {rows} x {cols} array: "
                 + ", ".join(f"{s:g}" for s in SPACINGS) + " um")
    summary = scan(n_train, n_test)

    # Step 2. Best spacing by the validation MSE of the QRC readout.
    q.print_step("test12, step 2. Choice of the spacing (lowest QRC validation MSE)")
    print_scan(summary)
    path = q.RUN_DIR / f"spacing_scan_{rows}x{cols}.csv"
    q.write_csv(summary, path)
    best = min(summary, key=lambda row: row["qrc_val_mse"])
    print(f"  Best spacing: {best['spacing_um']:g} um (table saved as {path.relative_to(q.ROOT)})")

    # Step 3. All the final arrays at that spacing, in one run.
    q.SPACING, q.ATOM_GRIDS = best["spacing_um"], dict(FINAL_GRIDS)
    big_rows, big_cols = FINAL_GRIDS[max(FINAL_GRIDS)]
    q.print_step(f"test12, step 3. Final run at {q.SPACING:g} um: "
                 + ", ".join(f"{r} x {c}" for r, c in FINAL_GRIDS.values()))
    print(f"  GPU memory check, {big_rows} x {big_cols} array:")
    choose_time_step(big_rows, big_cols, n_train + n_test)
    run_dir, _ = q.main()

    q.print_step("test12 finished")
    print(f"  Final run: {run_dir.relative_to(q.ROOT)}")
    print(f"  Total time: {q.format_duration(time.time() - start)}")


if __name__ == "__main__":
    main()
