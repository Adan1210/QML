"""Rydberg-atom analog simulator (GPU) for quantum reservoir computing.

Simulates a global adiabatic sweep on a rectangular array of neutral atoms
using Pasqal's emu-sv backend (exact state-vector, PyTorch/CUDA) and returns
the per-atom Rydberg occupation, which can be used as reservoir features.

Pipeline:
    build_register  ->  build_sequence           ->  run
    (geometry)          or build_constant_sequence   (simulation + readout)
                        (pulse program)

Conventions (Pulser):
    - Time is in nanoseconds, amplitude/detuning in rad/us, distances in um.
    - Qubit state "1" is the Rydberg state, "0" is the ground state.
    - Hamiltonian: H = sum_i [Omega/2 sigma_x^i - delta_i n_i]
                       + sum_{i<j} C6 / r_ij^6 n_i n_j,  with n_i = |r><r|_i.
"""

import logging

import numpy as np
import pulser
from pulser.devices import MockDevice
from pulser.waveforms import InterpolatedWaveform
from emu_sv import SVBackend, SVConfig, Occupation


def build_register(n_rows, n_cols, spacing=6.0):
    """Create a rectangular atom array.

    Args:
        n_rows: number of rows in the array.
        n_cols: number of columns in the array.
        spacing: distance between nearest neighbours, in um. Together with the
            Rydberg C6 coefficient it sets the interaction strength (and the
            blockade radius) between atoms.

    Returns:
        A pulser.Register with n_rows * n_cols atoms named "q0", "q1", ...
        Atoms are numbered row by row, so a result vector can be reshaped
        with `.reshape(n_rows, n_cols)`.
    """
    return pulser.Register.rectangle(n_rows, n_cols, spacing=spacing, prefix="q")


def build_sequence(reg, durations=(400, 3200, 400), max_rabi=15.8, max_detuning=16.33):
    """Build a global adiabatic sweep acting on every atom equally.

    The sequence has three segments (ramp up, sweep, ramp down):
        - Rabi amplitude: 0 -> max_rabi, hold at max_rabi, max_rabi -> 0.
        - Detuning: held at -max_detuning during the ramp up, swept linearly
          to +max_detuning during the middle segment, then held at
          +max_detuning during the ramp down.
    Sweeping the detuning across resonance with the interaction on drives
    the system towards a Rydberg-ordered state (e.g. antiferromagnetic).

    Args:
        reg: the pulser.Register returned by `build_register`.
        durations: (ramp up, sweep, ramp down) durations in ns.
        max_rabi: peak Rabi frequency in rad/us.
        max_detuning: absolute value of the detuning at both ends, rad/us.

    Returns:
        A pulser.Sequence ready to be simulated by `run`.
    """
    total = sum(durations)

    # Breakpoints of the piecewise waveforms, as fractions of the total
    # duration: start, end of ramp up, end of sweep, end of ramp down.
    times = [0, durations[0] / total, (durations[0] + durations[1]) / total, 1]

    # Waveforms are interpolated between the (time, value) breakpoints.
    amp = InterpolatedWaveform(total, [0.0, max_rabi, max_rabi, 0.0], times=times)
    det = InterpolatedWaveform(
        total, [-max_detuning, -max_detuning, max_detuning, max_detuning], times=times
    )

    # MockDevice has no hardware limits (atom number, max amplitude, minimum
    # spacing...), which is convenient for simulation. Use a real device such
    # as pulser.devices.AnalogDevice to check hardware feasibility.
    seq = pulser.Sequence(reg, MockDevice)

    # "rydberg_global" addresses all atoms with the same pulse.
    seq.declare_channel("ch", "rydberg_global")

    # Pulse(amplitude, detuning, phase): the phase is kept at 0 (no rotation
    # of the drive axis).
    seq.add(pulser.Pulse(amp, det, 0.0), "ch")
    return seq


def build_constant_sequence(reg, duration=4000, rabi=4 * np.pi, detuning=0.0):
    """Build a constant global drive (a quench) acting on every atom equally.

    The Rabi frequency and the detuning are switched on at t = 0 and kept
    constant for the whole sequence, so the atoms evolve under a fixed
    Hamiltonian starting from the ground state. This is the drive used in the
    QRC image-denoising paper (Das et al., arXiv:2512.18612).

    Args:
        reg: the pulser.Register returned by `build_register`.
        duration: length of the drive in ns.
        rabi: Rabi frequency Omega in rad/us (Pulser convention, Omega/2 sigma_x).
        detuning: global detuning delta in rad/us (Pulser convention, -delta n).

    Returns:
        A pulser.Sequence with a single "rydberg_global" channel named "ch".
    """
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch", "rydberg_global")

    # ConstantPulse(duration, amplitude, detuning, phase), with phase 0.
    seq.add(pulser.Pulse.ConstantPulse(duration, rabi, detuning, 0.0), "ch")
    return seq


def run(seq, dt=10, gpu=True):
    """Simulate the sequence and read out the final Rydberg occupation.

    Args:
        seq: the pulser.Sequence returned by `build_sequence`.
        dt: solver time step in ns. Larger values are faster (the number of
            steps scales as 1/dt) but less accurate.
        gpu: if True, store and evolve the state on a CUDA GPU when
            available; if False, run entirely on the CPU.

    Returns:
        A numpy array of length n_atoms with the probability of finding each
        atom in the Rydberg state at the end of the sequence (the
        expectation value of n_i = |r><r|_i), ordered as the register atoms.
    """
    config = SVConfig(
        # Observable measured at the end of the sequence. Times are given as a
        # fraction of the total duration (1.0 = the end), and they must be
        # divisible by dt.
        observables=[Occupation(evaluation_times=[1.0])],
        gpu=gpu,
        dt=dt,
        # Hide the per-step progress messages printed by the solver.
        log_level=logging.WARN,
    )
    results = SVBackend(seq, config=config).run()

    # results.occupation holds one entry per evaluation time; take the last
    # one. It may be a torch tensor living on the GPU, so move it to the CPU
    # before converting to numpy.
    occ = results.occupation[-1]
    return np.asarray(occ.cpu() if hasattr(occ, "cpu") else occ)


if __name__ == "__main__":
    # Quick example: 2x5 = 10 atoms.
    reg = build_register(n_rows=2, n_cols=5)
    print(run(build_sequence(reg)))
