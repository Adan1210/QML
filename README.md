# QML: Quantum Reservoir Computing with neutral atoms

Experiments in quantum machine learning on simulated Pasqal neutral-atom hardware. The main notebook implements **quantum reservoir computing (QRC)** for time-series prediction with a realistic hardware noise model.

## How it works

Quantum reservoir computing uses a fixed quantum system as a nonlinear feature map. Only a linear readout is trained.

1. **Encode:** for each time step t, take a window of the last inputs (x[t−1], x[t]). Each value sets the detuning of one laser pulse applied to a small register of Rydberg atoms.
2. **Evolve:** the pulses drive the atoms back to back. The atoms interact through the Rydberg (van der Waals) interaction, so the final state depends nonlinearly on the whole window.
3. **Measure:** sample bitstrings and turn them into features: the occupation of each atom ⟨nᵢ⟩ and the pair correlations ⟨nᵢnⱼ⟩.
4. **Read out:** fit a ridge regression from the features to the target y[t]. This is the only trained part.

The simulation includes the noise of Pasqal's FC1 device: state-preparation and readout errors, laser amplitude and detuning fluctuations, the Gaussian beam profile, thermal motion of the atoms, relaxation and dephasing (Table II of [Dalyac et al.](https://arxiv.org/pdf/2607.28364#page=14)).

## Repository contents

| File | Description |
|---|---|
| `qrc_pasqal_simple.ipynb` | Main notebook: end-to-end noisy QRC, one step per cell |
| `requirements.txt` | Pinned Python dependencies (pip) |
| `environment.yml` | Conda environment that installs `requirements.txt` |
| `data/` | Your input datasets (ignored by git, only its README is tracked) |
| `.gitignore` | Ignores data files, Python caches, virtual environments and OS files |
| `LICENSE` | MIT license |

## Setup

The notebook was tested with Python 3.13 and Pulser 1.10.0.

### Option A: conda

```bash
conda env create -f environment.yml   # creates an environment called "qml"
conda activate qml
python -m ipykernel install --user --name qml --display-name qml   # makes it selectable as a Jupyter kernel
```

To sync an existing environment after `requirements.txt` changes:

```bash
conda env update -f environment.yml --prune
```

### Option B: pip and venv

```bash
python3.13 -m venv .venv
source .venv/bin/activate              # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name qml --display-name qml
```

## Running the notebook

Open `qrc_pasqal_simple.ipynb`, select the `qml` kernel and run all cells in order. The demo uses 48 synthetic data points (a one-step memory task, y[t] = x[t−1]) and takes about 10–15 s on a laptop.

Expected output with the default settings:

| Model | MAE | NMSE |
|---|---|---|
| Noisy QRC | 0.080 | 0.197 |
| Ridge on inputs (baseline) | 7.8·10⁻⁵ | ≈ 0 |

The baseline sees x[t−1] directly, so it is almost perfect on this task; it checks the alignment of windows and targets. The QRC result shows how much of the input survives the encoding, the noise and finite sampling.

### Using your own data

Put your CSV in `data/` and, in section 1 of the notebook, set `CSV_PATH = "data/my_data.csv"` and the column names `X_COLUMN` and `Y_COLUMN`, or assign your own NumPy arrays to `x_raw` and `y_raw`. The rows must be in chronological order.

### Main parameters

| Parameter | Default | Meaning |
|---|---|---|
| `WINDOW` | 2 | Inputs per window = pulses per sequence (reservoir memory) |
| `Register.rectangle(1, 3, spacing=7.0)` | 3 atoms, 7 µm | Geometry; sets the interaction strength between atoms |
| `TAU_NS` | 500 | Duration of each pulse, in ns |
| `OMEGA` | 2π rad/µs | Rabi frequency of the global laser |
| `DELTA_MIN`, `DELTA_MAX` | −2π·0.5, 2π·1.5 rad/µs | Detuning range that the inputs are mapped to |
| `N_TRAJECTORIES`, `SHOTS_PER_TRAJECTORY` | 12, 40 | Noise realisations and shots per window |
| `RIDGE_ALPHA` | 1e-3 | Regularisation of the readout |

### Scaling to more atoms

The notebook uses `solver="MasterEquation"`, which stores the full density matrix: 4ᴺ complex entries for N atoms. That is fine up to about 10–12 atoms but does not fit in memory for 16 (about 69 GB). For larger registers, switch to `solver="MonteCarlo"` (quantum trajectories on state vectors) or to Pasqal's GPU emulators [emu-sv and emu-mps](https://docs.pasqal.com/emulationtools/emusv).

## Managing dependencies

`requirements.txt` lists only the direct dependencies, pinned. `environment.yml` reuses it, so there is a single list to maintain.

When you add a package, install it in the environment and add it to `requirements.txt` with its version:

```bash
pip install some-package
pip show some-package | grep Version   # copy the version into requirements.txt
```

To capture the full environment exactly, including every transitive dependency (useful for archiving a result, less portable across operating systems):

```bash
pip freeze > requirements-lock.txt                          # every pip package, pinned
conda env export -n qml --no-builds > environment-lock.yml  # full conda environment
conda env export -n qml --from-history                      # only what you explicitly installed with conda
```

## References

- Pulser documentation: https://pulser.readthedocs.io
- Pulser 1.10.0 source, dephasing convention: https://github.com/pasqal-io/Pulser/blob/v1.10.0/pulser-core/pulser/_hamiltonian_data/hamiltonian_data.py#L685-L700
- Dalyac et al., FC1 noise parameters (Table II): https://arxiv.org/pdf/2607.28364

## License

MIT, see [LICENSE](LICENSE).
