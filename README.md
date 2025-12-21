# learning_homogenization

Learn memory- and microstructure-dependnet constitutive models from cell-problem simulations (FEniCS/hIPPYlib), then use neural constitutive surrogates (PyTorch) for fast evaluation and (for KV) macro-scale comparisons.

This repo currently contains two main 1D problem families:

- KV 1D (Kelvin–Voigt viscoelasticity): generates strain/strain-rate/stress trajectories and (optionally) a memory-kernel representation; trains a recurrent neural operator-style surrogate; includes a macroscale benchmark comparing homogenized-memory vs multiscale vs neural surrogate.
- EVP 1D (elasto-viscoplasticity): generates strain/stress trajectories for random microstructures; trains a surrogate.

## Quickstart

### 1) Environment

You’ll typically need two stacks:

- Simulation/data generation: FEniCS (`dolfin`/`ufl`) + `hippylib` + MPI
- Learning/training: PyTorch + NumPy/SciPy + Matplotlib

Some scripts (especially under `data/` and `testing/kv_1d/macro.py`) import `dolfin` and `hippylib`, so make sure those are available.

Most scripts expect `hippylib` to be discoverable via:

```bash
export HIPPYLIB_PATH=/path/to/hippylib
```

## Dependencies

### Minimal (training-only, pip-installable)

For running the learning/training scripts under `testing/*/run_training.py` (without data generation / macro solvers):

```bash
python3 -m pip install -r requirements.txt
```

See [requirements.txt](requirements.txt).

### External (simulation/macro)

Data generation (`data/`) and the KV macro benchmark (`testing/kv_1d/run_macro.py`) require FEniCS (`dolfin`/`ufl`) and `hippylib` (typically not installed via `pip`).

A conda installation of FEniCS 2019.1.0 is recommended.

```bash
conda create -n fenics-2019.1 -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter
```

### 2) Generate datasets

Datasets are written as pickles with a filename pattern:

`<output_path>/<YYYY-MM-DD>_<TYPE>_process<process_id>_data.pkl`

Training scripts consume data via a `--data_prefix` that ends at `..._<TYPE>_` and then append `process<i>_data.pkl`.

#### KV 1D datasets

- Piecewise-constant (PC) microstructures: [data/kv_1d/generate_pc.py](data/kv_1d/generate_pc.py)
- High memory (mixture) random field microstructures: [data/kv_1d/generate_random.py](data/kv_1d/generate_random.py)

Example (from repo root):

```bash
mpirun -n 1 python3 data/kv_1d/generate_pc.py \
	--process_id 0 --seed 0 --n_samples 1000 \
	--output_path ./data/kv_1d/dataset/ \
	--file_name data.pkl
```

This produces something like:

`data/kv_1d/dataset/20xx-xx-xx_PC1D_process0_data.pkl`

#### EVP 1D datasets

- Piecewise-constant (PC) microstructures: [data/evp_1d/generate_pc.py](data/evp_1d/generate_pc.py)
- Random-field microstructures: [data/evp_1d/generate_random.py](data/evp_1d/generate_random.py)

Example:

```bash
mpirun -n 1 python3 data/evp_1d/generate_pc.py \
	--process_id 0 --seed 0 --n_samples 1000 \
	--output_path ./data/evp_1d/dataset/ \
	--file_name data.pkl
```

### 3) Train a surrogate constitutive model

The training entry points live in `testing/`:

- KV 1D training: [testing/kv_1d/run_training.py](testing/kv_1d/run_training.py)
- EVP 1D training: [testing/evp_1d/run_training.py](testing/evp_1d/run_training.py)

Both scripts:

- load one or more dataset pickles (`process0`, `process1`, …)
- build two neural models: a stress model `F` and an internal-rate model `G` (Fourier Neural Functionals)
- train via [learning/training.py](learning/training.py)
- save checkpoints + plots + evaluation summaries

#### KV 1D training example

From `testing/kv_1d/`:

```bash
python3 -u run_training.py \
	--data_prefix "../../data/kv_1d/dataset/2025-12-21_PC1D_" \
	--n_files 1 \
	--n_train 128 --n_valid 32 --n_test 32 \
	--n_internal 5 \
	--epochs 50 \
	--batch_size 32 \
	--downsample 10 \
	--penalty_weight 1.0 \
	--output_path ./results/
```

Notes:

- KV is rate-explicit (the model uses strain rate as an input).
- If your dataset file names don’t match the default prefix in the script, override `--data_prefix`.

Slurm helper: [testing/kv_1d/submit_training.sh](testing/kv_1d/submit_training.sh)

#### EVP 1D training example

From `testing/evp_1d/`:

```bash
python3 -u run_training.py \
	--data_prefix "../../data/evp_1d/dataset/2025-12-21_ElastoViscoplastic_PC1D_" \
	--n_files 1 \
	--n_train 128 --n_valid 32 --n_test 32 \
	--n_internal 1 \
	--epochs 50 \
	--batch_size 32 \
	--downsample 20 \
	--output_path ./results/
```

Notes:

- EVP is not rate-explicit (training passes `rate_explicit=False`).

### 4) KV macro benchmark (optional)

KV includes a macro-scale benchmark comparing:

- homogenized viscoelastic solver using an explicit memory kernel,
- periodic macro solver (many periods),
- neural surrogate (RNO-style solver wrapping `F`/`G`).

Entry points:

- Macro solvers: [testing/kv_1d/macro.py](testing/kv_1d/macro.py)
- Runner: [testing/kv_1d/run_macro.py](testing/kv_1d/run_macro.py)

Example (from `testing/kv_1d/`):

```bash
mpirun -n 1 python3 -u run_macro.py \
	--process_id 0 \
	--model_folder ./results_2048/train2048_internal5_penalty1.0e+00_lr1.0e-03_bs32_ep750/ \
	--data_file ../../data/kv_1d/dataset/2025-12-21_mixture_random_field_process0_data.pkl \
	--n_microstructures 16 \
	--kernel_stride 10 \
	--out_dir ./macro_outputs/
```

This writes `macro_outputs/process<id>_errors_timings.npz`.

## Outputs

Training runs create a timestamped folder under the chosen `--output_path`, e.g.

`testing/kv_1d/results_128/train128_internal5_penalty1.0e+00_lr1.0e-03_bs32_ep750/`

Output files:

- `settings.pkl`: CLI args used for the run
- `best_model.pt`: best checkpoint by validation loss
- `checkpoint_ep*.pt`: periodic checkpoints
- `training_loss.npy`, `validation_loss.npy`: loss curves
- `eval_summary.json`: summary statistics on train/valid/(test)
- `eval_arrays.npz`: per-sample arrays (relative losses/errors, and penalty values for KV)
- loss plots saved by `learning.utils.plot_evolution()`

## Repo Map

- `data/`: dataset generation (FEniCS + hippylib)
	- `data/kv_1d/`: KV 1D generators
	- `data/evp_1d/`: EVP 1D generators
- `modeling/`: PDE/cell-problem solvers, microstructure models, trajectory generators
- `learning/`: PyTorch models + training utilities
	- `learning/response.py`: time-marching “constitutive response” wrapper used during training
	- `learning/fnm.py`: Fourier Neural Functionals (1D/2D)
	- `learning/training.py`: training loops + losses
- `testing/`: runnable scripts for training and KV macro evaluation
- `tutorial/`: notebook for 1d cell problem demo