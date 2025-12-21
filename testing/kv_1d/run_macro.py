import dolfin as dl  # Import the fenics library. Make sure you can do this!
import numpy as np
import sys
import os
import argparse
import time
from pathlib import Path
import pickle

_hippylib_path = os.environ.get("HIPPYLIB_PATH")
if _hippylib_path:
    sys.path.append(_hippylib_path)

# Add project root (learning_homogenization/) to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modeling import *
from learning import *
from macro import *


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='macosimulation')
    
    parser.add_argument('--process_id',
                        default=0,
                        type=int,
                        help="The process id")
    
    parser.add_argument('--model_folder',
                        default="./results_2048/train2048_internal5_penalty1.0e+00_lr1.0e-03_bs32_ep750/",
                        type=str,
                        help="The folder to load the trained model")
    
    parser.add_argument('--data_file',
                        default="../../data/kv_1d/dataset/2024-12-10_mixture_random_field_process19_data.pkl",
                        type=str,
                        help="The data file to load")
    
    parser.add_argument('--n_microstructures',
                        default=1,
                        type=int,
                        help="The number of microstructures")

    parser.add_argument('--kernel_stride',
                        default=10,
                        type=int,
                        help="Downsample stride for the memory kernel/time series.")

    parser.add_argument('--out_dir',
                        default="./macro_outputs/",
                        type=str,
                        help="Directory to save errors/timings.")
    
    args = parser.parse_args()

    device = check_device()

    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    E_array = data["E"]
    nu_array = data["nu"]
    E_prime_array = data["E_prime"].flatten()
    nu_prime_array = data["nu_prime"].flatten()
    kernel_array = data["kernel"].squeeze()

    # Infer time discretization after downsampling
    if args.kernel_stride < 1:
        raise ValueError("--kernel_stride must be >= 1")
    if kernel_array.ndim != 2:
        raise ValueError(f"Expected kernel to squeeze to 2D (n_samples, nt+1); got shape {kernel_array.shape}")
    nt_full = kernel_array.shape[1] - 1
    if nt_full % args.kernel_stride != 0:
        raise ValueError(
            f"Kernel length (nt_full={nt_full}) not divisible by stride={args.kernel_stride}. "
            "Choose a stride that evenly divides nt_full."
        )
    nt = nt_full // args.kernel_stride

    # Load model settings and construct models with matching dimensions
    settings_path = os.path.join(args.model_folder, "settings.pkl")
    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Missing settings.pkl in model_folder: {settings_path}")
    with open(settings_path, "rb") as f:
        tracer = pickle.load(f)
    n_internal = tracer.get("n_internal", tracer.get("n_hidden", None))
    if n_internal is None:
        raise ValueError("Could not infer n_internal from settings.pkl (expected n_internal or n_hidden)")

    # Dataset is 1D strain/stress; microstructure has 2 channels (E, nu)
    strain_dim = int(data["strain"].shape[2])
    strain_rate_dim = int(data["strain_rate"].shape[2])
    stress_dim = int(data["stress"].shape[2])
    micro_channels = 2
    input_dim_F = strain_dim + strain_rate_dim + n_internal + micro_channels
    input_dim_G = strain_dim + n_internal + micro_channels
    output_dim_F = stress_dim
    output_dim_G = n_internal

    models = [
        FNM1D(modes1=4, width=32, width_final=64, d_in=input_dim_F, d_out=output_dim_F,
              width_lfunc=None, act='gelu', n_layers=3),
        FNM1D(modes1=4, width=32, width_final=64, d_in=input_dim_G, d_out=output_dim_G,
              width_lfunc=None, act='gelu', n_layers=3),
    ]
    best_model_path = os.path.join(args.model_folder, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Missing best_model.pt in model_folder: {best_model_path}")
    load_checkpoint(best_model_path, models, device=device)
    for m in models:
        m.to(device)
        m.eval()

    forcing_expr = dl.Expression("100*sin(8*pi*(x[0] + t))", t=0, degree=5)
    rno_solver = RNOViscoelasticSolver1D(n_cells=250, device=device)
    rno_solver.set_rno(models[0], models[1], n_internal=n_internal)
    kernel_solver = HomogenizedViscoelasticSolver1D(n_cells=40000)

    # Keep consistent time discretization across solvers
    for solver in (rno_solver, kernel_solver):
        solver.parameters["nt"] = nt
        solver.parameters["T"] = float(data.get("T", 1.0))

    n_microstructures = int(args.n_microstructures)
    start = int(args.process_id) * n_microstructures
    end = (int(args.process_id) + 1) * n_microstructures
    n_total = int(E_array.shape[0])
    if start >= n_total:
        raise ValueError(f"start index {start} exceeds available samples {n_total}")
    end = min(end, n_total)
    n_local = end - start

    periods_list = [5, 10, 20, 40, 80]
    error_macro = np.zeros((n_local, len(periods_list)))
    error_no_memory = np.zeros(n_local)
    error_rno = np.zeros(n_local)

    # Timings (seconds)
    t_homogenized = np.zeros(n_local)
    t_no_memory = np.zeros(n_local)
    t_rno = np.zeros(n_local)
    t_macro = np.zeros((n_local, len(periods_list)))
    t_macro_assemble = np.zeros((n_local, len(periods_list)))

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = os.path.join(args.out_dir, f"process{args.process_id}")

    def _time_call(fn):
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0
    

    for ii, micro_id in enumerate(np.arange(start, end)):
        print(f"Microstructure {micro_id} ({ii+1}/{n_local})")
        sys.stdout.flush()

        kernel_down = kernel_array[micro_id, :: args.kernel_stride]
        if kernel_down.shape[0] != nt + 1:
            raise RuntimeError(
                f"Downsampled kernel length mismatch: got {kernel_down.shape[0]} expected {nt+1}"
            )

        kernel_solver.set_microstructure(E_prime_array[micro_id], nu_prime_array[micro_id], kernel_down)
        u_list_homogenized, t_homogenized[ii] = _time_call(lambda: kernel_solver.solve(forcing_expr))

        kernel_solver.set_microstructure(E_prime_array[micro_id], nu_prime_array[micro_id], np.zeros_like(kernel_down))
        u_list_no_memory, t_no_memory[ii] = _time_call(lambda: kernel_solver.solve(forcing_expr))
        error_no_memory[ii] = compute_error(u_list_homogenized, u_list_no_memory, kernel_solver._Vh, kernel_solver._Vh)

        for jj, n_periods in enumerate(periods_list):
            macro_solver = MacroViscoelasticSolver1D(n_periods=n_periods, cells_per_periods=500)
            macro_solver.parameters["nt"] = nt
            macro_solver.parameters["T"] = kernel_solver.parameters["T"]
            _, t_macro_assemble[ii, jj] = _time_call(lambda: macro_solver.set_microstructure(E_array[micro_id], nu_array[micro_id]))
            u_list_macro, t_macro[ii, jj] = _time_call(lambda: macro_solver.solve(forcing_expr))
            error_macro[ii, jj] = compute_error(u_list_homogenized, u_list_macro, kernel_solver._Vh, macro_solver._Vh)

        rno_solver.set_microstructure(E_array[micro_id], nu_array[micro_id])
        u_list_rno, t_rno[ii] = _time_call(lambda: rno_solver.solve(forcing_expr))
        error_rno[ii] = compute_error(u_list_homogenized, u_list_rno, kernel_solver._Vh, rno_solver._Vh)

        # Save incrementally (robust to preemption)
        np.savez(
            out_prefix + "_errors_timings.npz",
            start=start,
            end=end,
            periods=np.array(periods_list, dtype=int),
            kernel_stride=int(args.kernel_stride),
            nt=int(nt),
            T=float(kernel_solver.parameters["T"]),
            error_macro=error_macro,
            error_no_memory=error_no_memory,
            error_rno=error_rno,
            t_homogenized=t_homogenized,
            t_no_memory=t_no_memory,
            t_rno=t_rno,
            t_macro=t_macro,
            t_macro_assemble=t_macro_assemble,
        )

    print(f"Done. Saved: {out_prefix + '_errors_timings.npz'}")