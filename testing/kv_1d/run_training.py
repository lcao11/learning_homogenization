import gc
import torch
import os, sys, argparse
_hippylib_path = os.environ.get('HIPPYLIB_PATH')
if _hippylib_path:
    sys.path.append(_hippylib_path)
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib
try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=15)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
except:
    pass
sys.path.insert(0, '../../')
from learning import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training constitutive model')

    parser.add_argument('--data_prefix',
                        default="../../data/kv_1d/dataset/2024-10-13_PC1D_",
                        type=str,
                        help="The prefix for loading data files.")

    parser.add_argument('--n_files',
                        default=10,
                        type=int,
                        help="the number of processes to use")

    parser.add_argument('--output_path',
                        default="./results/",
                        type=str,
                        help="The output path for saving.")

    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help="The breath of each layer.")

    parser.add_argument('--epochs',
                        default=25,
                        type=int,
                        help="The number of epochs for training.")

    parser.add_argument('--n_train',
                        default=128,
                        type=int,
                        help="The number of training data.")

    parser.add_argument('--n_valid',
                        default=32,
                        type=int,
                        help="The number of validation data.")

    parser.add_argument('--n_test',
                        default=32,
                        type=int,
                        help="The number of testing data (held-out).")

    parser.add_argument('--n_hidden',
                        default=5,
                        type=int,
                        help="DEPRECATED: use --n_internal. Kept for backward compatibility.")

    parser.add_argument('--n_internal',
                        default=None,
                        type=int,
                        help="The number of internal variables.")

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help="The batch size for optimization.")
    
    parser.add_argument('--penalty_weight',
                        default=1.0,
                        type=float,
                        help="The penalty weight for internal variable initial rate.")

    parser.add_argument('--downsample',
                        default=20,
                        type=int,
                        help="Downsample factor for time/strain/stress trajectories (must be >= 1).")

    parser.add_argument('--verbose',
                        default=1,
                        type=int,
                        help="whether to print loss evolutions.")

    device = check_device()
    args = parser.parse_args()
    tracer = vars(args)  # store the command line argument values into the dictionary

    # Backward-compatible rename: hidden variables -> internal variables
    if tracer["n_internal"] is None:
        tracer["n_internal"] = tracer["n_hidden"]

    if tracer["downsample"] < 1:
        raise ValueError("--downsample must be >= 1")

    output_path = (
        f"{tracer['output_path']}train{tracer['n_train']}_internal{tracer['n_internal']}"
        f"_penalty{tracer['penalty_weight']:.1e}_lr{tracer['learning_rate']:.1e}"
        f"_bs{tracer['batch_size']}_ep{tracer['epochs']}/"
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    with open(output_path + "settings.pkl", 'wb') as f:
        pickle.dump(tracer, f)

    E, nu = [], []
    strain, strain_rate, stress = [], [], []
    time = None

    total_n_samples = tracer["n_train"] + tracer["n_valid"] + tracer["n_test"]
    missing = []
    for ii in range(tracer["n_files"]):
        file_path = f"{tracer['data_prefix']}process{ii}_data.pkl"
        if not os.path.exists(file_path):
            missing.append(file_path)
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if time is None:
                time = torch.from_numpy(data["time"][::tracer["downsample"]]).to(device, dtype=torch.float32)
            # Keep per-file arrays intact; extend() would iterate sample-wise and break shapes.
            E.append(np.expand_dims(data["E"][:, ::2], axis=1))
            nu.append(np.expand_dims(data["nu"][:, ::2], axis=1))
            strain.append(data["strain"][:, ::tracer["downsample"], :])
            strain_rate.append(data["strain_rate"][:, ::tracer["downsample"], :])
            stress.append(data["stress"][:, ::tracer["downsample"], :])

    # Microstructure tensor should be (N, channels, n_cells_on_grid).
    # E and nu are already shaped (N, 1, n_grid); concatenate along channel axis.
    microstructure = np.concatenate((np.concatenate(E, axis=0), np.concatenate(nu, axis=0)), axis=1)[:total_n_samples]
    del E, nu
    gc.collect()

    strain = np.concatenate(strain, axis=0)[:total_n_samples]
    strain_rate = np.concatenate(strain_rate, axis=0)[:total_n_samples]
    stress = np.concatenate(stress, axis=0)[:total_n_samples]

    if strain.shape[0] < total_n_samples:
        raise ValueError(
            f"Not enough samples loaded for requested splits: loaded={strain.shape[0]}, requested={total_n_samples}. "
            f"(n_train={tracer['n_train']}, n_valid={tracer['n_valid']}, n_test={tracer['n_test']})"
        )

    (train_ds, valid_ds, test_ds), normalizers = data_preprocessing(
        microstructure, strain, strain_rate, stress,
        [tracer["n_train"], tracer["n_valid"], tracer["n_test"]]
    )

    # Save normalizers so predictions can be decoded later.
    torch.save({k: v.state_dict() for k, v in normalizers.items()}, output_path + "normalizers_state.pt")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=tracer["batch_size"], shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=min(256, tracer["n_valid"]), shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=min(256, tracer["n_test"]), shuffle=False)

    # For evaluation statistics, use non-shuffled loaders.
    train_eval_dl = torch.utils.data.DataLoader(train_ds, batch_size=min(256, tracer["n_train"]), shuffle=False)
    valid_eval_dl = torch.utils.data.DataLoader(valid_ds, batch_size=min(256, tracer["n_valid"]), shuffle=False)

    input_dim_F = strain.shape[2] + strain_rate.shape[2] + tracer["n_internal"] + microstructure.shape[1]
    output_dim_F = stress.shape[2]
    input_dim_G = strain.shape[2] + tracer["n_internal"] + microstructure.shape[1]
    output_dim_G = tracer["n_internal"]

    models = [FNM1D(modes1=4,
                    width=32,
                    width_final=64,
                    d_in=input_dim_F,
                    d_out=output_dim_F,
                    width_lfunc=None,
                    act='gelu',
                    n_layers=3),
              FNM1D(modes1=4,
                    width=32,
                    width_final=64,
                    d_in=input_dim_G,
                    d_out=output_dim_G,
                    width_lfunc=None,
                    act='gelu',
                    n_layers=3)]

    loss_function = L2LossPlusPenalty(time, weight=tracer["penalty_weight"])

    train_constitutive_model(models, (train_dl, valid_dl), loss_function, time, tracer["n_internal"], output_path,
                                   lr=tracer["learning_rate"],
                                   epochs=tracer["epochs"], verbose=bool(tracer["verbose"]), rate_explicit=True)

    def _eval_loader(models_local, loader, split_name: str):
        device_local = check_device()
        for m in models_local:
            m.to(device_local)
            m.eval()

        # Physical-space error statistics by decoding the stress normalizer
        l2_per_sample = L2Loss(time.to(device_local), reduction=False, size_average=True)
        l2_reduced = L2Loss(time.to(device_local), reduction=True, size_average=True)

        stress_norm = normalizers.get("stress", None)
        if stress_norm is None:
            raise RuntimeError(
                "Stress normalizer is missing; cannot compute physical-space error. "
                "Ensure data_preprocessing returned normalizers['stress']."
            )
        stress_norm = stress_norm.to(device_local)

        needs_rate = isinstance(loss_function, L2LossPlusPenalty)

        rel_phys_losses = []
        penalty_vals = []
        total_losses = []

        with torch.no_grad():
            for batch in loader:
                micro, eps, eps_rate, sig = batch
                micro = micro.to(device_local)
                eps = eps.to(device_local)
                eps_rate = eps_rate.to(device_local)
                sig = sig.to(device_local)

                if needs_rate:
                    sig_pred, internal_rate0 = constitutive_response(
                        models_local, time.to(device_local), micro, eps, eps_rate, tracer["n_internal"],
                        return_internal=False, return_initial_rate=True
                    )
                    pen = (torch.norm(internal_rate0, dim=1) ** 2).detach().cpu().numpy()
                    penalty_vals.append(pen)
                    tot = (l2_reduced(sig_pred, sig) + tracer["penalty_weight"] * torch.mean(torch.norm(internal_rate0, dim=1) ** 2))
                    total_losses.append(float(tot.detach().cpu().item()))
                else:
                    sig_pred = constitutive_response(
                        models_local, time.to(device_local), micro, eps, eps_rate, tracer["n_internal"],
                        return_internal=False, return_initial_rate=False
                    )
                    tot = l2_reduced(sig_pred, sig)
                    total_losses.append(float(tot.detach().cpu().item()))

                # Physical-space rel error (decoded)
                sig_pred_phys = stress_norm.decode(sig_pred)
                sig_phys = stress_norm.decode(sig)
                rel_phys = l2_per_sample(sig_pred_phys, sig_phys)
                rel_phys_losses.append(rel_phys.detach().cpu().numpy())

        rel_phys_losses = np.concatenate(rel_phys_losses) if rel_phys_losses else np.array([])
        penalty_vals = np.concatenate(penalty_vals) if penalty_vals else np.array([])
        total_losses = np.array(total_losses, dtype=float)

        # Convert per-sample relative loss to per-sample relative error.
        # L2Loss(rel) returns ||e||^2 / ||y||^2 (per-sample), so relative error is sqrt(rel_loss).
        rel_phys_errors = np.sqrt(np.maximum(rel_phys_losses, 0.0)) if rel_phys_losses.size else np.array([])

        summary = {
            "split": split_name,
            "n_samples": int(rel_phys_errors.shape[0]),
            "rel_mean": float(rel_phys_errors.mean()) if rel_phys_errors.size else float('nan'),
            "rel_std": float(rel_phys_errors.std()) if rel_phys_errors.size else float('nan'),
            "rel_median": float(np.median(rel_phys_errors)) if rel_phys_errors.size else float('nan'),
            "rel_p90": float(np.quantile(rel_phys_errors, 0.90)) if rel_phys_errors.size else float('nan'),
            "rel_max": float(rel_phys_errors.max()) if rel_phys_errors.size else float('nan'),
        }
        if penalty_vals.size:
            summary.update({
                "penalty_mean": float(penalty_vals.mean()),
                "penalty_std": float(penalty_vals.std()),
                "total_loss_mean_over_batches": float(total_losses.mean()) if total_losses.size else float('nan'),
            })

        return summary, rel_phys_losses, rel_phys_errors, penalty_vals

    # Evaluate best checkpoint on train/valid/test and save statistics.
    best_path = output_path + "best_model.pt"
    if os.path.exists(best_path):
        load_checkpoint(best_path, models, device=check_device())

    summaries = {}
    arrays = {}
    for name, loader in [("train", train_eval_dl), ("valid", valid_eval_dl), ("test", test_dl)]:
        summary, rel_phys_loss_arr, rel_phys_err_arr, pen_arr = _eval_loader(models, loader, name)
        summaries[name] = summary
        arrays[f"{name}_rel_phys_losses"] = rel_phys_loss_arr
        arrays[f"{name}_rel_phys_errors"] = rel_phys_err_arr
        if pen_arr.size:
            arrays[f"{name}_penalty_vals"] = pen_arr

    with open(output_path + "eval_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    np.savez(output_path + "eval_arrays.npz", **arrays)