import gc
import os
import sys
import argparse
import json
import pickle

import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib

_hippylib_path = os.environ.get("HIPPYLIB_PATH")
if _hippylib_path:
    sys.path.append(_hippylib_path)

try:
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=15)
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
except Exception:
    pass

# testing/evp_1d -> repo root
sys.path.insert(0, "../../")
from learning import *  # noqa


def _concat_channelwise(*arrays: np.ndarray) -> np.ndarray:
    """Concatenate arrays shaped (N, 1, n_grid) into (N, C, n_grid)."""
    return np.concatenate(arrays, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training constitutive model (evp_1d)")

    parser.add_argument(
        "--data_prefix",
        default="../../data/evp_1d/dataset/2024-12-20_Viscoplastic_PC1D_",
        type=str,
        help="The prefix for loading data files.",
    )
    parser.add_argument("--n_files", default=1, type=int, help="the number of processes to use")
    parser.add_argument("--output_path", default="./results/", type=str, help="The output path for saving.")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The learning rate.")
    parser.add_argument("--epochs", default=10, type=int, help="The number of epochs for training.")
    parser.add_argument("--n_train", default=128, type=int, help="The number of training data.")
    parser.add_argument("--n_valid", default=32, type=int, help="The number of validation data.")
    parser.add_argument("--n_test", default=32, type=int, help="The number of testing data.")
    parser.add_argument("--n_hidden", default=1, type=int, help="The number of hidden variables.")
    parser.add_argument("--n_internal", default=None, type=int, help="The number of internal variables (alias).")
    parser.add_argument("--n_channels", default=32, type=int, help="The number of channels.")
    parser.add_argument("--n_modes", default=2, type=int, help="The number of fourier modes.")
    parser.add_argument("--n_layers", default=3, type=int, help="The number of layers.")
    parser.add_argument("--batch_size", default=32, type=int, help="The batch size for optimization.")
    parser.add_argument("--downsample",default=20,type=int, help="Downsample factor for time/strain/stress trajectories (must be >= 1).",)
    parser.add_argument("--verbose", default=1, type=int, help="whether to print loss evolutions.")

    device = check_device()
    args = parser.parse_args()
    tracer = vars(args)

    if tracer["n_internal"] is None:
        tracer["n_internal"] = tracer["n_hidden"]

    if tracer["downsample"] < 1:
        raise ValueError("--downsample must be >= 1")

    output_path = (
        f"{tracer['output_path']}train{tracer['n_train']}_internal{tracer['n_internal']}"
        f"_lr{tracer['learning_rate']:.1e}_bs{tracer['batch_size']}_ep{tracer['epochs']}/"
    )
    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "settings.pkl", "wb") as f:
        pickle.dump(tracer, f)

    ym_list, rc_list, ys_list, re_list = [], [], [], []
    strain_list, stress_list = [], []
    time = None

    for ii in range(tracer["n_files"]):
        file_path = f"{tracer['data_prefix']}process{ii}_data.pkl"
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if time is None:
            time = torch.from_numpy(data["time"][:: tracer["downsample"]]).to(device, dtype=torch.float32)

        # IMPORTANT: use append(), not extend()
        ym_list.append(np.expand_dims(data["youngs_modulus"][:, ::2], axis=1))
        rc_list.append(np.expand_dims(data["rate_constant"][:, ::2], axis=1))
        ys_list.append(np.expand_dims(data["yield_stress"][:, ::2], axis=1))
        re_list.append(np.expand_dims(data["rate_exponent"][:, ::2], axis=1))

        strain_list.append(data["strain"][:, :: tracer["downsample"], :])
        stress_list.append(data["stress"][:, :: tracer["downsample"], :])

    total_n_samples = tracer["n_train"] + tracer["n_valid"] + tracer["n_test"]

    ym = np.concatenate(ym_list, axis=0)
    rc = np.concatenate(rc_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    re = np.concatenate(re_list, axis=0)
    microstructure = _concat_channelwise(ym, rc, ys, re)[:total_n_samples]

    del ym_list, rc_list, ys_list, re_list, ym, rc, ys, re
    gc.collect()

    strain = np.concatenate(strain_list, axis=0)[:total_n_samples]
    stress = np.concatenate(stress_list, axis=0)[:total_n_samples]
    del strain_list, stress_list
    gc.collect()

    if strain.shape[0] < total_n_samples:
        raise ValueError(
            f"Not enough samples loaded for requested splits: loaded={strain.shape[0]}, requested={total_n_samples}. "
            f"(n_train={tracer['n_train']}, n_valid={tracer['n_valid']}, n_test={tracer['n_test']})"
        )

    if tracer["n_test"] > 0:
        (train_ds, valid_ds, test_ds), normalizers = data_preprocessing(
            microstructure,
            strain,
            None,
            stress,
            [tracer["n_train"], tracer["n_valid"], tracer["n_test"]],
            normalize_inputs=True,
            normalize_targets=True,
        )
    else:
        (train_ds, valid_ds), normalizers = data_preprocessing(
            microstructure,
            strain,
            None,
            stress,
            [tracer["n_train"], tracer["n_valid"]],
            normalize_inputs=True,
            normalize_targets=True,
        )
        test_ds = None

    torch.save({k: v.state_dict() for k, v in normalizers.items()}, output_path + "normalizers_state.pt")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=tracer["batch_size"], shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=min(256, tracer["n_valid"]), shuffle=False)

    train_eval_dl = torch.utils.data.DataLoader(train_ds, batch_size=min(256, tracer["n_train"]), shuffle=False)
    valid_eval_dl = torch.utils.data.DataLoader(valid_ds, batch_size=min(256, tracer["n_valid"]), shuffle=False)
    test_dl = (
        torch.utils.data.DataLoader(test_ds, batch_size=min(256, tracer["n_test"]), shuffle=False)
        if test_ds is not None
        else None
    )

    input_dim_F = strain.shape[2] + tracer["n_internal"] + microstructure.shape[1]
    output_dim_F = stress.shape[2]
    input_dim_G = strain.shape[2] + tracer["n_internal"] + microstructure.shape[1]
    output_dim_G = tracer["n_internal"]

    models = [
        FNM1D(
            modes1=tracer["n_modes"],
            width=tracer["n_channels"],
            width_final=tracer["n_channels"] * 2,
            d_in=input_dim_F,
            d_out=output_dim_F,
            width_lfunc=None,
            act="gelu",
            n_layers=tracer["n_layers"],
        ),
        FNM1D(
            modes1=tracer["n_modes"],
            width=tracer["n_channels"],
            width_final=tracer["n_channels"] * 2,
            d_in=input_dim_G,
            d_out=output_dim_G,
            width_lfunc=None,
            act="gelu",
            n_layers=tracer["n_layers"],
        ),
    ]

    loss_function = L2Loss(time)

    train_constitutive_model(
        models,
        (train_dl, valid_dl),
        loss_function,
        time,
        tracer["n_internal"],
        output_path,
        lr=tracer["learning_rate"],
        epochs=tracer["epochs"],
        verbose=bool(tracer["verbose"]),
        rate_explicit=False,  # EVP is NOT rate-explicit
    )

    def _eval_loader(models_local, loader, split_name: str):
        device_local = check_device()
        for m in models_local:
            m.to(device_local)
            m.eval()

        l2_per_sample = L2Loss(time.to(device_local), reduction=False, size_average=True)

        stress_norm = normalizers.get("stress", None)
        if stress_norm is None:
            raise RuntimeError("Missing normalizers['stress']; cannot compute physical-stress errors.")
        stress_norm = stress_norm.to(device_local)

        rel_phys_losses = []

        with torch.no_grad():
            for batch in loader:
                micro, eps, sig = batch
                micro = micro.to(device_local)
                eps = eps.to(device_local)
                sig = sig.to(device_local)

                sig_pred = constitutive_response(
                    models_local,
                    time.to(device_local),
                    micro,
                    eps,
                    None,
                    tracer["n_internal"],
                    return_internal=False,
                    return_initial_rate=False,
                )

                sig_pred_phys = stress_norm.decode(sig_pred)
                sig_phys = stress_norm.decode(sig)
                rel_loss_phys = l2_per_sample(sig_pred_phys, sig_phys)  # per-sample rel LOSS
                rel_phys_losses.append(rel_loss_phys.detach().cpu().numpy())

        rel_phys_losses = np.concatenate(rel_phys_losses) if rel_phys_losses else np.array([])
        rel_phys_errors = np.sqrt(np.maximum(rel_phys_losses, 0.0)) if rel_phys_losses.size else np.array([])

        summary = {
            "split": split_name,
            "n_samples": int(rel_phys_errors.shape[0]),
            "rel_mean": float(rel_phys_errors.mean()) if rel_phys_errors.size else float("nan"),
            "rel_std": float(rel_phys_errors.std()) if rel_phys_errors.size else float("nan"),
            "rel_median": float(np.median(rel_phys_errors)) if rel_phys_errors.size else float("nan"),
            "rel_p90": float(np.quantile(rel_phys_errors, 0.90)) if rel_phys_errors.size else float("nan"),
            "rel_max": float(rel_phys_errors.max()) if rel_phys_errors.size else float("nan"),
        }

        return summary, rel_phys_losses, rel_phys_errors

    best_path = output_path + "best_model.pt"
    if os.path.exists(best_path):
        load_checkpoint(best_path, models, device=check_device())

    summaries = {}
    arrays = {}

    for name, loader in [("train", train_eval_dl), ("valid", valid_eval_dl)]:
        summary, rel_loss_arr, rel_err_arr = _eval_loader(models, loader, name)
        summaries[name] = summary
        arrays[f"{name}_rel_phys_losses"] = rel_loss_arr
        arrays[f"{name}_rel_phys_errors"] = rel_err_arr


    if test_dl is not None:
        summary, rel_loss_arr, rel_err_arr = _eval_loader(models, test_dl, "test")
        summaries["test"] = summary
        arrays["test_rel_phys_losses"] = rel_loss_arr
        arrays["test_rel_phys_errors"] = rel_err_arr

    with open(output_path + "eval_summary.json", "w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    np.savez(output_path + "eval_arrays.npz", **arrays)
