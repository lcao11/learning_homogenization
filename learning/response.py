import torch
from .utils import check_device

def constitutive_response(models: list[torch.nn.Module], times: torch.Tensor, microstructure: torch.Tensor,
                         strain: torch.Tensor,
                         strain_rate: torch.Tensor,
                         n_internal: int, return_internal: bool = False, return_initial_rate: bool = False) -> torch.Tensor:
    """
    Predict stress and internal variable trajectory using neural network models.
    
    Args:
        models: List of [stress_model, internal_rate_model].
        times: Time points for evaluation.
        microstructure: Material microstructure.
        strain: Strain trajectory.
        strain_rate: Strain rate trajectory (optional).
        n_internal: Number of internal variables.
        return_internal: If True, return internal variable trajectory.
        return_initial_rate: If True, return initial rate of internal variable.
        
    Returns:
        Stress evolution, and optionally internal variable evolution or initial rate.
    """
    device = check_device()
    batch_size = strain.shape[0]
    n_steps = len(times)
    
    # Initialize internal state
    internal = torch.zeros(batch_size, n_internal).to(device)
    
    # Prepare outputs
    stress_traj = torch.zeros_like(strain).to(device)
    if return_internal:
        internal_traj = torch.zeros(batch_size, n_steps, n_internal).to(device)
    
    dt = times[1:] - times[:-1]
    
    model_F, model_G = models[0], models[1]
    
    internal_rate_initial = None

    for ii, time in enumerate(times):
        # Current inputs
        s_curr = strain[:, ii]
        sr_curr = strain_rate[:, ii] if strain_rate is not None else None
        
        # Predict stress
        if sr_curr is not None:
            stress_curr = model_F(microstructure, s_curr, sr_curr, internal)
        else:
            stress_curr = model_F(microstructure, s_curr, internal)
            
        if strain.dim() == 3: # (batch, time, dim)
             stress_traj[:, ii, :] = stress_curr
        else:
             stress_traj[:, ii] = stress_curr

        if return_internal:
            internal_traj[:, ii, :] = internal

        # Update internal state for next step
        if ii < n_steps - 1:
            internal_rate = model_G(microstructure, s_curr, internal)
            
            if ii == 0 and return_initial_rate:
                internal_rate_initial = internal_rate
                
            internal = internal + internal_rate * dt[ii]

    if return_internal:
        return stress_traj, internal_traj
    elif return_initial_rate:
        return stress_traj, internal_rate_initial
    else:
        return stress_traj
