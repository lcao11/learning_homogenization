import torch
import numpy as np
from .response import constitutive_response
from .utils import plot_evolution, check_device, save_checkpoint


class L2Loss(torch.nn.Module):  # This class is based on code obtained from M. Trautner, CMS Caltech, 2024.
    """
    This class implement a L2 loss function assuming uniform discretization.
    """

    def __init__(self, time: torch.Tensor, size_average: bool = True, reduction: bool = True) -> None:
        """
        :param time: the time points for the time integration.
        :param size_average: Whether to use the mean (True) or sum (False) of the input batch
        :param reduction: Whether to return a reduced form of the loss by averaging or summing the input batch
        """
        super(L2Loss, self).__init__()

        # Dimension and Lp-norm type are positive
        self.time = time

        self.reduction = reduction
        self.size_average = size_average

    def eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor of (samples, times, stress values)
        :return: torch.Tensor containing squared L^p(0, T;R^d) norm of x utilizing the trapezoidal integration rule.
        """
        if x.dim() == 2:
            return torch.trapz(x ** 2, x=self.time, dim=1)
        return torch.trapz(torch.norm(x, dim=2) ** 2, x=self.time, dim=1)

    def abs(self, x: torch.Tensor, y: torch.Tensor):
        """
        :param x: a torch.Tensor containing values for different measure
        :param y: a torch.Tensor containing values for different measure
        :return: the absolution difference measure
        """
        loss_list = self.eval(x - y)
        if self.reduction:
            return torch.mean(loss_list) if self.size_average else torch.sum(loss_list)
        else:
            return loss_list

    def rel(self, x: torch.Tensor, y: torch.Tensor):
        """
        :param x: a torch.Tensor containing values for different measure
        :param y: a torch.Tensor containing values for different measure
        :return: the relative difference measure
        """
        diff_list = self.eval(x - y)
        ref_list = self.eval(y)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_list / ref_list)
            else:
                return torch.sum(diff_list / ref_list)
        else:
            return diff_list / ref_list

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.forward(x, y)


class L2LossPlusPenalty(torch.nn.Module):
    """
    This class implement a L2 loss function assuming uniform discretization.
    """

    def __init__(self, time: torch.Tensor, weight: float = 1.0) -> None:
        """
        :param size_average: Whether to use the mean (True) or sum (False) of the input batch
        :param reduction: Whether to return a reduced form of the loss by averaging or summing the input batch
        """
        super(L2LossPlusPenalty, self).__init__()
        self.loss_function = L2Loss(time, size_average=True, reduction=True)
        self.weight = weight

    def __call__(self, stress_approx: torch.Tensor, stress_true: torch.Tensor,
                 internal_rate: torch.Tensor) -> torch.Tensor:
        return self.loss_function(stress_approx, stress_true) + self.weight * torch.mean(torch.norm(internal_rate, dim=1) ** 2)


def train_constitutive_model(models: list[torch.nn.Module],
                                   data: tuple[
                                       torch.utils.data.DataLoader, torch.utils.data.DataLoader],
                                   loss_function: torch.nn.Module, times: torch.Tensor, n_internal: int,
                                   output_path: str, lr: float = 1e-3, epochs: int = 1000,
                                   verbose: bool = True, rate_explicit: bool = True,
                                   grad_clip_norm: float = 1.0) -> None:
    """
    :param models: a list of models in the order of model_F (stress output) and model_G (internal variable rate output)
    :param data: a tuple of training and validation data loader
    :param loss_function: the loss function
    :param times: the torch.Tensor of time positions
    :param n_internal: the number of internal variables
    :param lr: the learning rate for Adam
    :param epochs: the total number of epochs
    :param output_path: the output path for saving results
    :param verbose: whether to print loss evolutions
    :param rate_explicit: whether to use explicit strain rate as model input
    :param grad_clip_norm: clip gradient global norm to this value (set <= 0 to disable)
    :return:
    """
    train_losses = []
    valid_losses = []

    all_params = list(models[0].parameters()) + list(models[1].parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)

    device = check_device()
    for model in models:
        model.to(device)
    
    times = times.to(device)
        
    train_data, valid_data = data
    
    best_valid_loss = float('inf')
    
    # Check if loss function requires internal rate
    requires_internal_rate = isinstance(loss_function, L2LossPlusPenalty)

    for ep in range(epochs):
        running_loss = 0.0
        for model in models:
            model.train()
            
        for ii, data_batch in enumerate(train_data):
            if rate_explicit:
                microstructure, strain, strain_rate, stress = data_batch
                microstructure, strain, strain_rate, stress = microstructure.to(device, non_blocking=True), strain.to(device, non_blocking=True), strain_rate.to(
                    device, non_blocking=True), stress.to(device, non_blocking=True)
                sr_input = strain_rate
            else:
                microstructure, strain, stress = data_batch
                microstructure, strain, stress = microstructure.to(device, non_blocking=True), strain.to(device, non_blocking=True), stress.to(device, non_blocking=True)
                sr_input = None
            
            if requires_internal_rate:
                stress_approx, internal_rate = constitutive_response(models, times, microstructure, strain, sr_input, n_internal,
                                                                return_internal=False, return_initial_rate=True)
                loss = loss_function(stress_approx, stress, internal_rate)
            else:
                stress_approx = constitutive_response(models, times, microstructure, strain, sr_input, n_internal,
                                                                  return_internal=False, return_initial_rate=False)
                loss = loss_function(stress_approx, stress)
            
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    max_norm=grad_clip_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
                
            running_loss += loss.item()
            
        train_losses.append(running_loss / len(train_data))
        scheduler.step()

        for model in models:
            model.eval()
            
        with torch.no_grad():
            running_loss = 0.0
            for _, data_batch in enumerate(valid_data):
                if rate_explicit:
                    microstructure, strain, strain_rate, stress = data_batch
                    microstructure, strain, strain_rate, stress = microstructure.to(device, non_blocking=True), strain.to(device, non_blocking=True), strain_rate.to(
                        device, non_blocking=True), stress.to(device, non_blocking=True)
                    sr_input = strain_rate
                else:
                    microstructure, strain, stress = data_batch
                    microstructure, strain, stress = microstructure.to(device, non_blocking=True), strain.to(device, non_blocking=True), stress.to(device, non_blocking=True)
                    sr_input = None
                
                if requires_internal_rate:
                    stress_approx, internal_rate = constitutive_response(models, times, microstructure, strain, sr_input, n_internal,
                                                                    return_internal=False, return_initial_rate=True)
                    loss = loss_function(stress_approx, stress, internal_rate)
                else:
                    stress_approx = constitutive_response(models, times, microstructure, strain, sr_input, n_internal,
                                                                      return_internal=False, return_initial_rate=False)
                    loss = loss_function(stress_approx, stress)
                    
                running_loss += loss.item()
            valid_losses.append(running_loss / len(valid_data))

        if ep % 5 == 0 and ep > 0 or ep == epochs - 1:
            plot_evolution(train_losses, valid_losses, output_path)
            plot_evolution(train_losses[len(train_losses) // 2:], valid_losses[len(valid_losses) // 2:],
                           output_path, suffix="half")
            np.save(output_path + "training_loss.npy", np.array(train_losses))
            np.save(output_path + "validation_loss.npy", np.array(valid_losses))
        
        # Save best model
        if valid_losses[-1] < best_valid_loss:
            best_valid_loss = valid_losses[-1]
            save_checkpoint(models, output_path + "best_model.pt", optimizer, scheduler, epoch=ep)

        if ep % 50 == 0 and ep > 0 or ep == epochs - 1:
            save_checkpoint(models, output_path + "checkpoint_ep%d.pt" % ep, optimizer, scheduler, epoch=ep)
            
        if verbose:
            print("Epoch #%d, train loss:%1.6f, valid loss:%1.6f" % (ep, train_losses[-1], valid_losses[-1]))
