import torch
import numpy as np
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import matplotlib.pyplot as plt

def plot_evolution(train_loss: list, valid_loss: list, out_path: str, suffix=None) -> None:
    """
    This function plots the training losses and validation losses
    :param train_loss: a list of training loss evolution
    :param valid_loss: a list of validation loss evolution
    :param out_path: the output path for saving
    :param suffix: the suffix for the evolution plot
    :return:
    """
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_loss, color='C0', label=r'Training')
    ax.plot(valid_loss, color='C1', label=r'Validation')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Loss')
    ax.set_yscale('log')
    plt.grid(":")
    plt.legend(frameon=False, loc='best')
    if suffix is None:
        plt.savefig(out_path + "evolution_training.pdf", bbox_inches='tight')
    else:
        plt.savefig(out_path + "evolution_training_" + suffix + ".pdf", bbox_inches='tight')
    plt.close()


def data_preprocessing(microstructure: np.ndarray, strain: np.ndarray, strain_rate: np.ndarray, stress: np.ndarray,
                       partition: list[int], normalize_inputs: bool = False, normalize_targets: bool = True) -> tuple:
    """
    Preprocesses data: splits into partitions and optionally normalizes.
    
    Args:
        microstructure, strain, strain_rate, stress: Input numpy arrays.
        partition: List of integers specifying the size of each partition (e.g. [n_train, n_val]).
        normalize_inputs: If True, applies normalization to inputs (microstructure, strain, strain_rate) in the dataset.
                          If False, returns raw inputs (useful if model handles normalization).
        normalize_targets: If True, applies normalization to targets (stress) in the dataset.
    
    Returns:
        (datasets, normalizers)
        datasets: Tuple of TensorDatasets corresponding to partitions.
        normalizers: Dictionary of DataNormalization objects {'microstructure': ..., 'strain': ..., 'stress': ...}
                     computed from the first partition (training set).
    """
    start = 0
    datasets = []
    
    # Convert all to tensors first
    t_microstructure = torch.from_numpy(microstructure).float()
    t_strain = torch.from_numpy(strain).float()
    t_stress = torch.from_numpy(stress).float()
    t_strain_rate = torch.from_numpy(strain_rate).float() if strain_rate is not None else None

    # Define slices for the first partition (training set) to compute stats
    train_slice = slice(0, partition[0])
    
    # Compute normalizers based on training data
    normalizers = {}
    
    # Microstructure Normalizer
    # Determine spatial dimensions to reduce over (batch + spatial dims)
    micro_dims = [0] + list(range(2, t_microstructure.ndim))
    norm_microstructure = DataNormalization(t_microstructure[train_slice], dim=micro_dims)
    normalizers['microstructure'] = norm_microstructure
    
    # Strain Normalizer
    norm_strain = DataNormalization(t_strain[train_slice], dim=(0, 1))
    normalizers['strain'] = norm_strain
    
    # Strain Rate Normalizer
    if t_strain_rate is not None:
        norm_rate = DataNormalization(t_strain_rate[train_slice], dim=(0, 1))
        normalizers['strain_rate'] = norm_rate
    
    # Stress Normalizer
    norm_stress = DataNormalization(t_stress[train_slice], dim=(0, 1))
    normalizers['stress'] = norm_stress

    # Create datasets
    for length in partition:
        end = start + length
        s = slice(start, end)
        
        # Prepare data for this partition
        m_part = t_microstructure[s]
        s_part = t_strain[s]
        st_part = t_stress[s]
        sr_part = t_strain_rate[s] if t_strain_rate is not None else None
        
        # Apply normalization if requested
        if normalize_inputs:
            m_part = norm_microstructure.encode(m_part)
            s_part = norm_strain.encode(s_part)
            if sr_part is not None:
                sr_part = norm_rate.encode(sr_part)
        
        if normalize_targets:
            st_part = norm_stress.encode(st_part)
            
        if sr_part is not None:
            datasets.append(torch.utils.data.TensorDataset(m_part, s_part, sr_part, st_part))
        else:
            datasets.append(torch.utils.data.TensorDataset(m_part, s_part, st_part))
            
        start = end

    return tuple(datasets), normalizers


class DataNormalization(torch.nn.Module):
    def __init__(self, x: torch.Tensor = None, dim=0, shape=None) -> None:
        """
        Args:
            x: Input tensor to compute statistics from.
            dim: Dimensions to reduce (compute mean/std over).
            shape: Input shape (tuple). Used to initialize buffers if x is not provided.
                   Required if x is None.
        """
        super(DataNormalization, self).__init__()
        self.dim = dim
        
        if x is not None:
            mean = torch.mean(x, dim=dim, keepdim=True)
            std = torch.std(x, dim=dim, keepdim=True)
            std[std == 0] = 1.0
        elif shape is not None:
            # Create dummy mean/std with correct broadcastable shape
            # We start with the full shape and set reduced dims to 1
            stat_shape = list(shape)
            dims = dim if isinstance(dim, (list, tuple)) else [dim]
            for d in dims:
                stat_shape[d] = 1
            
            mean = torch.zeros(stat_shape)
            std = torch.ones(stat_shape)
        else:
            mean = torch.tensor(0.0)
            std = torch.tensor(1.0)
            
        # Register as buffers so they are saved in state_dict and moved to device automatically
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Dynamically resize buffers to match the incoming state_dict to prevent shape mismatch errors
        for k in ['mean', 'std']:
            key = prefix + k
            if key in state_dict:
                setattr(self, k, state_dict[key].clone())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return (x * self.std) + self.mean


def check_device() -> torch.device:
    """
    This is a function for checking and creating devices for training
    :return: a torch device, either cuda, mps (i.e., Apple Silicon), or cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def _get_act(act):
    """
    https://github.com/NeuralOperator/PINO/blob/master/models/utils.py
    """
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func


class MLP(nn.Module):
    """
    Pointwise single hidden layer fully-connected neural network applied to last axis of input
    """

    def __init__(self, channels_in, channels_hid, channels_out, act='gelu'):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(channels_in, channels_hid)
        self.act = _get_act(act)
        self.fc2 = nn.Linear(channels_hid, channels_out)

    def forward(self, x):
        """
        Input shape (of x):     (..., channels_in)
        Output shape:           (..., channels_out)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


################################################################
#
# 1d helpers
#
################################################################
def resize_rfft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N) tensor, must satisfy real conjugate symmetry (not checked)
        s: (int), desired irfft output dimension >= 1
    Output
        out: (..., s//2 + 1) tensor
    """
    N = ar.shape[-1]
    s = s // 2 + 1 if s >= 1 else s // 2
    if s >= N:  # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N], out), dim=-1)
    elif s >= 1:  # truncate
        out = ar[..., :s]
    else:  # edge case
        raise ValueError("s must be greater than or equal to 1.")

    return out


def resize_fft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.ifft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Reference: https://github.com/numpy/numpy/pull/7593
    Args
        ar: (..., N) tensor
        s: (int), desired ifft output dimension >= 1
    Output
        out: (..., s) tensor
    """
    N = ar.shape[-1]
    if s >= N:  # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N // 2], out, ar[..., N // 2:]), dim=-1)
    elif s >= 2:  # truncate modes
        if s % 2:  # odd
            out = torch.cat((ar[..., :s // 2 + 1], ar[..., -s // 2 + 1:]), dim=-1)
        else:  # even
            out = torch.cat((ar[..., :s // 2], ar[..., -s // 2:]), dim=-1)
    else:  # edge case s = 1
        if s < 1:
            raise ValueError("s must be greater than or equal to 1.")
        else:
            out = ar[..., 0:1]

    return out


def get_grid1d(shape, device):
    """
    Returns a discretization of the 1D identity function on [0,1]
    """
    size_x = shape[1]
    gridx = torch.linspace(0, 1, size_x, device=device)
    gridx = gridx.reshape(1, size_x, 1).expand(shape[0], -1, -1)
    return gridx


def projector1d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is int)
    """
    if s is not None and s != x.shape[-1]:
        x = fft.irfft(resize_rfft(fft.rfft(x, norm="forward"), s), n=s, norm="forward")

    return x


################################################################
#
# 2d helpers
#
################################################################
def resize_rfft2(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft2(ar, s=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N_1, N_2) tensor, must satisfy real conjugate symmetry (not checked)
        s: (2) tuple, s=(s_1, s_2) desired irfft2 output dimension (s_i >=1)
    Output
        out: (..., s1, s_2//2 + 1) tensor
    """
    s1, s2 = s
    out = resize_rfft(ar, s2)  # last axis (rfft)
    return resize_fft(out.transpose(-2, -1), s1).transpose(-2, -1)  # second to last axis (fft)


def get_grid2d(shape, device):
    """
    Returns a discretization of the 2D identity function on [0,1]^2
    """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, device=device)
    gridx = gridx.reshape(1, size_x, 1, 1).expand(batchsize, -1, size_y, -1)
    gridy = torch.linspace(0, 1, size_y, device=device)
    gridy = gridy.reshape(1, 1, size_y, 1).expand(batchsize, size_x, -1, -1)
    return torch.cat((gridx, gridy), dim=-1)


def projector2d(x, s=None):
    """
    Either truncate or zero pad the Fourier modes of x so that x has new resolution s (s is 2 tuple)
    """
    if s is not None and tuple(s) != tuple(x.shape[-2:]):
        x = fft.irfft2(resize_rfft2(fft.rfft2(x, norm="forward"), s), s=s, norm="forward")

    return x


################################################################
#
# 1d Fourier layers
#
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, ..., nx_in)
        s:                      (int): desired spatial resolution (s,) in output space
        """
        # Original resolution
        out_ft = list(x.shape)
        out_ft[1] = self.out_channels
        xsize = out_ft[-1]

        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(*out_ft[:-1], xsize // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes1] = compl_mul(x[..., :self.modes1], self.weights1)

        # Return to physical space
        if s is None or s == xsize:
            x = fft.irfft(out_ft, n=xsize)
        else:
            x = fft.irfft(resize_rfft(out_ft, s), n=s, norm="forward") / xsize

        return x


class LinearFunctionals1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
        Inputs:
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctionals1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1 + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, ..., nx_in)
        Output shape:           (batch, out_channels, ...)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = fft.rfft(x, norm="forward")

        # Truncate input modes
        x = resize_rfft(x, 2 * self.modes1)

        # Multiply relevant Fourier modes and take the real part
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        return 2 * torch.sum(x, dim=-1) - x[..., 0]


class LinearDecoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
        """
        super(LinearDecoder1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1 + 1, dtype=torch.cfloat))

    def forward(self, x, s):
        """
        Input shape (of x):     (batch, in_channels, ...)
        s            (int):     desired spatial resolution (nx,) of functions
        Output shape:           (batch, out_channels, ..., nx)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x[..., None].type(torch.cfloat), self.weights)

        # Zero pad modes
        x = resize_rfft(x, s)

        # Return to physical space
        return fft.irfft(x, n=s)


################################################################
#
# 2d Fourier layers
#
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier integral operator layer defined for functions over the torus. Maps functions to functions.
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x, s=None):
        """
        Input shape (of x):     (batch, channels, ..., nx_in, ny_in)
        s:                      (list or tuple, length 2): desired spatial resolution (s,s) in output space
        """
        # Original resolution
        out_ft = list(x.shape)
        out_ft[1] = self.out_channels
        xsize = out_ft[-2:]

        # Compute Fourier coeffcients (un-scaled)
        x = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(*out_ft[:-2], xsize[-2], xsize[-1] // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[..., :self.modes1, :self.modes2] = \
            compl_mul(x[..., :self.modes1, :self.modes2], self.weights1)
        out_ft[..., -self.modes1:, :self.modes2] = \
            compl_mul(x[..., -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if s is None or tuple(s) == tuple(xsize):
            x = fft.irfft2(out_ft, s=tuple(xsize))
        else:
            x = fft.irfft2(resize_rfft2(out_ft, s), s=s, norm="forward") / (xsize[-2] * xsize[-1])

        return x


class LinearFunctionals2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
        Inputs:
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctionals2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, 2 * self.modes1, self.modes2 + 1,
                                    dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, ..., nx_in, ny_in)
        Output shape:           (batch, out_channels, ...)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = fft.rfft2(x, norm="forward")

        # Truncate input modes
        x = resize_rfft2(x, (2 * self.modes1, 2 * self.modes2))

        # Multiply relevant Fourier modes and take the real part
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        x = 2 * torch.sum(x[..., :self.modes1, :], dim=(-2, -1)) + \
            2 * torch.sum(x[..., -self.modes1:, 1:], dim=(-2, -1)) - x[..., 0, 0]

        return x


class LinearDecoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
        """
        super(LinearDecoder2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, 2 * self.modes1, self.modes2 + 1,
                                    dtype=torch.cfloat))

    def forward(self, x, s):
        """
        Input shape (of x):             (batch, in_channels, ...)
        s (list or tuple, length 2):    desired spatial resolution (nx,ny) of functions
        Output shape:                   (batch, out_channels, ..., nx, ny)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x[..., None, None].type(torch.cfloat), self.weights)

        # Zero pad modes
        x = resize_rfft2(x, tuple(s))

        # Return to physical space
        return fft.irfft2(x, s=s)


def save_checkpoint(models, path, optimizer=None, scheduler=None, normalizers=None, epoch=None, **kwargs):
    """
    Save checkpoint to path.
    :param models: model or list/dict of models
    :param path: path to save checkpoint
    :param optimizer: optimizer (optional)
    :param scheduler: scheduler (optional)
    :param normalizers: dictionary of data normalizers (optional)
    :param epoch: current epoch (optional)
    :param kwargs: other items to save
    """
    checkpoint = {}
    if isinstance(models, torch.nn.Module):
        checkpoint['model_state_dict'] = models.state_dict()
    elif isinstance(models, list):
        checkpoint['model_state_dict'] = [m.state_dict() for m in models]
    elif isinstance(models, dict):
        checkpoint['model_state_dict'] = {k: m.state_dict() for k, m in models.items()}

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if normalizers is not None:
        checkpoint['normalizers'] = normalizers

    if epoch is not None:
        checkpoint['epoch'] = epoch

    checkpoint.update(kwargs)

    torch.save(checkpoint, path)


def load_checkpoint(path, models, optimizer=None, scheduler=None, device=None):
    """
    Load checkpoint from path.
    :param path: path to load checkpoint from
    :param models: model or list/dict of models to load state_dict into
    :param optimizer: optimizer to load state_dict into (optional)
    :param scheduler: scheduler to load state_dict into (optional)
    :param device: device to map location to
    :return: checkpoint dictionary
    """
    if device is None:
        device = check_device()

    checkpoint = torch.load(path, map_location=device)

    if isinstance(models, torch.nn.Module):
        models.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(models, list):
        if isinstance(checkpoint['model_state_dict'], list):
            for m, state in zip(models, checkpoint['model_state_dict']):
                m.load_state_dict(state)
    elif isinstance(models, dict):
        for k, m in models.items():
            if k in checkpoint['model_state_dict']:
                m.load_state_dict(checkpoint['model_state_dict'][k])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

