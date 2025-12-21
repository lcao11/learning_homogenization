import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .utils import _get_act

class FNM2D(torch.nn.Module):
    """
    Wrapper for FNF2d to handle multiple inputs by concatenation.
    """
    def __init__(self, **kargs) -> None:
        super(FNM2D, self).__init__()
        self._FNM = FNF2d(**kargs)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        args[0] is material microstructure (batch, channels, x, y).
        args[1:] are other inputs, which are expanded to match spatial dimensions if needed.
        """
        material = args[0]
        size_x, size_y = material.size(-2), material.size(-1)

        concatenated_input = torch.cat([material] + [
            arg if arg.dim() == 4 else arg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, size_x, size_y)
            for arg in args[1:]
        ], dim=1)

        return self._FNM(concatenated_input)


class FNM1D(torch.nn.Module):
    """
    Wrapper for FNF1d to handle multiple inputs by concatenation.
    """
    def __init__(self, **kargs) -> None:
        super(FNM1D, self).__init__()
        self._FNM = FNF1d(**kargs)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        args[0] is material microstructure (batch, channels, x).
        args[1:] are other inputs, which are expanded to match spatial dimensions if needed.
        """
        material = args[0]
        size = material.size(-1)

        concatenated_input = torch.cat([material] + [
            arg if arg.dim() == 3 else arg.unsqueeze(-1).expand(-1, -1, size)
            for arg in args[1:]
        ], dim=1)

        return self._FNM(concatenated_input)


class FNF2d(nn.Module):
    """
    Fourier Neural Functionals for mapping functions to finite-dimensional vectors
    """

    def __init__(self,
                 modes1=12,
                 modes2=12,
                 width=32,
                 width_final=128,
                 d_in=1,
                 d_out=1,
                 width_lfunc=None,
                 act='gelu',
                 n_layers=4
                 ):
        """
        modes1, modes2  (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_final     (int): width of the final projection layer
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): finite number of desired outputs (number of functionals)
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        """
        super(FNF2d, self).__init__()

        self.d_physical = 2
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_final = width_final
        self.d_in = d_in
        self.d_out = d_out
        if width_lfunc is None:
            self.width_lfunc = self.width
        else:
            self.width_lfunc = width_lfunc
        self.act = _get_act(act)
        self.n_layers = n_layers
        if self.n_layers is None:
            self.n_layers = 4

        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)

        self.speconvs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.n_layers - 1)]
        )

        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.n_layers - 1)]
        )

        self.lfunc0 = LinearFunctionals2d(self.width, self.width_lfunc, self.modes1, self.modes2)
        self.mlpfunc0 = MLP(self.width, self.width_final, self.width_lfunc, act)

        # Expand the hidden dim by 2 because the input is also twice as large
        self.mlp0 = MLP(2 * self.width_lfunc, 2 * self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels_in, nx_in, ny_in)
        Output shape:           (batch, self.d_out)

        The input resolution is determined by x.shape[-2:]
        """
        # Lifting layer
        x = x.permute(0, 2, 3, 1)
        x = torch.cat((x, get_grid2d(x.shape, x.device)), dim=-1)  # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Fourier integral operator layers on the torus
        for speconv, w in zip(self.speconvs, self.ws):
            x = w(x) + speconv(x)
            x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x_temp = self.lfunc0(x)

        # Retain the truncated modes (use all modes)
        x = x.permute(0, 2, 3, 1)
        x = self.mlpfunc0(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.trapz(x, dx=1. / x.shape[-1])
        x = torch.trapz(x, dx=1. / x.shape[-1])

        # Combine nonlocal and local features
        x = torch.cat((x_temp, x), dim=1)

        # Final projection layer
        x = self.mlp0(x)

        return x

class FNF1d(nn.Module):
    """
    Fourier Neural Functionals for mapping functions to finite-dimensional vectors
    """

    def __init__(self,
                 modes1=16,
                 width=64,
                 width_final=128,
                 d_in=1,
                 d_out=1,
                 width_lfunc=None,
                 act='gelu',
                 n_layers=4
                 ):
        """
        modes1          (int): Fourier mode truncation levels
        width           (int): constant dimension of channel space
        width_final     (int): width of the final projection layer
        d_in            (int): number of input channels (NOT including grid input features)
        d_out           (int): finite number of desired outputs (number of functionals)
        width_lfunc     (int): number of intermediate linear functionals to extract in FNF layer
        act             (str): Activation function = tanh, relu, gelu, elu, or leakyrelu
        n_layers        (int): Number of Fourier Layers, by default 4
        """
        super(FNF1d, self).__init__()

        self.d_physical = 1
        self.modes1 = modes1
        self.width = width
        self.width_final = width_final
        self.d_in = d_in
        self.d_out = d_out
        if width_lfunc is None:
            self.width_lfunc = self.width
        else:
            self.width_lfunc = width_lfunc
        self.act = _get_act(act)
        self.n_layers = n_layers
        if self.n_layers is None:
            self.n_layers = 4

        self.fc0 = nn.Linear(self.d_in + self.d_physical, self.width)

        self.speconvs = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes1)
            for _ in range(self.n_layers - 1)]
        )

        self.ws = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
            for _ in range(self.n_layers - 1)]
        )

        self.lfunc0 = LinearFunctionals1d(self.width, self.width_lfunc, self.modes1)
        self.mlpfunc0 = MLP(self.width, self.width_final, self.width_lfunc, act)

        # Expand the hidden dim by 2 because the input is also twice as large
        self.mlp0 = MLP(2 * self.width_lfunc, 2 * self.width_final, self.d_out, act)

    def forward(self, x):
        """
        Input shape (of x):     (batch, channels_in, nx_in)
        Output shape:           (batch, self.d_out)

        The input resolution is determined by x.shape[-1]
        """
        # Lifting layer
        x = x.permute(0, 2, 1)
        x = torch.cat((x, get_grid1d(x.shape, x.device)), dim=-1)  # grid ``features''
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # Fourier integral operator layers on the torus
        for speconv, w in zip(self.speconvs, self.ws):
            x = w(x) + speconv(x)
            x = self.act(x)

        # Extract Fourier neural functionals on the torus
        x_temp = self.lfunc0(x)

        # Retain the truncated modes (use all modes)
        x = x.permute(0, 2, 1)
        x = self.mlpfunc0(x)
        x = x.permute(0, 2, 1)
        x = torch.trapz(x, dx=1. / x.shape[-1])

        # Combine nonlocal and local features
        x = torch.cat((x_temp, x), dim=1)

        # Final projection layer
        x = self.mlp0(x)

        return x
