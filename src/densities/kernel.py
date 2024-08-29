
import torch
import torch.nn as nn
from warnings import warn

class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    @property
    def name(self) -> str:
        return "RBF"

    def forward(self, X, Y):
        _XX = X.matmul(X.t())
        _XY = X.matmul(Y.t())
        _YY = Y.matmul(Y.t())

        _dnorm2 = -2 * _XY + _XX.diag().unsqueeze(1) + _YY.diag().unsqueeze(0)
        
        if ~(_dnorm2.isfinite().all()):
            warn("In RBF kernel: at least one element of dnorm2 is not finite!")

        # Apply the median heuristic 
        # torch.median gives the "lower" median in case of an even number elements
        # we compute the mean of the "two" medians in this case using quantile(0.5)
        if self.sigma is None:
            _median = torch.nanquantile(_dnorm2.detach(), q=0.5)
            _sigma = _median / (2 * torch.log(torch.tensor(X.size(0) + 1)))
        else:
            _sigma = self.sigma ** 2

        _gamma = 1.0 / (2 * _sigma)
        _K_XY = (-_gamma * _dnorm2).exp()

        if ~_K_XY.isfinite().all():
            print(f"gram has infinite values! gram: {_K_XY}\n Y: {Y}\n XX: {_XX}\n dnorm2: {_dnorm2}\n sigma: {_sigma}")

        return _K_XY
    