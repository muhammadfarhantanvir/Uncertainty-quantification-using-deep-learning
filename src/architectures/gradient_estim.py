import torch
import torch.autograd as autograd

# based on an original implementation from https://github.com/AntixK/Spectral-Stein-Gradient/blob/master/score_estimator/spectral_stein.py
class SpectralSteinEstimator():
    def __init__(self, eta=None, num_eigs=None, kernel=None, samples = None):
        self.eta = eta
        self.num_eigs = num_eigs
        self.kernel = kernel
        self.samples = samples
        if samples is not None:
            self.beta, self.eigen_vals, self.eigen_vecs = self.compute_beta(samples)

    @property
    def name(self) -> str:
        return "SpectralSteinEstimator"

    def compute_beta(self,samples):
        _num_samples = torch.tensor(samples.size(-2), dtype=torch.float)

        samples = samples.detach().requires_grad_(True)

        #_gram = self.kernel(samples, samples.detach())
        _gram = self.kernel(samples, samples.detach().requires_grad_(True))

        _gram_grad = autograd.grad(_gram.sum(), samples)[0]

        # Kxx = Kxx + eta * I
        if self.eta is not None:
            _gram += self.eta * torch.eye(samples.size(-2), device=_gram.device)

        _eigen_vals, _eigen_vecs = torch.linalg.eig(_gram)

        if self.num_eigs is not None:
            _eigen_vals = _eigen_vals[:self.num_eigs]
            _eigen_vecs = _eigen_vecs[:, :self.num_eigs]

        # Compute the Monte Carlo estimate of the gradient of
        # the eigenfunction at x
        _gram_grad_avg = -_gram_grad/samples.shape[0] # [M x D]

        _beta = torch.matmul(- torch.sqrt(_num_samples) * _eigen_vecs.real.t(),
                             _gram_grad_avg)
        _beta *= (1. / _eigen_vals.real.unsqueeze(-1))

        return _beta, _eigen_vals, _eigen_vecs

    def nystrom_method(self, x, samples, eigen_vecs, eigen_vals):
        """
        Implements the Nystrom method for approximating the
        eigenfunction (generalized eigenvectors) for the kernel
        at x using the M samples (x_m). It is given
        by -
         .. math::
            phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)
        :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        :param samples: (Tensor) Sample points from the data of size M [M x D]
        :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        :return: Eigenfunction at x [N x M]
        """
        _num_samples = torch.tensor(samples.size(-2), dtype=torch.float)

        Kxxm = self.kernel(x, samples)
        _phi_x = torch.matmul(torch.sqrt(_num_samples) * Kxxm,
                              eigen_vecs.real)

        _phi_x *= 1. / eigen_vals.real  # Take only the real part of the eigenvals
                                        # as the Im is 0 (Symmetric matrix)

        return _phi_x
    
    def compute_score_gradients(self, x, samples=None):
        """
        Computes the Spectral Stein Gradient Estimate (SSGE) for the
        score function. The SSGE is given by
        .. math::
            \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)
            \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)
            \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)
        :param x: (Tensor) Point at which the gradient is evaluated [N x D]
        :param samples: (Tensor) Samples for the kernel [M x D]
        :return: gradient estimate [N x D]
        """
        
        if samples is None:
            _samples = self.samples
            _beta = self.beta
            _eigen_vecs = self.eigen_vecs
            _eigen_vals = self.eigen_vals
        else:
            _samples = samples
            _beta, _eigen_vals, _eigen_vecs = self.compute_beta(_samples)

        _phi_x = self.nystrom_method(x, _samples, _eigen_vecs, _eigen_vals)  # [N x M]

        _g = torch.matmul(_phi_x, _beta)  # [N x D]

        return _g
    