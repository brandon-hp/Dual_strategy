"""Poincare ball manifold."""

import torch

from manifolds.base import Manifold
from utils.math_utils import artanh, tanh
import scipy.special as sc

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c,eval_mode=False):
        # sqrt_c = c ** 0.5
        # dist_c = artanh(
        #     sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        # )
        # dist = dist_c * 2 / sqrt_c
        # return dist ** 2
        assert not torch.isnan(p1).any()
        assert not torch.isnan(p2).any()
        x = p1
        v = p2
        c = c.clamp_min(self.min_norm)
        sqrt_c = c ** 0.5
        if eval_mode:
            vnorm = torch.norm(v, p=2, dim=-1, keepdim=True).transpose(0, 1)
            vnorm=vnorm.clamp_min(self.min_norm)
            xv = x @ v.transpose(0, 1) / vnorm
        else:
            vnorm = torch.norm(v, p=2, dim=-1, keepdim=True)
            vnorm = vnorm.clamp_min(self.min_norm)
            xv = torch.sum(x * v / vnorm, dim=-1, keepdim=True)
        gamma = tanh(sqrt_c * vnorm) / sqrt_c
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        c1 = (1 - 2 * c * gamma * xv + c * gamma ** 2)
        c2 = (1 - c * x2)
        num = torch.sqrt(((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv).clamp_min(1e-15))
        denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
        pairwise_norm = num / denom.clamp_min(self.min_norm)

        # print(torch.mean(denom.clamp_min(MIN_NORM)))
        #print(torch.nonzero(torch.isnan(sqrt_c*pairwise_norm)))
        dist = artanh(sqrt_c * pairwise_norm)
        return 2 * dist / sqrt_c

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = (p.norm(dim=-1, p=2, keepdim=True)).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)


    def concat(self, v, c=None):
        """Concatnates a matrix of dim (M, N) across the last dimension.

        Note: of the inputs are given as a batch, it assumes that dim (B, M, N)
            where B is the batch size.

        The concat operation is based on "Hyperbolic Neural Network++" paper.
        Args:
            v: a tensor of dim(M,N) where M is the number of vectors to be
               concatnated, and N is the dimension of the vectors.

        Returns:
            A tensor of dim(M*N) (or dim(B, M*N in case of batch inputs)) in
            the poincare ball of the same radius.
        """
        del c
        concat_dim = 1 if len(v.shape) == 3 else 0
        a = sc.beta(v.shape[concat_dim] * v.shape[-1] / 2, 0.5) / sc.beta(v.shape[-1] / 2, 0.5)
        # Note that the following multiplication should not be a mobius mul.
        # It is there to normalize the raduis of the new PoincareBall to <= 1.
        return torch.cat(tensors=torch.unbind(v * a, dim=concat_dim), dim=concat_dim)