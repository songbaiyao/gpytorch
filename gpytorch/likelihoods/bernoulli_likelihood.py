#!/usr/bin/env python3

import torch
import math
import numpy as np
from torch.distributions import Bernoulli

from .. import settings
from ..distributions import MultivariateNormal
from ..functions import log_normal_cdf, normal_cdf
from .likelihood import Likelihood


class BernoulliLikelihood(Likelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF \Phi(x)). Given the identity \Phi(-x) =
    1-\Phi(x), we can write the likelihood compactly as:

    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi(yf)
        \end{equation*}
    """
    def __init__(self):
        super().__init__()
        def hermgauss(n):
            x, w = np.polynomial.hermite.hermgauss(n)
            return x, w

        x, w = hermgauss(20)
        x = torch.Tensor(x)
        w = torch.Tensor(w)
        self.register_buffer('gauss_hermite_locs', x)
        self.register_buffer('gauss_hermite_weights', w)

    def forward(self, input):
        if not isinstance(input, MultivariateNormal):
            raise RuntimeError(
                "BernoulliLikelihood expects a multi-variate normally distributed latent function to make predictions"
            )

        mean = input.mean
        var = input.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = normal_cdf(link)
        return Bernoulli(probs=output_probs)

    def variational_log_probability(self, latent_func, target):
        mus = latent_func.mean
        vars = latent_func.variance

        shifted_locs = torch.sqrt(2.0 * vars).unsqueeze(-1) * self.gauss_hermite_locs + mus.unsqueeze(-1)
        evals = log_normal_cdf(shifted_locs.mul(target.unsqueeze(-1)))
        res = (1 / math.sqrt(math.pi)) * (evals * self.gauss_hermite_weights).sum(-1)
        return res.sum()
        # num_samples = settings.num_likelihood_samples.value()
        # samples = latent_func.rsample(torch.Size([num_samples])).view(-1)
        # target = target.unsqueeze(0).repeat(num_samples, 1).view(-1)
        # return log_normal_cdf(samples.mul(target)).sum().div(num_samples)

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        f_samples = variational_dist_f(sample_shape)
        y_prob_samples = torch.distributions.Normal(0, 1).cdf(f_samples)
        y_dist = pyro.distributions.Bernoulli(y_prob_samples)
        pyro.sample(name_prefix + "._training_labels", y_dist.independent(1), obs=y_obs)
