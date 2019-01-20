#!/usr/bin/env python3

import math
import torch
from ..lazy import DiagLazyTensor, PsdSumLazyTensor, RootLazyTensor, MatmulLazyTensor
from .variational_strategy import VariationalStrategy
from ..distributions import MultivariateNormal


class CholeskyVariationalStrategy(VariationalStrategy):
    """
    VariationalStrategy objects control how certain aspects of variational inference should be performed. In particular,
    they define two methods that get used during variational inference:

    #. The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
       GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
       this is done simply by calling the user defined GP prior on the inducing point data directly.
    # The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
      inducing point function values. Specifically, forward defines how to transform a variational distribution
      over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
      specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

    In GPyTorch, we currently support two example instances of this latter functionality. In scenarios where the
    inducing points are learned or at least not constrained to a grid, we apply the derivation in Hensman et al., 2015
    to exactly marginalize out the variational distribution. When the inducing points are constrained to a grid, we
    apply the derivation in Wilson et al., 2016 and exploit a deterministic relationship between f and u.
    """

    def mean_diff_inv_quad(self):
        if not hasattr(self, "_mean_diff_inv_quad_memo"):
            variational_mean = self.variational_distribution.variational_distribution.mean
            self._mean_diff_inv_quad_memo = variational_mean.norm(2, dim=-1).pow(2)
        return self._mean_diff_inv_quad_memo

    def prior_covar_logdet(self):
        if not hasattr(self, "_logdet_memo"):
            self._logdet_memo = torch.zeros(
                self.inducing_points.shape[:-2], dtype=self.inducing_points.dtype,
                device=self.inducing_points.device
            )
        return self._logdet_memo

    def covar_trace(self):
        if not hasattr(self, "_covar_trace_memo"):
            variational_covar = self.variational_distribution.variational_distribution.lazy_covariance_matrix
            self._covar_trace_memo = variational_covar.diag().sum(-1)
        return self._covar_trace_memo

    def forward(self, x):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        Args:
            x (torch.tensor): Locations x to get the variational posterior of the function values at.
        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(f|x)
        """
        variational_dist = self.variational_distribution.variational_distribution
        inducing_points = self.inducing_points
        if inducing_points.dim() < x.dim():
            inducing_points = inducing_points.expand(*x.shape[:-2], *inducing_points.shape[-2:])
            variational_dist = variational_dist.expand(x.shape[:-2])

        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            # De-whiten the prior covar
            prior_covar = self.prior_distribution.lazy_covariance_matrix
            if isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor):
                predictive_covar = RootLazyTensor(
                    prior_covar @ variational_dist.lazy_covariance_matrix.root.evaluate()
                )
            else:
                predictive_covar = MatmulLazyTensor(
                    prior_covar @ variational_dist.covariance_matrix, prior_covar,
                )

            # Cache some values for the KL divergence
            if self.training:
                self._mean_diff_inv_quad_memo, self._logdet_memo = prior_covar.inv_quad_logdet(
                    (variational_dist.mean - self.prior_distribution.mean), logdet=True
                )

            return MultivariateNormal(variational_dist.mean, predictive_covar)

        # Otherwise, we have to marginalize
        else:
            num_induc = inducing_points.size(-2)
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs)
            full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

            # Mean terms
            test_mean = full_mean[..., num_induc:]

            # Covariance terms
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]

            # Compute the Cholesky decomposition of induc_induc_covar
            # Compute interp_data_data_root
            induc_induc_covar_chol = induc_induc_covar.evaluate().double().cholesky()
            interp_data_data_root = torch.trtrs(
                induc_data_covar.double(), induc_induc_covar_chol, upper=False
            )[0].type_as(induc_data_covar).transpose(-1, -2)

            # Compute predictive mean
            predictive_mean = torch.add(test_mean, interp_data_data_root @ variational_dist.mean)

            # Compute the predictive covariance
            if isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor):
                predictive_covar = RootLazyTensor(
                    interp_data_data_root @ variational_dist.lazy_covariance_matrix.root.evaluate()
                )
            else:
                predictive_covar = MatmulLazyTensor(
                    interp_data_data_root, predictive_covar @ interp_data_data_root.transpose(-1, -2)
                )
            interp_data_data_var = interp_data_data_root.pow(2).sum(-1)
            diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
            predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
            if self.training:
                induc_mean = full_mean[..., :num_induc]
                self._prior_distribution_memo = MultivariateNormal(induc_mean, induc_induc_covar)

            return MultivariateNormal(predictive_mean, predictive_covar)
