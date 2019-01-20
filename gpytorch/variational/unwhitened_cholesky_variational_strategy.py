#!/usr/bin/env python3

import math
import torch
from .. import beta_features
from ..lazy import DiagLazyTensor, PsdSumLazyTensor, RootLazyTensor, MatmulLazyTensor
from .variational_strategy import VariationalStrategy
from ..distributions import MultivariateNormal


class UnwhitenedCholeskyVariationalStrategy(VariationalStrategy):
    def prior_covar_logdet(self):
        if not hasattr(self, "_logdet_memo"):
            self._logdet_memo = self.prior_distribution.lazy_covariance_matrix.logdet()
        return self._logdet_memo

    def covar_trace(self):
        if not hasattr(self, "_covar_trace_memo"):
            variational_covar = self.variational_distribution.variational_distribution.lazy_covariance_matrix
            variational_covar_root = variational_covar.root_decomposition().root.evaluate()
            prior_covar = self.prior_distribution.lazy_covariance_matrix
            self._covar_trace_memo = prior_covar.inv_quad(variational_covar_root)
        return self._covar_trace_memo

    def forward(self, x):
        variational_dist = self.variational_distribution.variational_distribution
        inducing_points = self.inducing_points
        if inducing_points.dim() < x.dim():
            inducing_points = inducing_points.expand(*x.shape[:-2], *inducing_points.shape[-2:])
            variational_dist = variational_dist.expand(x.shape[:-2])

        num_induc = inducing_points.size(-2)
        num_data = x.size(-2)
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        test_mean = full_mean[..., num_induc:]
        induc_mean = full_mean[..., :num_induc]
        mean_diff = (variational_dist.mean - induc_mean).unsqueeze(-1)

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]
        variational_covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()

        # Compute the Cholesky decomposition of induc_induc_covar
        # Compute interp_data_data_root
        induc_induc_covar_chol = induc_induc_covar.evaluate().double().cholesky()
        left_hand_sides = torch.cat([induc_data_covar, variational_covar_root, mean_diff], -1).double()
        partial_solves = torch.trtrs(left_hand_sides, induc_induc_covar_chol, upper=False)[0].type_as(test_mean)
        # Break apart the solves into their individual compoennts
        induc_data_covar_solve = partial_solves[..., :num_data]
        data_induc_covar_solve = induc_data_covar_solve.transpose(-1, -2)
        variational_covar_root_solve = partial_solves[..., num_data:-1]
        mean_diff_solve = partial_solves[..., -1:]

        # Compute predictive mean
        predictive_mean = torch.add(test_mean, (data_induc_covar_solve @ mean_diff_solve).squeeze(-1))
        predictive_covar = RootLazyTensor(data_induc_covar_solve @ variational_covar_root_solve) 

        if beta_features.diagonal_correction.on():
            interp_data_data_var = (data_induc_covar_solve).pow(2).sum(-1)
            diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
            predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

        # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
        if self.training:
            self._prior_distribution_memo = MultivariateNormal(induc_mean, induc_induc_covar)
            self._logdet_memo = induc_induc_covar_chol.type_as(test_mean).diag().pow(2).log().sum()
            self._covar_trace_memo = variational_covar_root_solve.pow(2).sum(-1).sum(-1)
            self._mean_diff_inv_quad_memo = mean_diff_solve.squeeze(-1).norm(dim=-1).pow(2)

        return MultivariateNormal(predictive_mean, predictive_covar)
