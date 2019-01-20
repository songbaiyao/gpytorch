#!/usr/bin/env python3

import math
import torch
from .. import settings, beta_features
from ..lazy import DiagLazyTensor, CachedCGLazyTensor, PsdSumLazyTensor, RootLazyTensor, MatmulLazyTensor
from .variational_strategy import VariationalStrategy
from ..distributions import MultivariateNormal


class UnwhitenedVariationalStrategy(VariationalStrategy):
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
            return variational_dist

        # Otherwise, we have to marginalize
        else:
            num_induc = inducing_points.size(-2)
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
            root_variational_covar = variational_dist.lazy_covariance_matrix.root.evaluate()

            # Cache the CG results
            # For now: run variational inference without a preconditioner
            # The preconditioner screws things up for some reason
            with settings.max_preconditioner_size(0):
                # Cache the CG results
                left_tensors = torch.cat([mean_diff, root_variational_covar], -1)
                with torch.no_grad():
                    eager_rhs = torch.cat([left_tensors, induc_data_covar], -1)
                    solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                        induc_induc_covar, eager_rhs.detach(), logdet_terms=self.training,
                        include_tmats=(not settings.skip_logdet_forward.on())
                    )
                    eager_rhss = [
                        eager_rhs.detach(), eager_rhs[..., left_tensors.size(-1):].detach(),
                        eager_rhs[..., :left_tensors.size(-1)].detach()
                    ]
                    solves = [
                        solve.detach(), solve[..., left_tensors.size(-1):].detach(),
                        solve[..., :left_tensors.size(-1)].detach()
                    ]
                    if settings.skip_logdet_forward.on():
                        eager_rhss.append(torch.cat([probe_vecs, induc_data_covar], -1))
                        solves.append(torch.cat([probe_vec_solves, solve[..., left_tensors.size(-1):]], -1))
                induc_induc_covar = CachedCGLazyTensor(
                    induc_induc_covar, eager_rhss=eager_rhss, solves=solves, probe_vectors=probe_vecs,
                    probe_vector_norms=probe_vec_norms, probe_vector_solves=probe_vec_solves,
                    probe_vector_tmats=tmats,
                )

            # Compute predictive mean/covariance
            inv_products = induc_induc_covar.inv_matmul(induc_data_covar, left_tensors.transpose(-1, -2))
            predictive_mean = torch.add(test_mean, inv_products[..., 0, :])
            predictive_covar = RootLazyTensor(inv_products[..., 1:, :].transpose(-1, -2))
            if beta_features.diagonal_correction.on():
                interp_data_data_var, logdet = induc_induc_covar.inv_quad_logdet(
                    induc_data_covar, logdet=(self.training), reduce_inv_quad=False
                )
                diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
                predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
            if self.training:
                mean_diff_inv_quad_and_covar_trace = induc_induc_covar.inv_quad(left_tensors, reduce_inv_quad=False)
                self._prior_distribution_memo = MultivariateNormal(induc_mean, induc_induc_covar)
                self._logdet_memo = logdet
                self._mean_diff_inv_quad_memo = mean_diff_inv_quad_and_covar_trace[..., 0]
                self._covar_trace_memo = mean_diff_inv_quad_and_covar_trace[..., 0:].sum(-1)

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.variational_distribution.initialize_variational_distribution(self.prior_distribution)
            self.variational_params_initialized.fill_(1)
        if self.training:
            if hasattr(self, "_prior_distribution_memo"):
                del self._prior_distribution_memo
            if hasattr(self, "_logdet_memo"):
                del self._logdet_memo
            if hasattr(self, "_mean_diff_inv_quad_memo"):
                del self._mean_diff_inv_quad_memo
            if hasattr(self, "_covar_trace_memo"):
                del self._covar_trace_memo
        return super(VariationalStrategy, self).__call__(x)
