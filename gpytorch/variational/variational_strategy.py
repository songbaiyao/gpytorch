#!/usr/bin/env python3

import math
import torch
from .. import settings, beta_features
from ..lazy import DiagLazyTensor, CachedCGLazyTensor, PsdSumLazyTensor, RootLazyTensor, MatmulLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal


class VariationalStrategy(Module):
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

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=False):
        """
        Args:
            model (:obj:`gpytorch.model.AbstractVariationalGP`): Model this strategy is applied to. Typically passed in
            when the VariationalStrategy is created in the __init__ method of the user defined model.
            inducing_points (torch.tensor): Tensor containing a set of inducing points to use for variational inference.
            variational_distribution (:obj:`gpytorch.variational.VariationalDistribution`): A VariationalDistribution
                object that represents the form of the variational distribution :math:`q(u)`
            learn_inducing_locations (bool): Whether or not the inducing point locations should be learned (e.g. SVGP).
        """
        super(VariationalStrategy, self).__init__()
        object.__setattr__(self, "model", model)

        inducing_points = inducing_points.clone()

        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        self.variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    def mean_diff_inv_quad(self):
        if not hasattr(self, "_mean_diff_inv_quad_memo"):
            prior_mean = self.prior_distribution.mean
            prior_covar = self.prior_distribution.lazy_covariance_matrix
            variational_mean = self.variational_distribution.variational_distribution.mean
            self._mean_diff_inv_quad_memo = prior_covar.inv_quad(variational_mean - prior_mean)
        return self._mean_diff_inv_quad_memo

    @property
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        if not hasattr(self, "_prior_distribution_memo"):
            out = self.model.forward(self.inducing_points)
            self._prior_distribution_memo = MultivariateNormal(
                out.mean, out.lazy_covariance_matrix.evaluate_kernel().add_jitter()
            )
        return self._prior_distribution_memo

    def prior_covar_logdet(self):
        if not hasattr(self, "_logdet_memo"):
            self._logdet_memo = -self.prior_distribution.lazy_covariance_matrix.logdet()
        return self._logdet_memo

    def covar_trace(self):
        if not hasattr(self, "_covar_trace_memo"):
            variational_covar = self.variational_distribution.variational_distribution.covariance_matrix
            prior_covar = self.prior_distribution.covariance_matrix
            batch_shape = prior_covar.shape[:-2]
            self._covar_trace_memo = (variational_covar * prior_covar).view(*batch_shape, -1).sum(-1)
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
            induc_mean = full_mean[..., :num_induc]
            mean_diff = (variational_dist.mean - induc_mean).unsqueeze(-1)

            # Covariance terms
            induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter()
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]

            # Cache the CG results
            # For now: run variational inference without a preconditioner
            # The preconditioner screws things up for some reason
            with settings.max_preconditioner_size(0):
                with torch.no_grad():
                    eager_rhs = torch.cat([induc_data_covar, mean_diff], -1)
                    solve, probe_vecs, probe_vec_norms, probe_vec_solves, tmats = CachedCGLazyTensor.precompute_terms(
                        induc_induc_covar, eager_rhs.detach(), logdet_terms=self.training,
                        include_tmats=(not settings.skip_logdet_forward.on())
                    )
                    eager_rhss = [eager_rhs.detach()]
                    solves = [solve.detach()]
                    if settings.skip_logdet_forward.on() and self.training:
                        eager_rhss.append(torch.cat([probe_vecs, eager_rhs], -1))
                        solves.append(torch.cat([probe_vec_solves, solve[..., :eager_rhs.size(-1)]], -1))
                    elif not self.training:
                        eager_rhss.append(eager_rhs[..., :-1])
                        solves.append(solve[..., :-1])

                induc_induc_covar = CachedCGLazyTensor(
                    induc_induc_covar, eager_rhss=eager_rhss, solves=solves, probe_vectors=probe_vecs,
                    probe_vector_norms=probe_vec_norms, probe_vector_solves=probe_vec_solves,
                    probe_vector_tmats=tmats,
                )

            # Compute some terms that will be necessary for the predicitve covariance and KL divergence
            if self.training:
                interp_data_data_var_plus_mean_diff_inv_quad, logdet = induc_induc_covar.inv_quad_logdet(
                    torch.cat([induc_data_covar, mean_diff], -1), logdet=True, reduce_inv_quad=False
                )
                interp_data_data_var = interp_data_data_var_plus_mean_diff_inv_quad[..., :-1]
                mean_diff_inv_quad = interp_data_data_var_plus_mean_diff_inv_quad[..., -1]
            elif beta_features.diagonal_correction.on():
                interp_data_data_var = induc_induc_covar.inv_quad(induc_data_covar, reduce_inv_quad=False)

            # Compute predictive mean
            predictive_mean = torch.add(test_mean, induc_induc_covar.inv_matmul(
                mean_diff, left_tensor=induc_data_covar.transpose(-1, -2)
            ).squeeze(-1))

            # Compute the predictive covariance
            if isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor):
                predictive_covar = RootLazyTensor(
                    induc_data_covar.transpose(-1, -2) @ variational_dist.lazy_covariance_matrix.root.evaluate()
                )
            else:
                predictive_covar = MatmulLazyTensor(
                    induc_data_covar.transpose(-1, -2), predictive_covar @ induc_data_covar
                )

            if beta_features.diagonal_correction.on():
                diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
                predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
            if self.training:
                self._prior_distribution_memo = MultivariateNormal(induc_mean, induc_induc_covar)
                self._logdet_memo = -logdet
                self._mean_diff_inv_quad_memo = mean_diff_inv_quad

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
