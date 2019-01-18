#!/usr/bin/env python3

import math
import torch
from .. import beta_features, settings
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

    @property
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        if hasattr(self, "_prior_distribution_memo"):
            return self._prior_distribution_memo
        else:
            out = self.model.forward(self.inducing_points)
            return MultivariateNormal(out.mean, out.lazy_covariance_matrix.evaluate_kernel().add_jitter())

    def prior_covar_logdet(self):
        if hasattr(self, "_logdet_memo"):
            return self._logdet_memo
        else:
            return self.prior_distribution.lazy_covariance_matrix.logdet()

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
        if torch.equal(x, inducing_points):
            return variational_dist
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

            # Cache the prior distribution, for faster training

            # Compute predictive mean/covariance
            predictive_mean = torch.add(test_mean, induc_data_covar.transpose(-1, -2) @ variational_dist.mean)
            if isinstance(variational_dist.lazy_covariance_matrix, RootLazyTensor):
                predictive_covar = RootLazyTensor(
                    (variational_dist.lazy_covariance_matrix.root.transpose(-1, -2) @ induc_data_covar).transpose(-1, -2)
                )
            else:
                predictive_covar = MatmulLazyTensor(
                    induc_data_covar.transpose(-1, -2),
                    predictive_covar @ induc_data_covar
                )

            interp_data_data_var_cg, logdet = induc_induc_covar.inv_quad_logdet(
                induc_data_covar, logdet=(self.training), reduce_inv_quad=False
            )

            # logdet = induc_induc_covar.log_det()

            chol_induc = induc_induc_covar.evaluate().double().cholesky(upper=True)
            A = torch.trtrs(induc_data_covar.double(), chol_induc)[0] # m x n
            interp_data_data_var = (A * A).sum(0).float()
            logdet = chol_induc.diag().pow(2).log().sum().float()
            # solve = induc_induc_covar.inv_matmul(induc_data_covar)
            # norm = torch.norm(induc_induc_covar.matmul(solve) - induc_data_covar) / induc_data_covar.norm()
            # if norm > 0.001:
            from IPython.core.debugger import set_trace
            set_trace()

            diag_correction = DiagLazyTensor((data_data_covar.diag() - interp_data_data_var).clamp(0, math.inf))
            # print(diag_correction.diag().mean(), interp_data_data_var.mean())
            predictive_covar = PsdSumLazyTensor(predictive_covar, diag_correction)

            # Save the logdet, prior distribution for the ELBO
            if self.training:
                induc_mean = full_mean[..., :num_induc]
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
                self._prior_distribution_memo = prior_dist
                self._logdet_memo = logdet

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
        return super(VariationalStrategy, self).__call__(x)
