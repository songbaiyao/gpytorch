#!/usr/bin/env python3

import torch
from .marginal_log_likelihood import MarginalLogLikelihood
from .. import settings


class VariationalELBO(MarginalLogLikelihood):
    def __init__(self, likelihood, model, num_data, combine_terms=True):
        """
        A special MLL designed for variational inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        - combine_terms: (bool) - whether or not to sum the expected NLL with the KL terms (default True)
        """
        super(VariationalELBO, self).__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data

    def forward(self, variational_dist_f, target, **kwargs):
        num_batch = variational_dist_f.event_shape.numel()
        variational_dist_u = self.model.variational_strategy.variational_distribution.variational_distribution
        prior_dist = self.model.variational_strategy.prior_distribution

        if len(variational_dist_u.batch_shape) < len(prior_dist.batch_shape):
            variational_dist_u = variational_dist_u.expand(prior_dist.batch_shape)

        log_likelihood = self.likelihood.variational_log_probability(variational_dist_f, target, **kwargs).div(
            num_batch
        )

        kl_divergence = 0.5 * sum([
            # log|k| - log|S|
            # = log|K| - log|K var_dist_covar K|
            # = -log|K| - log|var_dist_covar|
            -self.model.variational_strategy.prior_covar_logdet(),
            -variational_dist_u.lazy_covariance_matrix.logdet(),
            # tr(K^-1 S) = tr(K^1 K var_dist_covar K) = tr(K var_dist_covar)
            (variational_dist_u.covariance_matrix * prior_dist.covariance_matrix).view(
                *prior_dist.batch_shape, -1
            ).sum(-1),
            # (m - \mu u)^T K^-1 (m - \mu u)
            # = (K^-1 (m - \mu u)) K (K^1 (m - \mu u))
            # = (var_dist_mean)^T K (var_dist_mean)
            (variational_dist_u.mean * (prior_dist.lazy_covariance_matrix @ variational_dist_u.mean)).sum(-1),
            # d
            -prior_dist.event_shape.numel()
        ])

        if kl_divergence.dim() > log_likelihood.dim():
            kl_divergence = kl_divergence.sum(-1)

        if log_likelihood.numel() == 1:
            kl_divergence = kl_divergence.sum()

        kl_divergence = kl_divergence.div(self.num_data)

        if self.combine_terms:
            res = log_likelihood - kl_divergence
            for _, prior, closure, _ in self.named_priors():
                res.add_(prior.log_prob(closure()).sum().div(self.num_data))
            return res
        else:
            log_prior = torch.zeros_like(log_likelihood)
            for _, prior, closure, _ in self.named_priors():
                log_prior.add_(prior.log_prob(closure()).sum())
            return log_likelihood, kl_divergence, log_prior.div(self.num_data)


'''
class VariationalELBO(MarginalLogLikelihood):
    def __init__(self, likelihood, model, num_data, combine_terms=True):
        """
        A special MLL designed for variational inference

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        - combine_terms: (bool) - whether or not to sum the expected NLL with the KL terms (default True)
        """
        super(VariationalELBO, self).__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data

    def forward(self, variational_dist_f, target, **kwargs):
        num_batch = variational_dist_f.event_shape.numel()
        variational_dist_u = self.model.variational_strategy.variational_distribution.variational_distribution
        prior_dist = self.model.variational_strategy.prior_distribution

        if len(variational_dist_u.batch_shape) < len(prior_dist.batch_shape):
            variational_dist_u = variational_dist_u.expand(prior_dist.batch_shape)

        log_likelihood = self.likelihood.variational_log_probability(variational_dist_f, target, **kwargs).div(
            num_batch
        )

        kl_divergence = torch.distributions.kl.kl_divergence(variational_dist_u, prior_dist)

        if kl_divergence.dim() > log_likelihood.dim():
            kl_divergence = kl_divergence.sum(-1)

        if log_likelihood.numel() == 1:
            kl_divergence = kl_divergence.sum()

        kl_divergence = kl_divergence.div(self.num_data)

        if self.combine_terms:
            res = log_likelihood - kl_divergence
            for _, prior, closure, _ in self.named_priors():
                res.add_(prior.log_prob(closure()).sum().div(self.num_data))
            return res
        else:
            log_prior = torch.zeros_like(log_likelihood)
            for _, prior, closure, _ in self.named_priors():
                log_prior.add_(prior.log_prob(closure()).sum())
            return log_likelihood, kl_divergence, log_prior.div(self.num_data)
'''


class VariationalELBOEmpirical(VariationalELBO):
    def __init__(self, likelihood, model, num_data):
        """
        A special MLL designed for variational inference.
        This computes an empirical (rather than exact) estimate of the KL divergence

        Args:
        - likelihood: (Likelihood) - the likelihood for the model
        - model: (Module) - the variational GP model
        - num_data: (int) - the total number of training data points (necessary for SGD)
        """
        super(VariationalELBOEmpirical, self).__init__(likelihood, model, num_data, combine_terms=True)

    def forward(self, variational_dist_f, target, **kwargs):
        num_batch = variational_dist_f.event_shape[0]
        variational_dist_u = self.model.variational_strategy.variational_distribution.variational_distribution
        prior_dist = self.model.variational_strategy.prior_distribution

        log_likelihood = self.likelihood.variational_log_probability(variational_dist_f, target, **kwargs)
        log_likelihood = log_likelihood.div(num_batch)

        num_samples = settings.num_likelihood_samples.value()
        variational_samples = variational_dist_u.rsample(torch.Size([num_samples]))
        kl_divergence = (
            variational_dist_u.log_prob(variational_samples) - prior_dist.log_prob(variational_samples)
        ).mean(0)
        kl_divergence = kl_divergence.div(self.num_data)

        res = log_likelihood - kl_divergence
        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum().div(self.num_data))
        return res
