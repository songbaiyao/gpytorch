#!/usr/bin/env python3

import math
import torch
import warnings
from .lazy_tensor import LazyTensor
from .. import settings


class CachedCGLazyTensor(LazyTensor):
    """
    A LazyTensor wrapper that eagerly computes many CG calls in batch.
    This maximizes CG parallelism for fast inference.
    Used primarily for variational inference with GPs.

    Args:
        :attr:`base_lazy_tensor` (:class:`gpytorch.lazy.LazyTensor`): the LazyTensor to wrap
    """

    def __init__(self, base_lazy_tensor, eager_rhs, solves=None, normed_eager_rhs=None):
        # We're precomputing the solves and the normed version of the eager_rhs
        # This will make it faster when we reconstruct the LazyTensor inside functions
        with torch.no_grad():
            if normed_eager_rhs is None:
                normed_eager_rhs = eager_rhs.view(-1, eager_rhs.size(-1))
                norm = normed_eager_rhs.norm(2, dim=0, keepdim=True)
                normed_eager_rhs = normed_eager_rhs.div(norm)

            if solves is None:
                solves = base_lazy_tensor._solve(eager_rhs, base_lazy_tensor._preconditioner()[0])

        super(CachedCGLazyTensor, self).__init__(
            base_lazy_tensor, eager_rhs=eager_rhs, solves=solves, normed_eager_rhs=normed_eager_rhs
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.eager_rhs = eager_rhs.requires_grad_(False)
        self.solves = solves.requires_grad_(False)
        self.normed_eager_rhs = normed_eager_rhs.requires_grad_(False)

    @property
    def requires_grad(self):
        return self.base_lazy_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self.base_lazy_tensor.requires_grad = val

    def _get_indices(self, left_indices, right_indices, *batch_indices):
        return self.base_lazy_tensor._get_indices(left_indices, right_indices, *batch_indices)

    def _getitem(self, *indices):
        return self.base_lazy_tensor._getitem(*indices)

    def _matmul(self, tensor):
        return self.base_lazy_tensor._matmul(tensor)

    def _quad_form_derivative(self, left_vecs, right_vecs):
        return self.base_lazy_tensor._quad_form_derivative(left_vecs, right_vecs)

    def _solve(self, rhs, preconditioner):
        # Here we check to see what solves we've already performed
        with torch.no_grad():
            normed_rhs = rhs.view(-1, rhs.size(-1))
            norm = normed_rhs.norm(2, dim=0, keepdim=True)
            normed_rhs = normed_rhs.div(norm)

            similarity_matrix = self.normed_eager_rhs.t() @ normed_rhs
            similarity_matrix = similarity_matrix.gt(0.9999).type_as(normed_rhs)
            num_precomputed_per_vec = similarity_matrix.sum(-2, keepdim=True)
            similarity_matrix = similarity_matrix.div(num_precomputed_per_vec.clamp(1, math.inf))

            # We want to ensure that either
            # 1) We have precomputed CG for all the vectors, or
            # 1) The vector is of all zeros
            if not torch.all(num_precomputed_per_vec.ge(1) | norm.squeeze_(0).eq(0.)):
                if settings.debug.on():
                    warnings.warn(
                        "CachedCGLazyTensor had to run CG on a tensor of size {}. For best performance, this "
                        "LazyTensor should pre-register all vectors to run CG against.".format(rhs.shape)
                    )
                return super(CachedCGLazyTensor, self)._solve(rhs, preconditioner)

            else:
                return self.solves @ similarity_matrix

    def _size(self):
        return self.base_lazy_tensor._size()

    def _t_matmul(self, tensor):
        return self.base_lazy_tensor._t_matmul(tensor)

    def _transpose_nonbatch(self):
        return self.base_lazy_tensor._transpose_nonbatch()

    def detach_(self):
        self.base_lazy_tensor.detach_()
        return self
