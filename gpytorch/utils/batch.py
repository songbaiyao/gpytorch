#!/usr/bin/env python3

import torch
from .. import settings

# A slice that does nothing to a dimension
_noop_index = slice(None, None, None)


def _create_batch_indices(batch_shape, repeat_size, device):
    """
    A helper method which creates a list of batch indices for use with the LazyTensor _get_indices method
    """
    batch_indices = []
    for i, batch_size in enumerate(batch_shape):
        batch_index = torch.arange(0, batch_size, dtype=torch.long, device=device).unsqueeze(-1)
        batch_index = batch_index.repeat(
            torch.Size(batch_shape[:i]).numel(),
            torch.Size(batch_shape[i + 1:]).numel() * repeat_size
        ).view(-1)
        batch_indices.append(batch_index)
    return batch_indices


def _compute_getitem_size(obj, indices):
    if obj.dim() != len(indices):
        raise RuntimeError(
            "_compute_getitem_size assumes that obj (size: {}) and indices (len: {}) have the "
            "same dimensionality.".format(obj.shape, len(indices))
        )

    final_shape = []
    first_tsr_idx = None
    tensor_idx_shape = None

    for i, (size, idx) in enumerate(zip(obj.shape, indices)):
        # Handle slice: that dimension gets downsized
        if isinstance(idx, slice):
            if idx == _noop_index:
                final_shape.append(size)
            else:
                final_shape.append(len(range(*idx.indices(size))))

        # Handle int: we "lose" that dimension
        elif isinstance(idx, int):
            if settings.debug.on():
                try:
                    range(size)[idx]
                except IndexError as e:
                    raise IndexError(
                        "index element {} ({}) is invalid: out of range for obj of size "
                        "{}.".format(i, idx, obj.shape)
                    )

        # Handle tensor index - this one is complicated
        elif torch.is_tensor(idx):
            if tensor_idx_shape is None:
                first_tsr_idx = len(final_shape)
                tensor_idx_shape = idx.numel()
                final_shape.append(tensor_idx_shape)
            else:
                if settings.debug.on():
                    if idx.numel() != tensor_idx_shape:
                        raise IndexError(
                            "index element {} is an invalid size: expected tensor indices of size {}, got "
                            "{}.".format(i, tenosr_idx_shape, idx.numel())
                        )

    return torch.Size(final_shape), first_tsr_idx
