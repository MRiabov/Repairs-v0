"Module to convert to torchsparse (a dependency necessary for 3d sparse convolutions) and back."

import torchsparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseSiLU(nn.Module):
    """Apply SiLU activation to torchsparse `SparseTensor` features."""

    def forward(self, x: torchsparse.SparseTensor):  
        x.feats = F.silu(x.feats)
        return x


def batch_sparse_coo_to_torchsparse(sparse_coos: list[torch.Tensor]):
    assert all(sparse_coo.ndim == 4 for sparse_coo in sparse_coos), (
        f"All tensors must be 4D, but got {[sparse_coo.ndim for sparse_coo in sparse_coos]}"
    )
    coo = torch.concat(sparse_coos, dim=0)
    coo = coo.coalesce()
    return torchsparse.SparseTensor(
        feats=coo.values().unsqueeze(-1),
        coords=coo.indices(),
        spatial_range=(256, 256, 256),
    )


def sparse_coo_to_torchsparse(sparse_coo: torch.Tensor, dtype=torch.float16):
    coo = sparse_coo.coalesce()
    device = torch.device("cuda")
    return torchsparse.SparseTensor(
        feats=coo.values().unsqueeze(-1).to(dtype=dtype, device=device),
        coords=coo.indices().to(dtype=torch.int32, device=device),
        spatial_range=sparse_coo.shape,
    )
