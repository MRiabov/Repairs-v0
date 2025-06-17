"Module to convert to torchsparse (a dependency necessary for 3d sparse convolutions) and back."

import torchsparse
import torch

def batch_sparse_coo_to_torchsparse(sparse_coos: list[torch.Tensor]):
    assert all(sparse_coo.ndim == 4 for sparse_coo in sparse_coos), (
        f"All tensors must be 4D, but got {[sparse_coo.ndim for sparse_coo in sparse_coos]}"
    )
    coo = torch.concat(sparse_coos, dim=0)
    coo = coo.coalesce()
    return torchsparse.SparseTensor(
        feats=coo.values(), coords=coo.indices(), spatial_range=(256, 256, 256)
    )