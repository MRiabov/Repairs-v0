"""Module for storing unique 3D voxel observations as sparse tensors in a buffer.

This is designed for RL environments with voxel observations where the same
voxel tensors are frequently repeated. Instead of storing duplicates, we store
unique tensors once and reference them by ID.

The buffer maintains a mapping of active IDs and can be periodically cleaned
up to remove unused voxel tensors.

Example:
    buffer = SparseVoxelBuffer()

    # Add voxels
    voxel1 = torch.zeros(32, 32, 32, dtype=torch.float32)
    vid1 = buffer.add(voxel1)

    # Get voxels by ID
    retrieved = buffer.get(vid1)

    # Periodically clean up unused voxels
    active_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    buffer.cleanup(active_ids)
"""

import torch
from typing import Optional, Tuple
from .singleton_buffer import SingletonBuffer


class SparseVoxelBuffer(SingletonBuffer[torch.Tensor]):
    """Buffer that stores unique 3D voxel tensors as sparse tensors.

    Voxels are stored in COO sparse format and referenced by integer IDs.
    The buffer maintains a free list of available IDs for reuse.
    """

    def __init__(
        self,
        batch_size: int,
        shape: tuple[int, ...],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the sparse voxel buffer.

        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self._buffer = torch.sparse_coo_tensor(
            indices=torch.zeros((0, 3), dtype=torch.int64),
            values=torch.zeros((0, *shape), device=device),
            size=(batch_size, *shape),
            device=device,
        )
        self.device = device
        self.known_used_positions = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

    def add(self, voxel_batch: torch.Tensor) -> torch.IntTensor:
        """Add a 4D sparse voxel tensor to the sparse voxel buffer.

        Args:
            voxel_batch: 4D sparse tensor of shape (B, D, H, W) to add

        Returns:
            torch.IntTensor: The IDs corresponding to the voxels in the buffer
        """
        # code tldr: add sparse tensors to another sparse tensor at available position.
        assert voxel_batch.ndim == 4, "voxel_batch must be a 4D tensor"
        assert voxel_batch.is_sparse, "voxel_batch must be a sparse tensor"
        available_positions = torch.nonzero(
            torch.logical_not(self.known_used_positions)
        )
        input_voxel_batch_positions = torch.unique(voxel_batch.indices()[:, 0])
        count_necessary_positions = len(input_voxel_batch_positions)
        assert available_positions.shape[0] >= count_necessary_positions, (
            f"Not enough available positions found: {available_positions.shape[0]} < {count_necessary_positions}"
        )
        assert not self._buffer[available_positions].any(), (
            "Some of the available positions have data in them."
        )
        set_at_positions = available_positions[:count_necessary_positions]
        new_voxel_batch_indices = voxel_batch.indices()
        # 3) Remap each unique batch index in `voxel_batch` so that it points
        #    to the corresponding free row selected in `set_at_positions`
        for i in torch.arange(count_necessary_positions):
            input_pos = input_voxel_batch_positions[i]
            set_at_pos = set_at_positions[i]
            new_voxel_batch_indices = new_voxel_batch_indices.where(
                new_voxel_batch_indices[:, 0] == input_pos, set_at_pos
            )

        self._buffer = torch.sparse_coo_tensor(
            torch.cat((new_voxel_batch_indices, voxel_batch.indices()), dim=0),
            torch.cat((self._buffer.values(), voxel_batch.values()), dim=0),
            size=(self._buffer.size(0), *voxel_batch.shape[1:]),
            device=self.device,
        )
        self._buffer.coalesce()
        self.known_used_positions[set_at_positions] = True
        return set_at_positions
