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
from repairs_components.processing.voxel_export import sparse_arr_put, sparse_arr_remove
from singleton_buffer import SingletonBuffer


class SparseVoxelBuffer(SingletonBuffer[torch.Tensor]):
    """Buffer that stores unique 3D voxel tensors as sparse tensors.

    Voxels are stored in COO sparse format and referenced by integer IDs.
    The buffer maintains a free list of available IDs for reuse.
    """

    def __init__(
        self,
        batch_size: int,
        buffer_size: int,
        voxel_shape: tuple[int, ...],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the sparse voxel buffer.

        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self._buffer = torch.sparse_coo_tensor(
            indices=torch.zeros((4, 0), dtype=torch.int16),
            values=torch.zeros((0), dtype=torch.int8, device=device),
            size=(buffer_size, *voxel_shape),
            device=device,
        )
        self._buffer = self._buffer.coalesce()
        self.device = device
        self.known_used_positions = torch.zeros(
            buffer_size, dtype=torch.bool, device=device
        )

    def add(self, voxel_batch: torch.Tensor) -> torch.IntTensor:
        """Add a 4D sparse voxel tensor to the sparse voxel buffer.

        Args:
            voxel_batch: 4D sparse tensor of shape (B, D, H, W) to add

        Returns:
            torch.IntTensor: The IDs corresponding to the voxels in the buffer
        """
        voxel_batch = voxel_batch.coalesce()
        assert not voxel_batch.indices().nonzero().shape[0] == 0, (
            "Passed empty voxel batch to buffer add"
        )
        # code tldr: add sparse tensors to another sparse tensor at available position.
        assert voxel_batch.ndim == 4, "voxel_batch must be a 4D tensor"
        assert voxel_batch.is_sparse, "voxel_batch must be a sparse tensor"
        available_positions = torch.nonzero(
            torch.logical_not(self.known_used_positions)
        ).squeeze(1)
        input_voxel_batch_positions = torch.unique(voxel_batch.indices()[0])  # [:, 0]
        count_necessary_positions = len(input_voxel_batch_positions)
        assert available_positions.shape[0] >= count_necessary_positions, (
            f"Not enough available positions found: {available_positions.shape[0]} < {count_necessary_positions}"
        )
        assert not self._buffer.index_select(0, available_positions).any(), (
            "Some of the available positions have data in them."
        )
        set_at_positions = available_positions[:count_necessary_positions]

        self._buffer = sparse_arr_put(
            self._buffer, voxel_batch, set_at_positions, dim=0
        )
        self.known_used_positions[set_at_positions] = True
        return set_at_positions
        # note: I've chnaged this to use the util, rollback if problematic.

    def get(self, item_ids: torch.IntTensor) -> torch.Tensor:
        """Retrieve an item by its ID.

        Args:
            item_id: The ID of the item to retrieve

        Returns:
            The stored item

        Raises:
            ValueError: If the item ID is not found in the buffer
        """
        assert (self.known_used_positions[item_ids]).all(), (
            "Some of queried positions were empty"
        )
        return self._buffer.index_select(0, item_ids.long())  # get by batch dim.

    def cleanup(self, active_ids: torch.IntTensor) -> None:
        """Remove items that are no longer in use."""
        assert (self.known_used_positions[active_ids]).all(), (
            "Not all active positions are marked as in use"
        )
        updated_known_used_positions = torch.zeros_like(self.known_used_positions)
        updated_known_used_positions[active_ids] = True
        self.known_used_positions = updated_known_used_positions
        remove_idx = torch.arange(self.known_used_positions.shape[0])[
            ~updated_known_used_positions
        ]  # basically a nonzero.

        # cleanup assuming the tensor is dense:
        # doc: all edge idx that are of not from active_ids should be removed.
        self._buffer = sparse_arr_remove(
            self._buffer.coalesce(), remove_idx=remove_idx, dim=0
        )
