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

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the sparse voxel buffer.
        
        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        super().__init__()
        self.device = device

    def _get_item_hash(self, item: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a hashable key for the voxel tensor.
        
        Args:
            item: Sparse tensor to hash
            
        Returns:
            Tuple of (indices, values) that uniquely identify the sparse tensor
        """
        if not item.is_sparse:
            item = item.to_sparse()
        return (item.indices(), item.values())
    
    def _prepare_item(self, item: torch.Tensor) -> torch.Tensor:
        """Convert dense tensor to sparse and move to correct device."""
        if not item.is_sparse:
            item = item.to_sparse()
        return item.to(self.device)

    def add(self, voxel: torch.Tensor) -> int:
        """Add a 3D voxel tensor to the buffer if not already present.
        
        Args:
            voxel: 3D tensor of shape (D, H, W) to add
            
        Returns:
            int: The ID corresponding to the voxel in the buffer
        """
        return super().add(voxel)

    def get(self, voxel_id: int, shape: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """Retrieve a voxel tensor by its ID.
        
        Args:
            voxel_id: The ID of the voxel to retrieve
            shape: Optional shape to reshape the dense tensor to
            
        Returns:
            torch.Tensor: The stored sparse voxel tensor
            
        Raises:
            ValueError: If the voxel ID is not found in the buffer
        """
        sparse_tensor = super().get(voxel_id)
        if shape is not None:
            return sparse_tensor.to_dense().view(shape)
        return sparse_tensor

    # All these methods are now inherited from SingletonBuffer:
    # - cleanup()
    # - get_active_ids()
    # - __len__()
    # - clear()
