import torch
from typing import Dict, Generic, TypeVar, Any
from tensordict import TensorDict
from torchrl.data import TensorStorage
from torchrl.data.replay_buffers import TensorDictReplayBuffer, ReplayBuffer


T = TypeVar("T")


class SingletonBuffer(Generic[T]):
    """Base class for singleton buffers that store unique items by ID.

    This class provides the core functionality for managing unique items with ID-based access.
    Subclasses should implement the specific item hashing and comparison logic.
    """

    def __init__(
        self,
        tensor_shape: tuple[int, ...] | dict[str, tuple[int, ...]],
        buffer_size: int = 512,
        batch_dim: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the singleton buffer.

        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.device = device
        # Maps item ID to item data
        self._tensor_shape = tensor_shape
        if isinstance(tensor_shape, dict):
            self._items = TensorDict(
                {
                    key: torch.zeros((batch_dim, *shape), device=device)
                    for key, shape in tensor_shape.items()
                },
                batch_size=(batch_dim,),
            )
            self._buffer_storage = TensorStorage(
                self._items, device=torch.device(device)
            )
            self._buffer = TensorDictReplayBuffer(
                storage=self._buffer_storage,
                batch_size=batch_dim,
            )
        else:
            self._items = torch.zeros((batch_dim,) + tensor_shape, device=device)
            self._buffer_storage = TensorStorage(
                self._items, device=torch.device(device)
            )
            self._buffer = ReplayBuffer(
                storage=self._buffer_storage,
                batch_size=batch_dim,
                compilable=False,  # TODO probably set to true (when ready).
            )

        # List of free IDs for reuse
        self.known_used_positions = torch.zeros(
            (buffer_size,), dtype=torch.bool, device=device
        )

        # Next available ID (when no free IDs are available)
        self._next_id = 0

    def add(self, batch: torch.Tensor | TensorDict) -> int:
        """Add an item to the buffer if not already present.

        Args:
            item: The item to add

        Returns:
            int: The ID corresponding to the item in the buffer
        """
        ids = self._buffer.add(batch)
        assert not torch.all(self.known_used_positions[ids]), (
            "None of the already used positions were overwritten."
        )
        self.known_used_positions[ids] = True
        return ids

    def get(self, item_ids: torch.IntTensor) -> T:
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
        return self._buffer[item_ids]

    def cleanup(self, active_ids: torch.Tensor) -> None:
        """Remove items that are no longer in use.

        Args:
            active_ids: Tensor of item IDs that are currently in use
        """
        assert (self.known_used_positions[active_ids]).any(), (
            "Some of queried positions were empty"
        )
        self.known_used_positions[active_ids] = False
        # cleanup assuming the tensor is dense:
        self._buffer[active_ids] = torch.zeros_like(self._buffer[active_ids])
