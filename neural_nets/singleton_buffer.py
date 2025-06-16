import torch
from typing import Dict, Generic, TypeVar, Tuple, Any

# Type variable for generic item type
T = TypeVar('T')

# Type aliases for better type hints
TensorDict = Dict[str, torch.Tensor]
ItemHash = Tuple[torch.Tensor, ...]

# Type aliases for better type hints
TensorDict = Dict[str, torch.Tensor]
ItemHash = Tuple[torch.Tensor, ...]

class SingletonBuffer(Generic[T]):
    """Base class for singleton buffers that store unique items by ID.
    
    This class provides the core functionality for managing unique items with ID-based access.
    Subclasses should implement the specific item hashing and comparison logic.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the singleton buffer.
        
        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.device = device
        # Maps item ID to item data
        self._items: Dict[int, T] = {}
        # Maps item hash to item ID (for deduplication)
        self._hash_to_id: Dict[Any, int] = {}
        # Tensor of active item IDs
        self._active_ids = torch.tensor([], dtype=torch.long, device=device)
        # List of free IDs for reuse
        self._free_ids = torch.tensor([], dtype=torch.long, device=device)
        # Next available ID (when no free IDs are available)
        self._next_id = 0
    
    def _get_item_hash(self, item: T) -> Any:
        """Generate a hashable key for the item for deduplication.
        
        Subclasses must implement this method to provide item-specific hashing.
        The hash should be a tuple of tensors or other hashable types.
        """
        raise NotImplementedError("Subclasses must implement _get_item_hash")
    
    def _prepare_item(self, item: T) -> T:
        """Prepare the item for storage (e.g., move to correct device).
        
        Subclasses can override this method to perform any necessary preprocessing.
        """
        return item
    
    def add(self, item: T) -> int:
        """Add an item to the buffer if not already present.
        
        Args:
            item: The item to add
            
        Returns:
            int: The ID corresponding to the item in the buffer
        """
        item = self._prepare_item(item)
        item_hash = self._get_item_hash(item)
        
        # Check if we've seen this item before
        if item_hash in self._hash_to_id:
            return self._hash_to_id[item_hash]
        
        # Get a new ID (either from free list or next available)
        if len(self._free_ids) > 0:
            item_id = int(self._free_ids[0])
            self._free_ids = self._free_ids[1:]  # Remove from free list
        else:
            item_id = self._next_id
            self._next_id += 1
        
        # Store the item
        self._items[item_id] = item
        self._hash_to_id[item_hash] = item_id
        
        # Update active IDs
        self._active_ids = torch.cat([self._active_ids, torch.tensor([item_id], device=self.device)])
        
        return item_id
    
    def get(self, item_id: int) -> T:
        """Retrieve an item by its ID.
        
        Args:
            item_id: The ID of the item to retrieve
            
        Returns:
            The stored item
            
        Raises:
            ValueError: If the item ID is not found in the buffer
        """
        if item_id not in self._items:
            raise ValueError(f"Item ID {item_id} not found in buffer")
        return self._items[item_id]
    
    def cleanup(self, active_ids: torch.Tensor) -> None:
        """Remove items that are no longer in use.
        
        Args:
            active_ids: Tensor of item IDs that are currently in use
        """
        if not isinstance(active_ids, torch.Tensor):
            active_ids = torch.tensor(active_ids, dtype=torch.long, device=self.device)
        
        # Convert to set for faster lookups
        active_set = set(active_ids.cpu().numpy())
        
        # Find IDs to remove
        to_remove = []
        for item_id in list(self._items.keys()):
            if item_id not in active_set:
                to_remove.append(item_id)
        
        # Remove unused items and add their IDs to free list
        for item_id in to_remove:
            # Remove from mappings
            item = self._items.pop(item_id)
            item_hash = self._get_item_hash(item)
            self._hash_to_id.pop(item_hash, None)
            
            # Add to free list
            self._free_ids = torch.cat([self._free_ids, torch.tensor([item_id], device=self.device)])
    
    def get_active_ids(self) -> torch.Tensor:
        """Get a tensor of all active item IDs."""
        return self._active_ids
    
    def __len__(self) -> int:
        """Get the number of unique items stored."""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all stored items from the buffer."""
        self._items.clear()
        self._hash_to_id.clear()
        self._active_ids = torch.tensor([], dtype=torch.long, device=self.device)
        self._free_ids = torch.tensor([], dtype=torch.long, device=self.device)
        self._next_id = 0
