import torch
from typing import Dict, List, Optional, Tuple

from .singleton_buffer import SingletonBuffer


class GraphBuffer(SingletonBuffer[Dict[str, torch.Tensor]]):
    """Buffer that stores unique graphs represented as tensor dictionaries.
    
    Each graph is stored as a dictionary containing:
    - edge_index: Tensor of shape [2, num_edges] containing graph connectivity
    - node_features: Optional tensor of shape [num_nodes, node_feature_dim]
    - edge_features: Optional tensor of shape [num_edges, edge_feature_dim]
    
    The buffer uses tensor-based operations for efficient deduplication and management.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize the graph buffer.
        
        Args:
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        super().__init__(device)
    
    def _get_item_hash(self, graph: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Generate a hashable key for the graph for deduplication.
        
        The hash is based on the graph's connectivity and feature tensors.
        """
        edge_index = graph['edge_index']
        
        # Sort edges to make the hash order-invariant
        if edge_index.size(1) > 0:
            # Sort edges by source node, then by target node
            _, sorted_indices = torch.sort(edge_index[0] * (edge_index[0].max() + 1) + edge_index[1])
            edge_index = edge_index[:, sorted_indices]
        
        # Include node features in hash if present
        node_hash = graph.get('node_features')
        if node_hash is not None:
            # Flatten and sort node features for order-invariant hashing
            node_hash = node_hash.flatten().sort()[0]
        
        # Include edge features in hash if present
        edge_hash = graph.get('edge_features')
        if edge_hash is not None and edge_hash.size(0) > 0:
            if 'edge_index' in graph and edge_hash.size(0) == graph['edge_index'].size(1):
                # Sort edge features to match the sorted edges
                edge_hash = edge_hash[sorted_indices]
            edge_hash = edge_hash.flatten().sort()[0]
        
        # Create a tuple of tensors that uniquely identifies this graph
        hash_components = [edge_index.flatten()]
        if node_hash is not None:
            hash_components.append(node_hash)
        if edge_hash is not None:
            hash_components.append(edge_hash)
            
        return tuple(hash_components)
    
    def _prepare_item(self, graph: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare the graph for storage by ensuring all tensors are on the correct device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in graph.items()}
    
    def add(
        self,
        edge_index: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> int:
        """Add a graph to the buffer if not already present.
        
        Args:
            edge_index: Tensor of shape [2, num_edges] containing graph connectivity
            node_features: Optional tensor of shape [num_nodes, node_feature_dim]
            edge_features: Optional tensor of shape [num_edges, edge_feature_dim]
            **kwargs: Additional graph attributes to store
            
        Returns:
            int: The ID corresponding to the graph in the buffer
        """
        graph = {'edge_index': edge_index, **kwargs}
        if node_features is not None:
            graph['node_features'] = node_features
        if edge_features is not None:
            graph['edge_features'] = edge_features
            
        return super().add(graph)
    
    def get(self, graph_id: int) -> Dict[str, torch.Tensor]:
        """Retrieve a graph by its ID.
        
        Args:
            graph_id: The ID of the graph to retrieve
            
        Returns:
            Dict containing the graph's edge_index, node_features, and edge_features
            
        Raises:
            ValueError: If the graph ID is not found in the buffer
        """
        return super().get(graph_id)
    
    def get_batch(self, graph_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Retrieve multiple graphs and batch them together.
        
        Args:
            graph_ids: Tensor of graph IDs to retrieve (must be integer type)
            
        Returns:
            Batched graph data with the following keys:
            - edge_index: Stacked edge indices with appropriate offsets
            - node_features: Stacked node features (if present in input graphs)
            - edge_features: Stacked edge features (if present in input graphs)
            - batch: Batch indices for nodes
            - edge_batch: Batch indices for edges (if edge features are present)
            
        Raises:
            ValueError: If graph_ids contains non-integer values
        """
        if not isinstance(graph_ids, torch.Tensor):
            graph_ids = torch.tensor(graph_ids, dtype=torch.long, device=self.device)
        
        # Ensure we have integer indices
        if not torch.all(torch.eq(graph_ids, graph_ids.to(torch.long))):
            raise ValueError("graph_ids must contain integer values")
        graph_ids = graph_ids.to(torch.long)
        
        graphs: List[Dict[str, torch.Tensor]] = [self.get(int(gid.item())) for gid in graph_ids]
        
        # Initialize lists to store batched data
        edge_indices = []
        node_features = []
        edge_features = []
        node_offsets = [0]
        edge_offsets = [0]
        
        has_node_features = 'node_features' in graphs[0]
        has_edge_features = 'edge_features' in graphs[0]
        
        # Process each graph
        for graph in graphs:
            # Update edge indices with node offset
            edge_idx = graph['edge_index'].clone()
            if edge_idx.size(1) > 0:
                edge_idx[0, :] += node_offsets[-1]
                edge_idx[1, :] += node_offsets[-1]
            edge_indices.append(edge_idx)
            
            # Collect node and edge features
            if has_node_features:
                node_features.append(graph['node_features'])
            if has_edge_features and 'edge_features' in graph:
                edge_features.append(graph['edge_features'])
            
            # Update offsets
            node_offsets.append(node_offsets[-1] + (graph['node_features'].size(0) if has_node_features else 0))
            edge_offsets.append(edge_offsets[-1] + edge_idx.size(1))
        
        # Stack everything together
        result = {
            'edge_index': torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long, device=self.device),
            'batch': torch.repeat_interleave(
                torch.arange(len(graphs), device=self.device),
                torch.tensor([g.get('node_features', g['edge_index'].new_zeros((1,))).size(0) 
                             for g in graphs], device=self.device)
            )
        }
        
        if has_node_features:
            result['node_features'] = torch.cat(node_features, dim=0)
        if has_edge_features and edge_features:
            result['edge_features'] = torch.cat(edge_features, dim=0)
        
        return result