import torch
from typing import List, Optional, Sequence, cast
from torch_geometric.data import Batch, Data


class GraphBuffer:
    """Fixed-size buffer that stores small PyG graphs densely and references them by ID.

    Assumptions (configurable via ``max_nodes`` / ``max_edges``):
    • each graph has at most *max_nodes* nodes and *max_edges* edges
    • node-feature dimensionality is the same for every stored graph

    Internally we keep two dense tensors:
        edge_index : (B, 2, max_edges)    -1-padded
        node_feat  : (B, max_nodes, F)    -0-padded
    plus ``num_nodes`` / ``num_edges`` vectors so we know the true sizes.
    ``used_mask`` functions like the one in SparseVoxelBuffer to locate free
    rows quickly.
    """

    def __init__(
        self,
        buffer_size: int = 1024,
        max_nodes: int = 16,
        max_edges: int = 10,
        node_feat_dim: Optional[int] = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        values_dtype=torch.float16,
    ) -> None:
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.buffer_size = buffer_size
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_feat_dim = node_feat_dim  # can be inferred on first add

        # Dense storage -----------------------------------------------------
        self._edge_index = torch.full(
            (buffer_size, 2, max_edges), -1, dtype=torch.long, device=self.device
        )  # ideally shorten this to int8.
        self._node_feat: Optional[torch.Tensor] = None
        if node_feat_dim is not None:
            self._node_feat = torch.zeros(
                (buffer_size, max_nodes, node_feat_dim),
                device=self.device,
                dtype=values_dtype,
            )

        self._num_nodes = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self._num_edges = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self._used_mask = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _allocate_rows(self, count: int) -> torch.Tensor:
        free = torch.nonzero(~self._used_mask, as_tuple=False).flatten()
        assert free.numel() >= count, (
            f"GraphBuffer full: need {count} free slots, have {free.numel()}"
        )
        rows = free[:count]
        self._used_mask[rows] = True
        return rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, batch: Batch) -> torch.Tensor:
        """Insert a *Batch* of graphs and return their row IDs."""
        data_list: List[Data] = batch.to_data_list()
        if batch.batch.numel() == 0:
            # Handle empty batch: return empty tensor of correct type and device
            return torch.tensor([-1], dtype=torch.long, device=self.device)
        rows = self._allocate_rows(len(data_list))

        # Infer node feature dimension on first add if needed
        if self._node_feat is None:
            first_x = getattr(data_list[0], "x", None)
            feat_dim = first_x.size(-1) if first_x is not None else 0
            self.node_feat_dim = feat_dim
            self._node_feat = torch.zeros(
                (self.buffer_size, self.max_nodes, feat_dim), device=self.device
            )

        for buf_idx, data in zip(rows.tolist(), data_list):
            # --- Sanity checks --------------------------------------------
            e_idx = data.edge_index.to(self.device)
            raw_x = getattr(data, "x", None)
            n_feat = raw_x.to(self.device) if raw_x is not None else None
            assert e_idx.size(1) <= self.max_edges, ValueError(
                "Graph has more edges than max_edges"
            )
            if n_feat is not None and n_feat.size(0) > 0:
                assert n_feat.size(0) <= self.max_nodes, ValueError(
                    "Graph has more nodes than max_nodes"
                )
                assert n_feat.size(1) == self.node_feat_dim, ValueError(
                    "Node-feature dimension mismatch across graphs"
                )

            # --- Store edge_index -----------------------------------------
            self._edge_index[buf_idx].fill_(-1)
            self._edge_index[buf_idx, :, : e_idx.size(1)] = e_idx
            self._num_edges[buf_idx] = e_idx.size(1)

            # --- Store node features --------------------------------------
            self._num_nodes[buf_idx] = n_feat.size(0) if n_feat is not None else 0
            if n_feat is not None and n_feat.size(0) > 0:
                self._node_feat[buf_idx].zero_()
                self._node_feat[buf_idx, : n_feat.size(0)] = n_feat

        return rows.to(torch.long)

    def get(self, ids: Sequence[int] | torch.Tensor) -> Batch:
        """Fetch graphs by ID as a PyG *Batch*."""
        if isinstance(ids, torch.Tensor):
            ids_list = ids.tolist()
        else:
            ids_list = list(ids)
        datas: List[Data] = []
        for idx in ids_list:
            if idx == -1:
                datas.append(Data())
                continue
            assert self._used_mask[idx], f"Slot {idx} is empty"
            e_cnt = int(self._num_edges[idx].item())
            n_cnt = int(self._num_nodes[idx].item())
            edge_index = self._edge_index[idx, :, :e_cnt].clone()
            x = None
            if self._node_feat is not None and n_cnt > 0:
                x = self._node_feat[idx, :n_cnt].clone()
            datas.append(Data(x=x, edge_index=edge_index))
        return Batch.from_data_list(datas)

    def cleanup(self, active_ids: torch.Tensor | Sequence[int]):
        """Mark only *active_ids* as used, freeing everything else."""
        if isinstance(active_ids, torch.Tensor):
            active_set = set(int(i) for i in active_ids.tolist())
        else:
            active_set = set(int(i) for i in active_ids)
        for idx in range(self.buffer_size):
            if self._used_mask[idx] and idx not in active_set:
                self._used_mask[idx] = False
                self._num_nodes[idx] = 0
                self._num_edges[idx] = 0
                self._edge_index[idx].fill_(-1)
                if self._node_feat is not None:
                    self._node_feat[idx].zero_()

    # Convenience ---------------------------------------------------------
    def __len__(self) -> int:
        return int(self._used_mask.sum().item())
