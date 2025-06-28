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
        node_feat_dim: int,
        store_global_feat: bool,
        buffer_size: int = 1024,
        max_nodes: int = 16,
        max_edges: int = 10,
        max_globals: int = 12,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        values_dtype=torch.float16,
    ) -> None:
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self.buffer_size = buffer_size
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_globals = max_globals
        self.node_feat_dim = node_feat_dim
        self.store_global_feat = store_global_feat
        if store_global_feat:
            self.follow_batch = ["global_feat"]
        else:
            self.follow_batch = []

        self._node_feat = torch.zeros(
            (buffer_size, max_nodes, node_feat_dim),
            device=self.device,
            dtype=values_dtype,
        )

        # Dense storage -----------------------------------------------------
        self._edge_index = torch.full(
            (buffer_size, 2, max_edges), -1, dtype=torch.long, device=self.device
        )  # ideally shorten this to int8.
        self._num_nodes = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self._num_edges = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self._used_mask = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)

        # Optional global features ---------------------------------------
        if self.store_global_feat:
            self._global_feat = torch.zeros(
                (buffer_size, max_globals, node_feat_dim),
                device=self.device,
                dtype=values_dtype,
            )
            self._num_globals = torch.zeros(
                buffer_size, dtype=torch.long, device=self.device
            )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _allocate_row(self) -> torch.Tensor:
        free = torch.nonzero(~self._used_mask, as_tuple=False).flatten()
        assert free.numel() >= 1, "Can't allocate row: GraphBuffer is full"
        row = free[0]
        self._used_mask[row] = True
        return row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, data: Data | Batch) -> torch.Tensor:
        """Insert one or many graphs and return the **row IDs** that were
        allocated for them.

        The method now understands both individual ``Data`` objects and a
        :class:`torch_geometric.data.Batch`.  When a batch is supplied we
        split it via :py:meth:`Batch.to_data_list` and store each graph in a
        separate row.  The returned tensor therefore has ``len(data)``
        elements (or ``1`` for a single ``Data`` instance).
        """

        # ------------------------------------------------------------------
        # Normalise to a list[Data] -----------------------------------------
        # ------------------------------------------------------------------
        if isinstance(data, Batch):
            graphs: List[Data] = data.to_data_list()
        else:
            graphs = [data]

        num_graphs = len(graphs)
        if num_graphs == 0:
            # Empty batch → nothing to add, return [-1]
            return torch.tensor([-1], dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------
        # Determine how many *non-empty* graphs require storage --------------
        # ------------------------------------------------------------------
        needs_row = [g.x is not None and g.x.size(0) > 0 for g in graphs]
        n_required = sum(needs_row)

        # ------------------------------------------------------------------
        # Ensure there is enough free capacity ------------------------------
        # ------------------------------------------------------------------
        if n_required > 0:
            free_rows = torch.nonzero(~self._used_mask, as_tuple=False).flatten()
            assert free_rows.numel() >= n_required, "GraphBuffer is full"

        row_ids = torch.full((num_graphs,), -1, dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------
        # Store graphs that are not empty -----------------------------------
        # ------------------------------------------------------------------
        for idx, g in enumerate(graphs):
            if not needs_row[idx]:
                # empty graph → leave row_ids[idx] == -1
                continue

            assert g.edge_index is not None, "Graph is missing edge_index"
            assert g.edge_index.size(1) <= self.max_edges, "Graph has too many edges"
            assert g.x.size(0) <= self.max_nodes, "Graph has too many nodes"

            row = self._allocate_row()
            row_ids[idx] = row

            # Move tensors to buffer device once to avoid implicit copies
            g = g.to(self.device)

            # Edge index ----------------------------------------------------
            self._edge_index[row].fill_(-1)
            if g.edge_index.numel() > 0:
                self._edge_index[row, :, : g.edge_index.size(1)] = g.edge_index
            self._num_edges[row] = g.edge_index.size(1)

            # Node features -------------------------------------------------
            self._node_feat[row].zero_()
            if g.x.numel() > 0:
                self._node_feat[row, : g.x.size(0)] = g.x
            self._num_nodes[row] = g.x.size(0)

            # Optional global features -------------------------------------
            if self.store_global_feat:
                assert hasattr(g, "global_feat") and g.global_feat is not None, (
                    "Graph has no global features"
                )
                gf = g.global_feat
                assert gf.size(0) <= self.max_globals, "Too many global features"
                self._global_feat[row].zero_()
                self._global_feat[row, : gf.size(0)] = gf
                self._num_globals[row] = gf.size(0)

        return row_ids

    def get(self, ids: torch.Tensor) -> Batch:
        """Fetch graphs by ID as a PyG *Batch*.

        The original implementation instantiated an intermediate
        :class:`torch_geometric.data.Data` object for every requested graph
        and then called :meth:`Batch.from_data_list`.  While simple, that
        approach involves a Python-level loop and a fair amount of small
        allocations.  This version constructs the resulting batch in a
        *single* pass:

        1.  We pre-compute the total number of nodes and edges across the
            requested graph IDs and allocate the final `x` and
            `edge_index` tensors once.
        2.  We copy the per-graph slices into those tensors while
            shifting the edge indices on-the-fly and filling in the
            ``batch`` vector.
        3.  Finally we wrap the tensors into a :class:`~torch_geometric.data.Batch`.

        An ID of ``-1`` is treated as padding and yields an empty graph so
        that the positional correspondence between ``ids`` and the
        returned graphs remains intact.
        """
        # Ensure device placement
        ids = ids.to(self.device)

        # ------------------------------------------------------------------
        # Filter out padding IDs (-1) --------------------------------------
        # ------------------------------------------------------------------
        valid_mask = ids >= 0  # (B,)
        valid_ids = ids[valid_mask]  # (G,) where G <= B
        if valid_ids.numel() == 0:
            # Request only had padding → return *empty* Batch
            return Batch()

        # ------------------------------------------------------------------
        # Gather per-graph sizes (vectorised) -------------------------------
        # ------------------------------------------------------------------
        node_counts = self._num_nodes[valid_ids]  # (G,)
        edge_counts = self._num_edges[valid_ids]  # (G,)

        # Offsets so that edges get shifted by #previous nodes
        node_offsets = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.device, dtype=torch.long), node_counts[:-1]]
            ),
            dim=0,
        )  # (G,)

        total_nodes = int(node_counts.sum())
        total_edges = int(edge_counts.sum())

        # ------------------------------------------------------------------
        # Build *x* (node features) in one go ------------------------------
        # ------------------------------------------------------------------
        # Mask to select only *real* nodes from the padded storage
        node_idx = (
            torch.arange(self.max_nodes, device=self.device)
            .unsqueeze(0)
            .expand(valid_ids.size(0), self.max_nodes)
        ) < node_counts.unsqueeze(1)
        # (G, max_nodes) boolean

        x_all = self._node_feat[valid_ids][node_idx]  # (total_nodes, F)

        # Batch vector: repeat each graph idx by its node count.
        batch_vec = torch.repeat_interleave(
            torch.arange(valid_ids.size(0), device=self.device), node_counts
        )  # (total_nodes,)

        # ------------------------------------------------------------------
        # Build *edge_index* in one go -------------------------------------
        # ------------------------------------------------------------------
        edge_idx_mask = (
            torch.arange(self.max_edges, device=self.device)
            .unsqueeze(0)
            .expand(valid_ids.size(0), self.max_edges)
        ) < edge_counts.unsqueeze(1)  # (G, max_edges)

        edges_padded = self._edge_index[valid_ids].permute(0, 2, 1)  # (G, max_edges, 2)
        edges_all = edges_padded[edge_idx_mask]  # (total_edges, 2)

        # Shift edges by node offsets so that they reference *x_all*
        edge_offsets = torch.repeat_interleave(node_offsets, edge_counts)
        edges_all = (edges_all + edge_offsets.unsqueeze(1)).t().contiguous()  # (2,E)

        # ------------------------------------------------------------------
        # Wrap into PyG Batch ----------------------------------------------
        # ------------------------------------------------------------------
        kwargs = dict(
            batch=batch_vec,
            x=x_all,
            edge_index=edges_all,
            num_graphs=valid_ids.numel(),
        )
        if self.store_global_feat:
            kwargs["global_feat"] = self._global_feat[valid_ids]
        batch = Batch(**kwargs)

        # Compact; padded ids ignored
        return batch

    def cleanup(self, active_ids: torch.Tensor):
        """Mark only *active_ids* as used, freeing everything else."""
        active_set = torch.unique(active_ids).to(self.device)
        self._used_mask = torch.zeros_like(self._used_mask)
        self._used_mask[active_set] = True
        self._num_nodes[active_set] = 0
        self._num_edges[active_set] = 0
        self._edge_index[active_set].fill_(-1)
        self._node_feat[active_set] = torch.zeros_like(self._node_feat[active_set])
        if self.store_global_feat:
            self._global_feat[active_set] = torch.zeros_like(
                self._global_feat[active_set]
            )
            self._num_globals[active_set] = 0

    # Convenience ---------------------------------------------------------
    def __len__(self) -> int:
        return int(self._used_mask.sum().item())
