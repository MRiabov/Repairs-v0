import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms.normalize_features import NormalizeFeatures
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add


class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__(aggr="sum", node_dim=0)
        self.heads = heads
        self.out_channels = out_channels

        # linear transformation for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention mechanism parameters
        # one parameter per head and out_channel
        self.att_targ = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_source = nn.Parameter(torch.Tensor(1, heads, out_channels))
        # att_targ - parameters of attention for target nodes (to which pass)
        # att_source - parameters of attention for source nodes (from which pass)
        # to compute attention in GAT they are multiplied with messages and summed; and softmax is applied.
        # To pass the messages forward the softmaxed attention is multiplied with messages and messages are aggregated by their attention weight (aggregated e.g. summed.)
        # (the aggregated messages are passed forward with `propagate` function consisting of - `message`, `aggregate` and `update`)

        # Optional bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        # ^note: # could be made nn.Linear but less efficient.

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_targ)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # Add self-loops to adjacency matrix
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.size(0))

        # apply linear transformation
        x = self.lin(x)  # [num_nodes, heads*out_channels]
        x = x.view(-1, self.heads, self.out_channels)  # num_nodes, heads, out_channels

        # start propagating messages (calls "message" and "aggregate")
        return self.propagate(edge_index, x=x)

    def message(
        self, x_i, x_j, index, ptr, size_i
    ):  # rename edge_index to index for softmax
        # x_i: target_node_features [num_edges]
        # x_j: features of source nodes [num_edges, out channels]
        # x_cat = torch.cat([x_i, x_j], dim=-1)

        # Compute attention scores (e^T * [Wh_i || Wh_j])
        alpha = (x_i * self.att_targ).sum(dim=-1) + (x_j * self.att_source).sum(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        # ^ ensure that even highly negative relationships are weighted less, but are still included (leaky relu.).
        alpha = softmax(alpha, index, ptr, size_i)  # note: PyG softmax on index
        # weight messages by attention.
        return x_j * alpha.unsqueeze(-1)  # [num_edges, heads, out_channels]

    def update(self, aggr_out):
        # aggregate across heads (mean or sum)
        aggr_out = aggr_out.mean(dim=1)  # [num_nodes, out_channels]
        aggr_out = aggr_out + self.bias
        return aggr_out


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads: int):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(num_features, hidden_dim, heads=heads)
        self.gat2 = GATLayer(hidden_dim, num_classes, heads=heads)
        self.learnable_empty = nn.Parameter(torch.zeros(num_classes))

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        if (
            x is None or x.shape[0] == 0
        ):  # note: sometimes there will be no graph, e.g. no electronics present. then, return a learnable parameter.
            return self.learnable_empty
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        out = self.gat2(x, edge_index)
        return out


class GraphEncoder(nn.Module):
    """Encode a graph with two-layer GAT and gated readout.

    The network first computes node embeddings with a two-layer GAT. Then it
    aggregates all node embeddings to a single graph-level embedding using a
    learnable gating mechanism similar to weighted mean pooling.
    """

    def __init__(self, num_features, hidden_dim, out_dim, heads: int):
        """Parameters
        ----------
        num_features : int
            Input feature dimension for each node.
        hidden_dim : int
            Hidden dimension used inside GAT layers.
        out_dim : int
            Dimension of the final graph-level embedding.
        heads : int
            Number of attention heads per GAT layer.
        """
        super(GraphEncoder, self).__init__()
        # Two-layer GAT encoder that returns node embeddings of size ``out_dim``.
        self.gat = GAT(num_features, hidden_dim, out_dim, heads=heads)
        # Gating network that outputs a scalar gate for every node.
        self.gate = nn.Linear(out_dim, 1)

    def forward(self, batch: Batch):
        """Return a gated, graph-level embedding.

        The method supports batched graphs (``data.batch`` attribute). If no
        batch information is provided, the input is assumed to contain a single
        graph.
        """
        # Node-level embeddings from the GAT encoder [N, out_dim]
        node_emb = self.gat(batch)

        # Compute scalar gates per node in range (0, 1).
        gates = torch.sigmoid(self.gate(node_emb))  # [N, 1]
        gated_emb = node_emb * gates  # [N, out_dim]

        # Weighted sum pooling followed by normalisation (weighted mean).
        num_graphs = batch.num_graphs
        if batch.batch.numel() == 0:
            # for debug: if no nodes to pool, return zeros instead.
            pooled = torch.zeros(
                (num_graphs, gated_emb.size(-1)),
                device=gated_emb.device,
                dtype=gated_emb.dtype,
            )
            norm = torch.ones(
                (num_graphs, 1), device=gated_emb.device, dtype=gated_emb.dtype
            )  # Or 1s to avoid div by zero
        else:
            pooled = scatter_add(gated_emb, batch.batch, dim=0, dim_size=num_graphs)
            norm = scatter_add(gates, batch.batch, dim=0, dim_size=num_graphs).clamp(
                min=1e-6
            )
        graph_emb = pooled / norm

        return graph_emb.squeeze(0)
