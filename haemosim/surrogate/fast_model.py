import torch
from torch import nn

try:
    from haemosim.surrogate.gnn_model import GNNSurrogate
except ImportError:
    from surrogate.gnn_model import GNNSurrogate


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class FastStateSurrogate(nn.Module):
    """
    Fast fixed-topology surrogate for short k-step haemodynamic prediction.

    This avoids PyG message-passing overhead by flattening the small fixed graph
    state, while still exposing node/edge predictions and graph metadata.
    """

    def __init__(self, graph, hidden_dim=256, num_layers=4, dropout=0.05):
        super().__init__()
        graph_info = GNNSurrogate(graph, hidden_dim=8, num_layers=1)
        self.graph = graph
        self.node_names = graph_info.node_names
        self.edge_names = graph_info.edge_names
        self.num_nodes = graph_info.num_nodes
        self.num_edges = graph_info.num_edges
        self.register_buffer("edge_index", graph_info.edge_index)

        self.input_proj = nn.LazyLinear(hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_nodes + self.num_edges),
        )

    def forward(self, x_node, x_edge, params):
        batched = x_node.dim() == 3
        if not batched:
            x_node = x_node.unsqueeze(0)
            x_edge = x_edge.unsqueeze(0)
            params = params.unsqueeze(0) if params.dim() == 1 else params

        flat = torch.cat(
            [
                x_node.flatten(start_dim=1),
                x_edge.flatten(start_dim=1),
                params,
            ],
            dim=-1,
        )
        h = self.input_proj(flat)
        h = self.blocks(h)
        output = self.decoder(h)
        pressure_pred = output[:, : self.num_nodes]
        flow_pred = output[:, self.num_nodes :]

        if not batched:
            return pressure_pred.squeeze(0), flow_pred.squeeze(0)
        return pressure_pred, flow_pred
