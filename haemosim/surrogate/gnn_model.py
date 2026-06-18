import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class GNNSurrogate(nn.Module):
    """
    Fixed-topology GNN surrogate for haemodynamic state prediction.

    The vascular graph is a dict-of-dicts adjacency structure where each edge
    stores resistance, length, radius, and wall_compliance.
    """

    def __init__(self, graph, hidden_dim=128, num_layers=3):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.graph = graph
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.node_names, edge_index, edge_attr = self._graph_tensors(graph)
        self.edge_names = self._edge_names(graph, self.node_names)
        self.num_nodes = len(self.node_names)
        self.num_edges = len(self.edge_names)

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("base_edge_attr", edge_attr)
        self.register_buffer("node_compliance", self._node_compliance(graph))

        self.node_encoder = nn.LazyLinear(self.hidden_dim)
        self.edge_encoder = nn.LazyLinear(self.hidden_dim)
        self.param_encoder = nn.LazyLinear(self.hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                GATv2Conv(
                    self.hidden_dim,
                    self.hidden_dim,
                    heads=1,
                    edge_dim=self.hidden_dim,
                    add_self_loops=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.activation = nn.ReLU()
        self.node_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x_node, x_edge, params):
        """
        Predict next-state node pressures and edge flows.

        x_node shape: [N, F_node] or [B, N, F_node]
        x_edge shape: [E, F_edge] or [B, E, F_edge]
        params shape: [F_param] or [B, F_param]
        """
        batched = x_node.dim() == 3
        if not batched:
            x_node = x_node.unsqueeze(0)
            x_edge = x_edge.unsqueeze(0)
            params = params.unsqueeze(0) if params.dim() == 1 else params

        pressure_preds = []
        flow_preds = []
        for batch_index in range(x_node.shape[0]):
            pressure_pred, flow_pred = self._forward_single(
                x_node[batch_index], x_edge[batch_index], params[batch_index]
            )
            pressure_preds.append(pressure_pred)
            flow_preds.append(flow_pred)

        pressure_preds = torch.stack(pressure_preds, dim=0)
        flow_preds = torch.stack(flow_preds, dim=0)

        if not batched:
            return pressure_preds.squeeze(0), flow_preds.squeeze(0)
        return pressure_preds, flow_preds

    def _forward_single(self, x_node, x_edge, params):
        if x_node.shape[0] != self.num_nodes:
            raise ValueError("x_node first dimension must match graph node count")
        if x_edge.shape[0] != self.num_edges:
            raise ValueError("x_edge first dimension must match graph edge count")

        device = x_node.device
        edge_index = self.edge_index.to(device)
        base_edge_attr = self.base_edge_attr.to(device)
        node_compliance = self.node_compliance.to(device).unsqueeze(-1)

        node_input = torch.cat([x_node, node_compliance], dim=-1)
        edge_input = torch.cat([x_edge, base_edge_attr], dim=-1)
        param_context = self.param_encoder(params).unsqueeze(0)

        h = self.node_encoder(node_input) + param_context
        edge_h = self.edge_encoder(edge_input)

        for layer in self.message_layers:
            h = self.activation(layer(h, edge_index, edge_h) + h)

        pressure_pred = self.node_decoder(h).squeeze(-1)
        src, dst = edge_index
        edge_state = torch.cat([h[src], h[dst], edge_h], dim=-1)
        flow_pred = self.edge_decoder(edge_state).squeeze(-1)
        return pressure_pred, flow_pred

    @staticmethod
    def _edge_names(graph, node_names):
        edges = []
        for start in node_names:
            for end in graph.get(start, {}):
                edges.append((start, end))
        return edges

    @classmethod
    def _graph_tensors(cls, graph):
        node_names = cls._topological_nodes(graph)
        node_index = {node: i for i, node in enumerate(node_names)}
        edge_names = cls._edge_names(graph, node_names)

        if not edge_names:
            raise ValueError("graph must contain at least one edge")

        edge_index = torch.tensor(
            [[node_index[start], node_index[end]] for start, end in edge_names],
            dtype=torch.long,
        ).t().contiguous()
        edge_attr = torch.tensor(
            [
                [
                    graph[start][end]["resistance"],
                    graph[start][end]["length"],
                    graph[start][end]["radius"],
                ]
                for start, end in edge_names
            ],
            dtype=torch.float32,
        )
        return node_names, edge_index, edge_attr

    @classmethod
    def _node_compliance(cls, graph):
        node_names = cls._topological_nodes(graph)
        values = {node: 0.0 for node in node_names}
        for start, children in graph.items():
            for end, segment in children.items():
                half = 0.5 * segment.get("wall_compliance", 0.0)
                values[start] += half
                values[end] += half
        return torch.tensor([values[node] for node in node_names], dtype=torch.float32)

    @staticmethod
    def _topological_nodes(graph):
        nodes = set(graph)
        for children in graph.values():
            nodes.update(children)

        indegree = {node: 0 for node in nodes}
        for children in graph.values():
            for child in children:
                indegree[child] += 1

        queue = [node for node, degree in indegree.items() if degree == 0]
        ordered = []
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            for child in graph.get(node, {}):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(nodes):
            raise ValueError("graph must be acyclic")
        return ordered
