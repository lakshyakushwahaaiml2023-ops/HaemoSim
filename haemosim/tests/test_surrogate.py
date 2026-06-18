import torch

from core.vessel_network import default_systemic_tree
from surrogate.fast_model import FastStateSurrogate
from surrogate.gnn_model import GNNSurrogate
from surrogate.train import batch_to_graph_tensors, mass_conservation_penalty


def test_gnn_surrogate_output_shapes_for_single_graph():
    network = default_systemic_tree()
    model = GNNSurrogate(network.graph, hidden_dim=16, num_layers=3)

    x_node = torch.randn(model.num_nodes, 3)
    x_edge = torch.randn(model.num_edges, 3)
    params = torch.randn(3)

    pressure_pred, flow_pred = model(x_node, x_edge, params)

    assert pressure_pred.shape == (model.num_nodes,)
    assert flow_pred.shape == (model.num_edges,)


def test_gnn_surrogate_output_shapes_for_batched_graphs():
    network = default_systemic_tree()
    model = GNNSurrogate(network.graph, hidden_dim=16, num_layers=3)
    batch_size = 4

    x_node = torch.randn(batch_size, model.num_nodes, 3)
    x_edge = torch.randn(batch_size, model.num_edges, 3)
    params = torch.randn(batch_size, 3)

    pressure_pred, flow_pred = model(x_node, x_edge, params)

    assert pressure_pred.shape == (batch_size, model.num_nodes)
    assert flow_pred.shape == (batch_size, model.num_edges)


def test_fast_surrogate_output_shapes_for_batched_graphs():
    network = default_systemic_tree()
    model = FastStateSurrogate(network.graph, hidden_dim=32, num_layers=2)
    batch_size = 4

    x_node = torch.randn(batch_size, model.num_nodes, 3)
    x_edge = torch.randn(batch_size, model.num_edges, 1)
    params = torch.randn(batch_size, 3)

    pressure_pred, flow_pred = model(x_node, x_edge, params)

    assert pressure_pred.shape == (batch_size, model.num_nodes)
    assert flow_pred.shape == (batch_size, model.num_edges)


def test_training_helpers_build_graph_batch_and_conservation_penalty():
    network = default_systemic_tree()
    model = GNNSurrogate(network.graph, hidden_dim=16, num_layers=3)

    class DatasetStub:
        param_keys = ("heart_rate", "compliance_scale", "peripheral_resistance")

    batch_size = 2
    n_nodes = model.num_nodes
    n_edges = model.num_edges
    n_params = len(DatasetStub.param_keys)
    features = torch.randn(batch_size, n_nodes + n_edges + n_params)
    targets = torch.randn(batch_size, n_nodes + n_edges)

    x_node, x_edge, params, target_pressure, target_flow = batch_to_graph_tensors(
        (features, targets), model, DatasetStub(), torch.device("cpu")
    )

    assert x_node.shape == (batch_size, n_nodes, 3)
    assert x_edge.shape == (batch_size, n_edges, 1)
    assert params.shape == (batch_size, n_params)
    assert target_pressure.shape == (batch_size, n_nodes)
    assert target_flow.shape == (batch_size, n_edges)

    zero_flow = torch.zeros(batch_size, n_edges)
    penalty = mass_conservation_penalty(zero_flow, model.edge_index, n_nodes)
    assert penalty.item() == 0.0
