import math

from core.vessel_network import VesselNetwork, default_systemic_tree


def test_segment_resistance_is_computed_from_poiseuille_law():
    network = VesselNetwork(blood_viscosity=0.0035)

    network.add_segment("a", "b", length=0.2, radius=0.01, wall_compliance=1.0)

    segment = network.graph["a"]["b"]
    expected = 8.0 * 0.0035 * 0.2 / (math.pi * 0.01**4)
    assert math.isclose(segment["resistance"], expected)


def test_default_systemic_tree_has_expected_topology():
    network = default_systemic_tree()

    assert len(network.graph["aorta"]) == 4
    assert len(network.outlet_nodes) == 8
    assert sum(len(children) for children in network.graph.values()) == 12


def test_steady_solution_conserves_mass_at_internal_nodes():
    network = default_systemic_tree()

    results = network.solve()

    for node in network.nodes:
        if node == network.inlet_node or node in network.outlet_nodes:
            continue

        flows = results["flows"][node]
        assert math.isclose(flows["in"], flows["out"], rel_tol=1e-9, abs_tol=1e-12)


def test_transient_solution_reports_storage_as_internal_net_flow():
    network = default_systemic_tree()

    results = network.solve(dt=0.01)

    for node in network.nodes:
        if node == network.inlet_node or node in network.outlet_nodes:
            continue

        flows = results["flows"][node]
        assert math.isfinite(flows["net"])
