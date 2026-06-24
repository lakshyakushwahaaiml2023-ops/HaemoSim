"""
HaemoSim BloodFlowSolver Tests
Academic/Educational Demonstration Tool. Not for clinical use.
"""

from backend.solver import BloodFlowSolver

def test_solver_initialization():
    """
    Verify that the BloodFlowSolver class can be instantiated.
    """
    solver = BloodFlowSolver()
    assert solver is not None
