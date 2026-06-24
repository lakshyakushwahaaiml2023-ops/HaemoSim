"""
HaemoSim PINN Module Tests
Academic/Educational Demonstration Tool. Not for clinical use.

Verifies the neural network architecture, forward pass scaling, and autograd physical loss computations.
"""

import numpy as np
import torch
from backend.pinn import PINNSurrogate, compute_physics_loss

def test_pinn_forward_and_physics_loss():
    """
    Verify that PINNSurrogate correctly instantiates, scales output shapes,
    and computes the autograd Navier-Stokes physics loss without NaN/Inf.
    """
    model = PINNSurrogate()
    
    # Mock batch of 10 coordinates
    x = torch.linspace(0.0, 0.08, 10).view(-1, 1).requires_grad_(True)
    y = torch.linspace(-0.004, 0.004, 10).view(-1, 1).requires_grad_(True)
    t = torch.linspace(0.0, 0.83, 10).view(-1, 1).requires_grad_(True)
    severity = torch.ones(10, 1) * 0.5
    heart_rate = torch.ones(10, 1) * 72.0
    
    # 1. Test physical prediction scaling
    u, v, p = model.predict_physical(x, y, t, severity, heart_rate)
    assert u.shape == (10, 1)
    assert v.shape == (10, 1)
    assert p.shape == (10, 1)
    
    # 2. Test physics loss gradient computation
    loss = compute_physics_loss(model, x, y, t, severity, heart_rate)
    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() >= 0.0
