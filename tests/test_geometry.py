"""
HaemoSim Vessel Geometry Tests
Academic/Educational Demonstration Tool. Not for clinical use.

Verifies the geometric profiles and clinical metric calculations.
"""

import numpy as np
from backend.geometry import VesselGeometry, compute_area_reduction

def test_vessel_geometries():
    Nx = 192
    L = 0.08
    R = 0.004
    geom = VesselGeometry(Nx=Nx, length=L, radius_max=R)
    
    # 1. Test healthy (constant radius)
    r_healthy = geom.healthy()
    assert r_healthy.shape == (Nx,)
    np.testing.assert_allclose(r_healthy, R)
    
    # 2. Test single stenosis (narrowing)
    severity = 0.6
    r_stenosis = geom.stenosis(severity=severity)
    assert r_stenosis.shape == (Nx,)
    np.testing.assert_allclose(np.min(r_stenosis), R * (1.0 - severity), rtol=1e-2)
    assert r_stenosis[0] == R
    assert r_stenosis[-1] == R
    
    # 3. Test aneurysm (expansion)
    dilation = 0.5
    r_aneurysm = geom.aneurysm(dilation=dilation)
    assert r_aneurysm.shape == (Nx,)
    np.testing.assert_allclose(np.max(r_aneurysm), R * (1.0 + dilation), rtol=1e-2)
    assert r_aneurysm[0] == R
    assert r_aneurysm[-1] == R
    
    # 4. Test tandem stenosis (two narrowings in series)
    r_tandem = geom.tandem_stenosis(severity1=0.5, severity2=0.3)
    assert r_tandem.shape == (Nx,)
    # Verify local minima around 30mm and 50mm
    idx_1 = np.argmin(r_tandem[:Nx//2])
    idx_2 = Nx//2 + np.argmin(r_tandem[Nx//2:])
    np.testing.assert_allclose(r_tandem[idx_1], R * 0.5, rtol=1e-2)
    np.testing.assert_allclose(r_tandem[idx_2], R * 0.7, rtol=1e-2)

def test_metrics():
    # Healthy vessel has 0% reduction
    r_healthy = np.ones(100) * 0.004
    metrics = compute_area_reduction(r_healthy, r_ref=0.004)
    assert metrics["diameter_stenosis_percent"] == 0.0
    assert metrics["area_stenosis_percent"] == 0.0
    
    # 60% diameter reduction -> 84% area reduction
    r_sten = np.ones(100) * 0.004
    r_sten[50] = 0.0016  # 4mm * (1 - 0.6) = 1.6mm
    metrics_sten = compute_area_reduction(r_sten, r_ref=0.004)
    np.testing.assert_allclose(metrics_sten["diameter_stenosis_percent"], 60.0)
    np.testing.assert_allclose(metrics_sten["area_stenosis_percent"], 84.0)
