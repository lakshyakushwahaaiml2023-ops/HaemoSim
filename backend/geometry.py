"""
HaemoSim Vessel Geometry
Academic/Educational Demonstration Tool. Not for clinical use.

Generates 2D idealized vessel radius maps for healthy and diseased states
and calculates clinical stenosis metrics.
"""

import numpy as np

class VesselGeometry:
    """
    Generates 2D idealized vessel profiles (radius_map) representing different
    pathologies for blood flow simulation.
    """
    def __init__(self, Nx: int = 192, length: float = 0.08, radius_max: float = 0.004) -> None:
        self.Nx = Nx
        self.length = length
        self.radius_max = radius_max  # Healthy reference radius (4mm)
        self.x_coords = (np.arange(self.Nx) + 0.5) * (self.length / self.Nx)

    def healthy(self) -> np.ndarray:
        """
        Constant healthy vessel profile.
        """
        return np.ones(self.Nx) * self.radius_max

    def stenosis(self, severity: float = 0.6) -> np.ndarray:
        """
        Single smooth localized narrowing (stenosis) centered at the vessel midpoint.
        Uses a cosine-squared profile to ensure smooth transitions at boundaries.
        
        Parameters:
            severity (float): Radius reduction ratio, in [0, 1]. Default is 0.6 (60%).
        """
        radius_map = np.ones(self.Nx) * self.radius_max
        x_mid = self.length / 2.0
        width = 0.02  # 20mm stenosis region
        
        for i, x in enumerate(self.x_coords):
            if abs(x - x_mid) <= width / 2.0:
                reduction = self.radius_max * severity * (np.cos(np.pi * (x - x_mid) / width))**2
                radius_map[i] -= reduction
                
        return radius_map

    def aneurysm(self, dilation: float = 0.6) -> np.ndarray:
        """
        Single smooth localized expansion (aneurysm) centered at the vessel midpoint.
        Uses a cosine-squared profile.
        
        Parameters:
            dilation (float): Radius expansion ratio. Default is 0.6 (60% expansion).
        """
        radius_map = np.ones(self.Nx) * self.radius_max
        x_mid = self.length / 2.0
        width = 0.03  # 30mm aneurysm region
        
        for i, x in enumerate(self.x_coords):
            if abs(x - x_mid) <= width / 2.0:
                expansion = self.radius_max * dilation * (np.cos(np.pi * (x - x_mid) / width))**2
                radius_map[i] += expansion
                
        return radius_map

    def tandem_stenosis(self, severity1: float = 0.5, severity2: float = 0.4) -> np.ndarray:
        """
        Two consecutive narrowings in series to simulate complex flow disturbances.
        """
        radius_map = np.ones(self.Nx) * self.radius_max
        x1 = 0.03  # Center of first stenosis (30mm)
        x2 = 0.05  # Center of second stenosis (50mm)
        width = 0.015  # 15mm width for each bump
        
        for i, x in enumerate(self.x_coords):
            if abs(x - x1) <= width / 2.0:
                reduction = self.radius_max * severity1 * (np.cos(np.pi * (x - x1) / width))**2
                radius_map[i] -= reduction
            elif abs(x - x2) <= width / 2.0:
                reduction = self.radius_max * severity2 * (np.cos(np.pi * (x - x2) / width))**2
                radius_map[i] -= reduction
                
        return radius_map

def compute_area_reduction(radius_map: np.ndarray, r_ref: float = 0.004) -> dict:
    """
    Compute percentage stenosis using clinical conventions:
    - Diameter Stenosis (%): (1 - D_min / D_ref) * 100
    - Area Stenosis (%): (1 - A_min / A_ref) * 100 = (1 - (D_min / D_ref)^2) * 100
    
    Reflects the clinical standard terminology (e.g. NASCET / ECST classification).
    """
    r_min = np.min(radius_map)
    diameter_reduction = (1.0 - r_min / r_ref) * 100.0
    area_reduction = (1.0 - (r_min / r_ref)**2) * 100.0
    
    return {
        "diameter_stenosis_percent": diameter_reduction,
        "area_stenosis_percent": area_reduction
    }
