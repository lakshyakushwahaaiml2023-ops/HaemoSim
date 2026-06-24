"""
HaemoSim Physics-Informed Neural Network (PINN) Surrogate
Academic/Educational Demonstration Tool. Not for clinical use.

Implements a deep MLP with skip connections to learn blood velocity and pressure
fields. Enforces Navier-Stokes physical constraints using autograd.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ResidualBlock(nn.Module):
    """
    Residual feedforward block: 2 linear layers with Tanh activation and skip connection.
    """
    def __init__(self, neurons: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        return out + residual

class PINNSurrogate(nn.Module):
    """
    8-hidden-layer deep MLP with skip connections (residual blocks)
    predicting u, v, p from x, y, t, severity, heart_rate.
    """
    def __init__(self) -> None:
        super().__init__()
        # Input layer: 5 features
        self.fc_in = nn.Linear(5, 128)
        self.tanh = nn.Tanh()
        
        # 3 Residual blocks (6 layers)
        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)
        self.block3 = ResidualBlock(128)
        
        # Final hidden layer (1 layer)
        self.fc_out_hidden = nn.Linear(128, 128)
        
        # Output layer: 3 outputs (normalized u, v, p)
        self.fc_out = nn.Linear(128, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.tanh(self.fc_in(inputs))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.tanh(self.fc_out_hidden(out))
        return self.fc_out(out)

    def predict_physical(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                         severity: torch.Tensor, heart_rate: torch.Tensor) -> tuple:
        """
        Normalized inputs forward pass, returning outputs scaled back to physical units:
        - u, v in m/s
        - p in Pascals
        """
        # Coordinate ranges for normalization to [-1, 1]
        T_cycle = 60.0 / heart_rate
        x_norm = 2.0 * x / 0.08 - 1.0
        y_norm = y / 0.0048
        t_norm = 2.0 * t / T_cycle - 1.0
        severity_norm = 2.0 * severity / 0.75 - 1.0
        heart_rate_norm = 2.0 * (heart_rate - 50.0) / 50.0 - 1.0
        
        inputs = torch.cat([x_norm, y_norm, t_norm, severity_norm, heart_rate_norm], dim=1)
        outputs = self.forward(inputs)
        
        # Scale to physical range matching dataset profiles
        u = outputs[:, 0:1] * 0.15
        v = outputs[:, 1:2] * 0.05
        p = outputs[:, 2:3] * 100.0
        
        return u, v, p

def compute_physics_loss(model: PINNSurrogate, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
                         severity: torch.Tensor, heart_rate: torch.Tensor) -> torch.Tensor:
    """
    Computes the 2D incompressible Navier-Stokes momentum and continuity residuals
    using PyTorch autograd.
    """
    # Predict physical fields
    u, v, p = model.predict_physical(x, y, t, severity, heart_rate)
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    # Blood properties
    rho = 1060.0
    mu = 0.0035
    nu = mu / rho
    
    # Navier-Stokes residuals
    res_continuity = u_x + v_y
    res_x_momentum = u_t + u * u_x + v * u_y + (1.0 / rho) * p_x - nu * (u_xx + u_yy)
    res_y_momentum = v_t + u * v_x + v * v_y + (1.0 / rho) * p_y - nu * (v_xx + v_yy)
    
    loss = torch.mean(res_continuity**2) + torch.mean(res_x_momentum**2) + torch.mean(res_y_momentum**2)
    return loss

def sample_dataset_pool(filepath: str, n_samples: int, severity_indices: list) -> tuple:
    """
    Randomly samples fluid points from specified severity indices in the dataset.
    """
    data = np.load(filepath)
    u_all = data["u"]
    v_all = data["v"]
    p_all = data["p"]
    t_all = data["t"]
    is_fluid_all = data["is_fluid"]
    severities = data["severities"]
    heart_rates = data["heart_rates"]
    x_coords = data["x"]
    y_coords = data["y"]
    
    # Collect coordinates and values for fluid cells
    inputs = []
    outputs = []
    
    # To sample fast, pick random indices across the dimensions
    count = 0
    while count < n_samples:
        # Sample random coordinates
        s_idx = np.random.choice(severity_indices)
        h_idx = np.random.randint(0, len(heart_rates))
        t_idx = np.random.randint(0, 100)
        
        # Find fluid cells for this severity
        fluid_indices = np.where(is_fluid_all[s_idx])
        num_fluid = len(fluid_indices[0])
        if num_fluid == 0:
            continue
            
        # Select random fluid cell
        idx = np.random.randint(0, num_fluid)
        i = fluid_indices[0][idx]
        j = fluid_indices[1][idx]
        
        # Map to physical values
        x_val = x_coords[i]
        y_val = y_coords[j]
        t_val = t_all[s_idx, h_idx, t_idx]
        sev_val = severities[s_idx]
        hr_val = heart_rates[h_idx]
        
        u_val = u_all[s_idx, h_idx, t_idx, i, j]
        v_val = v_all[s_idx, h_idx, t_idx, i, j]
        p_val = p_all[s_idx, h_idx, t_idx, i, j]
        
        inputs.append([x_val, y_val, t_val, sev_val, hr_val])
        outputs.append([u_val, v_val, p_val])
        count += 1
        
    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)

def train_pinn(epochs: int = 1000, batch_size: int = 8000, lambda_physics: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "dataset.npz")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_dataset.py first.")
        
    print("Pre-sampling training and validation pools from dataset...")
    # Train on 10 severities, hold out index 3 and 7 for validation (generalization)
    train_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
    val_indices = [3, 7]
    
    X_train_raw, Y_train_raw = sample_dataset_pool(dataset_path, 400000, train_indices)
    X_val_raw, Y_val_raw = sample_dataset_pool(dataset_path, 80000, val_indices)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_raw, device=device)
    Y_train = torch.tensor(Y_train_raw, device=device)
    X_val = torch.tensor(X_val_raw, device=device)
    Y_val = torch.tensor(Y_val_raw, device=device)
    
    # Initialize model
    model = PINNSurrogate().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    train_data_losses = []
    train_phys_losses = []
    val_losses = []
    
    print("\nStarting PINN Surrogate training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Sample mini-batch
        idx = torch.randperm(X_train.shape[0], device=device)[:batch_size]
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]
        
        # Parse inputs for autograd physics loss
        x = X_batch[:, 0:1].clone().detach().requires_grad_(True)
        y = X_batch[:, 1:2].clone().detach().requires_grad_(True)
        t = X_batch[:, 2:3].clone().detach().requires_grad_(True)
        sev = X_batch[:, 3:4]
        hr = X_batch[:, 4:5]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # 1. Data loss
        u_pred, v_pred, p_pred = model.predict_physical(x, y, t, sev, hr)
        pred_fields = torch.cat([u_pred, v_pred, p_pred], dim=1)
        loss_data = torch.mean((pred_fields - Y_batch)**2)
        
        # 2. Physics loss (Autograd)
        loss_physics = compute_physics_loss(model, x, y, t, sev, hr)
        
        # Combined loss
        loss_total = loss_data + lambda_physics * loss_physics
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        
        # Record loss
        train_data_losses.append(loss_data.item())
        train_phys_losses.append(loss_physics.item())
        
        # Validation evaluation
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # Random batch from validation pool
                val_idx = torch.randperm(X_val.shape[0], device=device)[:20000]
                X_val_batch = X_val[val_idx]
                Y_val_batch = Y_val[val_idx]
                
                xv = X_val_batch[:, 0:1]
                yv = X_val_batch[:, 1:2]
                tv = X_val_batch[:, 2:3]
                sevv = X_val_batch[:, 3:4]
                hrv = X_val_batch[:, 4:5]
                
                up, vp, pp = model.predict_physical(xv, yv, tv, sevv, hrv)
                pred_v = torch.cat([up, vp, pp], dim=1)
                loss_val = torch.mean((pred_v - Y_val_batch)**2)
                val_losses.append(loss_val.item())
                
            print(f"Epoch {epoch}/{epochs} | Data Loss: {loss_data.item():.2e} | Phys Loss: {loss_physics.item():.2e} | Val Loss (Held-out): {loss_val.item():.2e}")
            
    # Save final model checkpoint
    checkpoint_path = os.path.join(project_dir, "pinn_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to {checkpoint_path}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds (~{elapsed/60:.1f} mins).")
    
    # Plotting training curves
    plt.figure(figsize=(10, 5))
    epochs_range = np.arange(1, epochs + 1)
    val_epochs = np.arange(50, epochs + 1, 50)
    if len(val_losses) > len(val_epochs):
        # include epoch 1
        val_epochs = np.insert(val_epochs, 0, 1)
        
    plt.semilogy(epochs_range, train_data_losses, 'b-', label='Train Data Loss')
    plt.semilogy(epochs_range, train_phys_losses, 'g--', label='Train Physics Loss')
    plt.semilogy(val_epochs, val_losses, 'r-o', label='Val Loss (Held-out geometries)')
    
    plt.title('PINN Surrogate Training History', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = os.path.join(project_dir, "pinn_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {plot_path}")

if __name__ == "__main__":
    train_pinn()
