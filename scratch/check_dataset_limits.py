import numpy as np
import os

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "dataset.npz")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    data = np.load(dataset_path)
    u = data["u"]
    v = data["v"]
    p = data["p"]
    
    print(f"Dataset shapes: u={u.shape}, v={v.shape}, p={p.shape}")
    print(f"u: min = {u.min():.5f}, max = {u.max():.5f}, mean = {u.mean():.5f}")
    print(f"v: min = {v.min():.5f}, max = {v.max():.5f}, mean = {v.mean():.5f}")
    print(f"p: min = {p.min():.5f}, max = {p.max():.5f}, mean = {p.mean():.5f}")

if __name__ == "__main__":
    main()
