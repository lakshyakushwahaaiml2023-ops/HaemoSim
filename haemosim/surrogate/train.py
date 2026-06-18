import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from haemosim.core.vessel_network import default_systemic_tree
    from haemosim.data.dataset import HaemodynamicsDataset
    from haemosim.surrogate.fast_model import FastStateSurrogate
    from haemosim.surrogate.gnn_model import GNNSurrogate
except ImportError:
    from core.vessel_network import default_systemic_tree
    from data.dataset import HaemodynamicsDataset
    from surrogate.fast_model import FastStateSurrogate
    from surrogate.gnn_model import GNNSurrogate


def batch_to_graph_tensors(batch, model, dataset, device):
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device)

    n_nodes = model.num_nodes
    n_edges = model.num_edges
    n_params = len(dataset.param_keys)

    pressures = features[:, :n_nodes]
    edge_flows = features[:, n_nodes : n_nodes + n_edges]
    params = features[:, n_nodes + n_edges : n_nodes + n_edges + n_params]

    incoming = torch.zeros_like(pressures)
    outgoing = torch.zeros_like(pressures)
    edge_index = model.edge_index.to(device)
    for edge_id in range(n_edges):
        src = edge_index[0, edge_id]
        dst = edge_index[1, edge_id]
        outgoing[:, src] = outgoing[:, src] + edge_flows[:, edge_id]
        incoming[:, dst] = incoming[:, dst] + edge_flows[:, edge_id]

    x_node = torch.stack([pressures, incoming, outgoing], dim=-1)
    x_edge = edge_flows.unsqueeze(-1)
    target_pressure = targets[:, :n_nodes]
    target_flow = targets[:, n_nodes : n_nodes + n_edges]
    return x_node, x_edge, params, target_pressure, target_flow


def mass_conservation_penalty(flow_pred, edge_index, n_nodes):
    net_flow = torch.zeros(
        flow_pred.shape[0],
        n_nodes,
        device=flow_pred.device,
        dtype=flow_pred.dtype,
    )
    for edge_id in range(flow_pred.shape[1]):
        src = edge_index[0, edge_id]
        dst = edge_index[1, edge_id]
        net_flow[:, src] = net_flow[:, src] - flow_pred[:, edge_id]
        net_flow[:, dst] = net_flow[:, dst] + flow_pred[:, edge_id]
    return torch.mean(net_flow**2)


def compute_loss(model, batch, dataset, device):
    x_node, x_edge, params, target_pressure, target_flow = batch_to_graph_tensors(
        batch, model, dataset, device
    )
    pressure_pred, flow_pred = model(x_node, x_edge, params)
    pressure_loss = F.mse_loss(pressure_pred, target_pressure)
    flow_loss = F.mse_loss(flow_pred, target_flow)
    conservation_loss = mass_conservation_penalty(
        flow_pred, model.edge_index.to(device), model.num_nodes
    )
    total = pressure_loss + 0.1 * flow_loss + 0.05 * conservation_loss
    return total, pressure_loss, flow_loss


def run_epoch(model, loader, dataset, device, optimizer=None, epoch=None, log_every=0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    pressure_loss = 0.0
    flow_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            loss, p_loss, f_loss = compute_loss(model, batch, dataset, device)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pressure_loss += p_loss.item()
            flow_loss += f_loss.item()
            n_batches += 1

            if training and log_every and batch_index % log_every == 0:
                print(
                    f"epoch {epoch:03d} "
                    f"batch {batch_index}/{len(loader)} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

    scale = max(n_batches, 1)
    return total_loss / scale, pressure_loss / scale, flow_loss / scale


def evaluate_rmse(model, loader, dataset, device):
    model.eval()
    pressure_sse = 0.0
    flow_sse = 0.0
    pressure_count = 0
    flow_count = 0

    with torch.no_grad():
        for batch in loader:
            x_node, x_edge, params, target_pressure, target_flow = (
                batch_to_graph_tensors(batch, model, dataset, device)
            )
            pressure_pred, flow_pred = model(x_node, x_edge, params)
            pressure_sse += torch.sum((pressure_pred - target_pressure) ** 2).item()
            flow_sse += torch.sum((flow_pred - target_flow) ** 2).item()
            pressure_count += target_pressure.numel()
            flow_count += target_flow.numel()

    pressure_rmse = (pressure_sse / max(pressure_count, 1)) ** 0.5
    flow_rmse = (flow_sse / max(flow_count, 1)) ** 0.5
    return pressure_rmse, flow_rmse


def train(args):
    device = torch.device(args.device)
    print(f"loading dataset from {args.data}", flush=True)
    train_dataset, val_dataset, test_dataset = HaemodynamicsDataset.create_splits(
        args.data,
        k=args.k,
        seed=args.seed,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    print(
        f"loaded examples: train={len(train_dataset)} "
        f"val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )
    print(f"building {args.model} surrogate on {device}", flush=True)
    graph = default_systemic_tree().graph
    if args.model == "gnn":
        model = GNNSurrogate(
            graph,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
    else:
        model = FastStateSurrogate(
            graph,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pressure_loss, train_flow_loss = run_epoch(
            model,
            train_loader,
            train_dataset,
            device,
            optimizer=optimizer,
            epoch=epoch,
            log_every=args.log_every,
        )
        val_loss, val_pressure_loss, val_flow_loss = run_epoch(
            model, val_loader, val_dataset, device
        )
        scheduler.step()

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_pressure_mse={train_pressure_loss:.6f} "
            f"train_flow_mse={train_flow_loss:.6f} "
            f"val_pressure_mse={val_pressure_loss:.6f} "
            f"val_flow_mse={val_flow_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "feature_mean": train_dataset.feature_mean,
                    "feature_std": train_dataset.feature_std,
                    "param_keys": train_dataset.param_keys,
                    "node_names": model.node_names,
                    "edge_names": model.edge_names,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "model": args.model,
                    "k": args.k,
                },
                checkpoint_path,
            )
            print(f"saved best checkpoint to {checkpoint_path}", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break

    print("loading best checkpoint and evaluating test split", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    pressure_rmse, flow_rmse = evaluate_rmse(model, test_loader, test_dataset, device)
    print(f"test_pressure_rmse={pressure_rmse:.6f}", flush=True)
    print(f"test_flow_rmse={flow_rmse:.6f}", flush=True)


def _build_parser():
    parser = argparse.ArgumentParser(description="Train the HaemoSim surrogate.")
    parser.add_argument("--data", required=True, help="Path to trajectories HDF5.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--model",
        choices=("fast", "gnn"),
        default="fast",
        help="Use fast dense surrogate by default; gnn keeps the PyG GATv2 model.",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Use every Nth time window. Default 10 is meant for in-between-frame use.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Cap total examples loaded from HDF5. Use 0 for all examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training batch progress every N batches. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to save the best checkpoint.",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.max_samples == 0:
        args.max_samples = None
    train(args)


if __name__ == "__main__":
    main()
