"""
run_experiments.py
------------------
Entry point for training and evaluation.

Usage:
    # Train with default config
    python experiments/run_experiments.py

    # Train with custom config
    python experiments/run_experiments.py --config configs/config.yaml

    # Evaluate best checkpoint on test set
    python experiments/run_experiments.py --eval --checkpoint outputs/experiment_01/checkpoints/best_stage2.pt

    # Override specific config keys from CLI
    python experiments/run_experiments.py --set batch_size=4 stage1_epochs=5
"""

import sys
import os
import argparse
import json

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.training import Trainer
from models.model    import DegradationAwareSpeechEnhancer
from data.dataset    import build_dataloaders
from utils.metrics   import evaluate_batch


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Degradation-Aware Speech Enhancement")
    p.add_argument("--config",     type=str, default="configs/config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--eval",       action="store_true",
                   help="Run evaluation only (no training)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path for --eval mode")
    p.add_argument("--set", nargs="*", default=[],
                   help="Override config keys: key=value ...")
    return p.parse_args()


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[run] Config not found at {path}, using defaults.")
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_overrides(cfg: dict, overrides: list) -> dict:
    for kv in overrides:
        k, v = kv.split("=", 1)
        # Auto-cast
        try:    v = int(v)
        except ValueError:
            try:    v = float(v)
            except ValueError:
                if v.lower() == "true":  v = True
                elif v.lower() == "false": v = False
                elif v.lower() == "null":  v = None
        cfg[k] = v
        print(f"[run] Override: {k} = {v}")
    return cfg


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
def run_training(cfg: dict):
    trainer = Trainer(cfg)
    trainer.train()

    # Final test-set evaluation after training
    print("\n[run] Running final test-set evaluation...")
    run_eval(cfg, checkpoint=str(
        trainer.ckpt_dir / "best_stage2.pt"
    ))


# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------
def run_eval(cfg: dict, checkpoint: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DegradationAwareSpeechEnhancer(
        expert_dim        = cfg.get("expert_dim",        512),
        num_expert_blocks = cfg.get("num_expert_blocks", 4),
        decoder_dim       = cfg.get("decoder_dim",       512),
        num_upsample      = cfg.get("num_upsample",      8),
        dropout           = cfg.get("dropout",           0.1),
    ).to(device)

    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[run] Loaded checkpoint: {checkpoint}")
    else:
        print("[run] WARNING: No checkpoint loaded — evaluating random weights.")

    _, _, test_loader = build_dataloaders(
        data_root  = cfg.get("data_root", "data/final_processed"),
        batch_size = cfg.get("batch_size", 8),
        num_workers= cfg.get("num_workers", 4),
    )

    print("[run] Evaluating on test set (SI-SDR + STOI)...")
    results = evaluate_batch(
        model,
        test_loader,
        device,
        compute_pesq = False,   # set True if pesq is installed
        compute_stoi = True,
    )

    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    print(f"  SI-SDR  : {results['si_sdr_mean']:.2f} dB")
    print(f"  STOI    : {results['stoi_mean']:.4f}")
    print(f"  PESQ    : {results['pesq_mean']:.4f}")
    print()
    for name, vals in results["per_type"].items():
        print(f"  [{name:>6}]  SI-SDR={vals['si_sdr']:.2f}  "
              f"STOI={vals['stoi']:.4f}  PESQ={vals['pesq']:.4f}")
    print("="*50)

    # Save results JSON
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[run] Results saved to {result_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args.set)

    if args.eval:
        run_eval(cfg, checkpoint=args.checkpoint)
    else:
        run_training(cfg)