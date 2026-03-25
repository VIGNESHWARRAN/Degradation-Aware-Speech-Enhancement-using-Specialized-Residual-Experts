"""
run_experiments.py
------------------
Entry point for training and evaluation.

Usage:
    python experiments/run_experiments.py                          # train
    python experiments/run_experiments.py --config configs/config.yaml
    python experiments/run_experiments.py --eval --checkpoint outputs/experiment_02/checkpoints/best_stage2.pt
    python experiments/run_experiments.py --set batch_size=4 stage1_epochs=10
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.training import Trainer
from models.model    import DegradationAwareSpeechEnhancer
from data.dataset    import build_dataloaders
from utils.metrics   import evaluate_batch


# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--eval",       action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_args()


def load_config(path):
    if not os.path.exists(path):
        print(f"[run] Config not found at {path}, using defaults.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg, overrides):
    for kv in overrides:
        k, v = kv.split("=", 1)
        for cast in (int, float):
            try: v = cast(v); break
            except ValueError: pass
        if v == "true":  v = True
        if v == "false": v = False
        if v == "null":  v = None
        cfg[k] = v
        print(f"[run] Override: {k} = {v}")
    return cfg


# ------------------------------------------------------------------
def run_training(cfg):
    trainer = Trainer(cfg)
    trainer.train()
    print("\n[run] Running final test-set evaluation on best_stage2...")
    best = str(trainer.ckpt_dir / "best_stage2.pt")
    if not os.path.exists(best):
        best = str(trainer.ckpt_dir / "latest_stage2.pt")
    run_eval(cfg, checkpoint=best)


# ------------------------------------------------------------------
def run_eval(cfg, checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DegradationAwareSpeechEnhancer(
        expert_dim        = cfg.get("expert_dim",        256),  # v3 defaults
        num_expert_blocks = cfg.get("num_expert_blocks", 3),
        decoder_dim       = cfg.get("decoder_dim",       256),
        num_upsample      = cfg.get("num_upsample",      8),
        dropout           = cfg.get("dropout",           0.05),
    ).to(device)

    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[run] Loaded: {checkpoint}")
    else:
        print("[run] WARNING: checkpoint not found — random weights.")

    _, _, test_loader = build_dataloaders(
        data_root   = cfg.get("data_root", "data/final_processed"),
        batch_size  = cfg.get("batch_size", 8),
        num_workers = cfg.get("num_workers", 0),
    )

    print("[run] Evaluating on test set...")
    results = evaluate_batch(
        model, test_loader, device,
        compute_pesq      = False,
        compute_stoi      = True,
        compute_composite = True,
        compute_lsd       = True,
    )

    # ── Pretty print ─────────────────────────────────────────────
    W = 65
    print("\n" + "="*W)
    print("  TEST SET RESULTS")
    print("="*W)

    print(f"\n  ── Spectral & SDR ──────────────────────────────────")
    print(f"  SI-SDR  (model)    : {results['si_sdr_mean']:>8.2f} dB")
    print(f"  SI-SDR  (baseline) : {results['si_sdr_baseline_mean']:>8.2f} dB  ← degraded input")
    print(f"  SI-SDR  (improve)  : {results['si_sdr_improvement']:>+8.2f} dB")
    print(f"  LSD                : {results['lsd_mean']:>8.2f} dB  (lower better)")

    print(f"\n  ── Perceptual Quality (MOS 1–5) ────────────────────")
    print(f"  CSIG (signal qual) : {results['csig_mean']:>8.3f}")
    print(f"  CBAK (bg suppress) : {results['cbak_mean']:>8.3f}")
    print(f"  COVL (overall)     : {results['covl_mean']:>8.3f}")

    print(f"\n  ── Intelligibility ─────────────────────────────────")
    print(f"  STOI               : {results['stoi_mean']:>8.4f}  (higher better)")
    print(f"  PESQ               : {results['pesq_mean']:>8.4f}")

    print(f"\n  ── Efficiency ──────────────────────────────────────")
    print(f"  RTF                : {results['rtf_mean']:>8.4f}  (<1 = real-time)")
    print(f"  Expert utilization :")
    for name, frac in results["expert_utilization"].items():
        bar = "█" * int(frac * 20)
        print(f"    {name:<8} {frac*100:5.1f}%  {bar}")

    print(f"\n  ── Per Degradation Type ────────────────────────────")
    hdr = f"  {'Type':<8}  {'SI-SDR':>7}  {'Improve':>8}  {'LSD':>6}  {'CSIG':>6}  {'CBAK':>6}  {'COVL':>6}  {'STOI':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, v in results["per_type"].items():
        print(f"  {name:<8}  {v['si_sdr']:>7.2f}  {v['si_sdr_improvement']:>+8.2f}"
              f"  {v['lsd']:>6.2f}  {v['csig']:>6.3f}  {v['cbak']:>6.3f}"
              f"  {v['covl']:>6.3f}  {v['stoi']:>6.4f}")
    print("="*W)

    # Save
    out_dir = cfg.get("output_dir", "outputs/experiment_02")
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[run] Results saved → {result_path}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)
    cfg  = apply_overrides(cfg, args.set)
    if args.eval:
        run_eval(cfg, checkpoint=args.checkpoint)
    else:
        run_training(cfg)