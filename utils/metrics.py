"""
metrics.py
----------
Evaluation metrics for speech enhancement.

Metrics implemented:
    - SI-SDR  (Scale-Invariant Signal-to-Distortion Ratio)  — in-house, no deps
    - STOI    (Short-Time Objective Intelligibility)         — requires pystoi
    - PESQ    (Perceptual Evaluation of Speech Quality)      — requires pesq

All functions accept numpy arrays (float32, shape: (T,)) at 16 kHz.
The evaluate_batch() helper accepts torch tensors.
"""

import numpy as np
import torch


# ------------------------------------------------------------------
# SI-SDR  (pure numpy, no extra deps)
# ------------------------------------------------------------------
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio.
    Higher is better. Units: dB.
    """
    reference = reference - reference.mean()
    estimate  = estimate  - estimate.mean()

    dot        = np.dot(reference, estimate)
    ref_pow    = np.dot(reference, reference) + 1e-8
    proj       = dot / ref_pow * reference
    noise      = estimate - proj

    ratio = np.dot(proj, proj) / (np.dot(noise, noise) + 1e-8)
    return 10 * np.log10(ratio + 1e-8)


# ------------------------------------------------------------------
# STOI  (requires: pip install pystoi)
# ------------------------------------------------------------------
def stoi_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int = 16_000, extended: bool = False) -> float:
    """
    Short-Time Objective Intelligibility. Range [0, 1], higher is better.
    extended=True uses extended STOI (eSTOI).
    """
    try:
        from pystoi import stoi
        return stoi(reference, estimate, sr, extended=extended)
    except ImportError:
        print("[metrics] pystoi not installed. Run: pip install pystoi")
        return float("nan")


# ------------------------------------------------------------------
# PESQ  (requires: pip install pesq)
# ------------------------------------------------------------------
def pesq_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int = 16_000) -> float:
    """
    Perceptual Evaluation of Speech Quality.
    WB mode (16kHz): range [-0.5, 4.5]. Higher is better.
    """
    try:
        from pesq import pesq
        mode = "wb" if sr == 16_000 else "nb"
        return pesq(sr, reference, estimate, mode)
    except ImportError:
        print("[metrics] pesq not installed. Run: pip install pesq")
        return float("nan")


# ------------------------------------------------------------------
# Batch evaluation helper  (used by run_experiments.py)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_batch(
    model,
    loader,
    device: torch.device,
    compute_pesq: bool = False,
    compute_stoi: bool = True,
) -> dict:
    """
    Run model on an entire DataLoader and return averaged metrics.

    Returns:
        dict with keys: si_sdr, stoi (opt), pesq (opt),
        and per degradation type breakdowns.
    """
    from data.dataset import IDX_TO_NAME

    results = {
        "si_sdr": [],
        "stoi"  : [],
        "pesq"  : [],
    }
    per_type = {name: {"si_sdr": [], "stoi": [], "pesq": []}
                for name in ["noise", "reverb", "device"]}

    model.eval()
    for batch in loader:
        degraded = batch["degraded"].to(device)
        clean    = batch["clean"].to(device)
        labels   = batch["label"].to(device)

        enhanced = model(degraded, labels)   # (B, T)

        # Move to CPU numpy for metric computation
        enhanced_np = enhanced.cpu().float().numpy()
        clean_np    = clean.cpu().float().numpy()
        labels_np   = labels.cpu().numpy()

        for i in range(len(clean_np)):
            ref = clean_np[i]
            est = enhanced_np[i]
            lbl = int(labels_np[i])
            name = IDX_TO_NAME.get(lbl, "unknown")

            sdr = si_sdr(ref, est)
            results["si_sdr"].append(sdr)
            per_type[name]["si_sdr"].append(sdr)

            if compute_stoi:
                st = stoi_score(ref, est)
                results["stoi"].append(st)
                per_type[name]["stoi"].append(st)

            if compute_pesq:
                pq = pesq_score(ref, est)
                results["pesq"].append(pq)
                per_type[name]["pesq"].append(pq)

    def _mean(lst):
        lst = [x for x in lst if not np.isnan(x)]
        return float(np.mean(lst)) if lst else float("nan")

    summary = {
        "si_sdr_mean" : _mean(results["si_sdr"]),
        "stoi_mean"   : _mean(results["stoi"]),
        "pesq_mean"   : _mean(results["pesq"]),
        "per_type"    : {
            name: {
                "si_sdr": _mean(per_type[name]["si_sdr"]),
                "stoi"  : _mean(per_type[name]["stoi"]),
                "pesq"  : _mean(per_type[name]["pesq"]),
            }
            for name in per_type
        },
    }
    return summary