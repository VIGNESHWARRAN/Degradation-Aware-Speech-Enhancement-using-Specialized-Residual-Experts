"""
metrics.py
----------
Full evaluation metrics for Degradation-Aware Speech Enhancement.

Metrics implemented:
    ── Perceptual Quality (composite) ──────────────────────────────
    CSIG  : Signal quality MOS prediction    (range 1–5, higher better)
    CBAK  : Background suppression MOS       (range 1–5, higher better)
    COVL  : Overall quality MOS              (range 1–5, higher better)
    (CSIG/CBAK/COVL use Hu & Loizou 2008 composite measure)

    ── Spectral & Intelligibility ───────────────────────────────────
    SI-SDR : Scale-Invariant Signal-to-Distortion Ratio  (dB, higher better)
    LSD    : Log-Spectral Distance  (dB, lower better)
    STOI   : Short-Time Objective Intelligibility  (0–1, higher better)
    PESQ   : Perceptual Evaluation of Speech Quality  (requires pesq pkg)

    ── Efficiency ───────────────────────────────────────────────────
    RTF    : Real-Time Factor  (lower better; <1.0 means faster-than-real-time)
    Expert Utilization: fraction of samples routed to each expert

All signal-level functions take numpy float32 (T,) arrays at 16 kHz.
"""

import time
import numpy as np
import torch

# ── Optional deps — checked ONCE at import, no per-sample spam ────
_PYSTOI_AVAILABLE = False
_PESQ_AVAILABLE   = False

try:
    from pystoi import stoi as _stoi_fn
    _PYSTOI_AVAILABLE = True
except ImportError:
    print("[metrics] pystoi not installed — STOI=nan. Fix: pip install pystoi")

try:
    from pesq import pesq as _pesq_fn
    _PESQ_AVAILABLE = True
except ImportError:
    pass  # silent — hard to install on Windows


# ==================================================================
# ── Core signal metrics ───────────────────────────────────────────
# ==================================================================

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant SDR in dB. Higher is better."""
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    dot     = np.dot(ref, est)
    ref_pow = np.dot(ref, ref) + 1e-8
    proj    = dot / ref_pow * ref
    noise   = est - proj
    return float(10 * np.log10((np.dot(proj, proj) + 1e-8) /
                                (np.dot(noise, noise) + 1e-8)))


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray,
                           sr: int = 16_000, n_fft: int = 512) -> float:
    """
    Log-Spectral Distance (LSD) in dB. Lower is better.
    Measures spectral envelope difference per frame, then averages.
    """
    hop    = n_fft // 4
    window = np.hanning(n_fft)

    def _log_power_spectrum(x):
        frames = []
        for start in range(0, len(x) - n_fft + 1, hop):
            frame  = x[start: start + n_fft] * window
            spec   = np.abs(np.fft.rfft(frame)) ** 2
            frames.append(10 * np.log10(spec + 1e-8))
        return np.array(frames)   # (T', F)

    ref_log = _log_power_spectrum(reference)
    est_log = _log_power_spectrum(estimate)

    # Trim to same length
    n = min(len(ref_log), len(est_log))
    lsd = np.sqrt(np.mean((ref_log[:n] - est_log[:n]) ** 2))
    return float(lsd)


def stoi_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int = 16_000) -> float:
    """STOI in [0, 1]. Higher is better."""
    if not _PYSTOI_AVAILABLE:
        return float("nan")
    try:
        return float(_stoi_fn(reference, estimate, sr, extended=False))
    except Exception:
        return float("nan")


def pesq_score(reference: np.ndarray, estimate: np.ndarray,
               sr: int = 16_000) -> float:
    """PESQ WB MOS in [-0.5, 4.5]. Higher is better."""
    if not _PESQ_AVAILABLE:
        return float("nan")
    try:
        mode = "wb" if sr == 16_000 else "nb"
        return float(_pesq_fn(sr, reference, estimate, mode))
    except Exception:
        return float("nan")


# ==================================================================
# ── Composite perceptual metrics (Hu & Loizou 2008) ─────────────
# CSIG / CBAK / COVL via PESQ + wss + llr
# ==================================================================

def _wss(reference: np.ndarray, estimate: np.ndarray,
         sr: int = 16_000, n_fft: int = 512) -> float:
    """Weighted Spectral Slope distance (internal helper)."""
    hop    = n_fft // 4
    window = np.hanning(n_fft)

    def _spectra(x):
        out = []
        for start in range(0, len(x) - n_fft + 1, hop):
            frame = x[start: start + n_fft] * window
            out.append(np.abs(np.fft.rfft(frame)))
        return np.array(out)   # (T', F)

    ref_s = _spectra(reference)
    est_s = _spectra(estimate)
    n     = min(len(ref_s), len(est_s))
    ref_s, est_s = ref_s[:n], est_s[:n]

    # Spectral slope
    ref_dif = np.diff(ref_s, axis=1)
    est_dif = np.diff(est_s, axis=1)

    # Weights based on reference magnitude
    w = ref_s[:, :-1] / (ref_s[:, :-1].sum(axis=1, keepdims=True) + 1e-8)
    wss_val = np.mean(np.sum(w * (ref_dif - est_dif) ** 2, axis=1))
    return float(wss_val)


def _llr(reference: np.ndarray, estimate: np.ndarray,
         sr: int = 16_000, n_fft: int = 512) -> float:
    """Log-Likelihood Ratio distortion (internal helper)."""
    from numpy.linalg import LinAlgError

    hop    = n_fft // 4
    window = np.hanning(n_fft)
    order  = 10   # LPC order

    def _lpc(frame):
        """Compute LPC coefficients via autocorrelation."""
        r = np.correlate(frame, frame, mode='full')
        r = r[len(r)//2: len(r)//2 + order + 1]
        R = np.array([[r[abs(i-j)] for j in range(order)]
                      for i in range(order)])
        try:
            a = np.linalg.solve(R, -r[1:order + 1])
        except LinAlgError:
            return np.zeros(order)
        return a

    llr_vals = []
    for start in range(0, min(len(reference), len(estimate)) - n_fft + 1, hop):
        ref_frame = reference[start: start + n_fft] * window
        est_frame = estimate [start: start + n_fft] * window

        a_ref = _lpc(ref_frame)
        a_est = _lpc(est_frame)

        # Autocorrelation of reference frame
        r_ref = np.correlate(ref_frame, ref_frame, mode='full')
        r_ref = r_ref[len(r_ref)//2: len(r_ref)//2 + order + 1]

        num = a_est @ np.array([[r_ref[abs(i-j)] for j in range(order)]
                                 for i in range(order)]) @ a_est
        den = a_ref @ np.array([[r_ref[abs(i-j)] for j in range(order)]
                                 for i in range(order)]) @ a_ref

        val = np.log((num + 1e-8) / (den + 1e-8))
        llr_vals.append(np.clip(val, -10, 10))

    return float(np.mean(llr_vals)) if llr_vals else 0.0


def composite_measures(reference: np.ndarray, estimate: np.ndarray,
                        sr: int = 16_000) -> dict:
    """
    Compute CSIG, CBAK, COVL using Hu & Loizou (2008) formulas.

    Uses PESQ if available, otherwise approximates with SI-SDR.

    Returns dict with keys: csig, cbak, covl
    """
    # PESQ score (or proxy)
    pesq_val = pesq_score(reference, estimate, sr)
    if np.isnan(pesq_val):
        # Fallback: approximate PESQ from SI-SDR (rough linear mapping)
        sdr = si_sdr(reference, estimate)
        pesq_val = float(np.clip(1.0 + (sdr + 20) / 15, 1.0, 4.5))

    wss_val = _wss(reference, estimate, sr)
    llr_val = _llr(reference, estimate, sr)

    # Hu & Loizou 2008 composite equations
    csig = float(np.clip(3.093 - 1.029 * llr_val
                         + 0.603 * pesq_val - 0.009 * wss_val, 1, 5))
    cbak = float(np.clip(1.634 + 0.478 * pesq_val
                         - 0.007 * wss_val - 0.138 * llr_val, 1, 5))
    covl = float(np.clip(1.594 + 0.805 * pesq_val
                         - 0.512 * llr_val - 0.007 * wss_val, 1, 5))

    return {"csig": csig, "cbak": cbak, "covl": covl}


# ==================================================================
# ── Batch evaluation ─────────────────────────────────────────────
# ==================================================================

@torch.no_grad()
def evaluate_batch(
    model,
    loader,
    device: torch.device,
    compute_pesq      : bool = False,
    compute_stoi      : bool = True,
    compute_composite : bool = True,
    compute_lsd       : bool = True,
) -> dict:
    """
    Full evaluation over a DataLoader.

    Returns dict with:
        si_sdr_mean, si_sdr_baseline_mean, si_sdr_improvement
        lsd_mean
        stoi_mean
        pesq_mean
        csig_mean, cbak_mean, covl_mean
        rtf_mean                          ← Real-Time Factor
        expert_utilization                ← fraction per expert {noise, reverb, device}
        per_type                          ← all metrics broken down by degradation type
    """
    from data.dataset import IDX_TO_NAME

    NAMES = ["noise", "reverb", "device"]
    empty_bucket = lambda: {
        "si_sdr": [], "si_sdr_base": [], "lsd": [],
        "stoi": [], "pesq": [], "csig": [], "cbak": [], "covl": [],
    }
    buckets = {n: empty_bucket() for n in NAMES}
    all_rtf = []
    expert_counts = {n: 0 for n in NAMES}
    total_samples = 0

    model.eval()
    total_batches = len(loader)
    sr = 16_000

    for batch_idx, batch in enumerate(loader):
        degraded = batch["degraded"].to(device)
        clean    = batch["clean"].to(device)
        labels   = batch["label"].to(device)

        # ── RTF timing ───────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start  = time.perf_counter()
        enhanced = model(degraded, labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end    = time.perf_counter()

        audio_duration = degraded.shape[0] * degraded.shape[1] / sr  # seconds
        elapsed        = t_end - t_start
        all_rtf.append(elapsed / (audio_duration + 1e-8))

        # ── Expert utilization ───────────────────────────────────
        for idx, name in IDX_TO_NAME.items():
            cnt = (labels == idx).sum().item()
            expert_counts[name]  += cnt
        total_samples += len(labels)

        # ── Per-sample metrics ───────────────────────────────────
        enh_np  = enhanced.cpu().float().numpy()
        deg_np  = degraded.cpu().float().numpy()
        cln_np  = clean.cpu().float().numpy()
        lbl_np  = labels.cpu().numpy()

        for i in range(len(cln_np)):
            ref  = cln_np[i]
            est  = enh_np[i]
            deg  = deg_np[i]
            name = IDX_TO_NAME.get(int(lbl_np[i]), "noise")

            buckets[name]["si_sdr"].append(si_sdr(ref, est))
            buckets[name]["si_sdr_base"].append(si_sdr(ref, deg))

            if compute_lsd:
                buckets[name]["lsd"].append(log_spectral_distance(ref, est, sr))

            if compute_stoi:
                buckets[name]["stoi"].append(stoi_score(ref, est, sr))

            if compute_pesq:
                buckets[name]["pesq"].append(pesq_score(ref, est, sr))

            if compute_composite:
                cm = composite_measures(ref, est, sr)
                buckets[name]["csig"].append(cm["csig"])
                buckets[name]["cbak"].append(cm["cbak"])
                buckets[name]["covl"].append(cm["covl"])

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            print(f"  [eval] {batch_idx+1}/{total_batches} batches", flush=True)

    # ── Aggregate ────────────────────────────────────────────────
    def _m(lst):
        lst = [x for x in lst if not np.isnan(x)]
        return float(np.mean(lst)) if lst else float("nan")

    def _all(key):
        return [v for n in NAMES for v in buckets[n][key]]

    expert_util = {n: expert_counts[n] / max(total_samples, 1) for n in NAMES}

    per_type = {}
    for name in NAMES:
        b = buckets[name]
        sdr  = _m(b["si_sdr"])
        base = _m(b["si_sdr_base"])
        per_type[name] = {
            "si_sdr"            : sdr,
            "si_sdr_baseline"   : base,
            "si_sdr_improvement": sdr - base,
            "lsd"               : _m(b["lsd"]),
            "stoi"              : _m(b["stoi"]),
            "pesq"              : _m(b["pesq"]),
            "csig"              : _m(b["csig"]),
            "cbak"              : _m(b["cbak"]),
            "covl"              : _m(b["covl"]),
            "expert_utilization": expert_util[name],
        }

    all_sdr  = _m(_all("si_sdr"))
    all_base = _m(_all("si_sdr_base"))

    return {
        "si_sdr_mean"          : all_sdr,
        "si_sdr_baseline_mean" : all_base,
        "si_sdr_improvement"   : all_sdr - all_base,
        "lsd_mean"             : _m(_all("lsd")),
        "stoi_mean"            : _m(_all("stoi")),
        "pesq_mean"            : _m(_all("pesq")),
        "csig_mean"            : _m(_all("csig")),
        "cbak_mean"            : _m(_all("cbak")),
        "covl_mean"            : _m(_all("covl")),
        "rtf_mean"             : float(np.mean(all_rtf)),
        "expert_utilization"   : expert_util,
        "per_type"             : per_type,
    }