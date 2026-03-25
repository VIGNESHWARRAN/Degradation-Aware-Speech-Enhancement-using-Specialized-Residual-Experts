"""
diagnose.py
-----------
Deep diagnostic to understand the -36 dB SI-SDR.
Run from project root:
    python diagnose.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import torchaudio
from models.model import DegradationAwareSpeechEnhancer
from data.dataset import build_dataloaders, IDX_TO_NAME

CHECKPOINT = "outputs/experiment_02/checkpoints/best_stage1.pt"
DATA_ROOT  = "C:/Users/Sai Raman/OneDrive/Desktop/All Semesters/semester 6/Speech Recognition/final_processed/final_processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DegradationAwareSpeechEnhancer(
    expert_dim=256, num_expert_blocks=3, decoder_dim=256,
    num_upsample=8, dropout=0.05
).to(device)

ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# Print learned output_scale
scale = model.decoder.output_scale.item()
print(f"\n[DIAG] Learned output_scale = {scale:.6f}")

_, _, test_loader = build_dataloaders(DATA_ROOT, batch_size=4, num_workers=0)
batch    = next(iter(test_loader))
degraded = batch["degraded"].to(device)
clean    = batch["clean"].to(device)
labels   = batch["label"].to(device)

with torch.no_grad():
    # Step through model manually
    features = model.backbone(degraded)
    refined  = model.experts(features, labels)
    enhanced = model.decoder(refined, degraded.shape[-1])

print(f"\n[DIAG] Tensor shapes:")
print(f"  degraded  : {degraded.shape}")
print(f"  features  : {features.shape}")
print(f"  refined   : {refined.shape}")
print(f"  enhanced  : {enhanced.shape}")

print(f"\n[DIAG] Amplitude statistics (first 4 samples):")
print(f"  {'':10}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}  {'abs_max':>10}")
for i in range(min(4, len(clean))):
    c = clean[i].cpu().numpy()
    e = enhanced[i].cpu().numpy()
    d = degraded[i].cpu().numpy()
    print(f"  clean  [{i}]  {c.mean():>10.4f}  {c.std():>10.4f}  {c.min():>10.4f}  {c.max():>10.4f}  {np.abs(c).max():>10.4f}")
    print(f"  enh    [{i}]  {e.mean():>10.4f}  {e.std():>10.4f}  {e.min():>10.4f}  {e.max():>10.4f}  {np.abs(e).max():>10.4f}")
    print(f"  degrad [{i}]  {d.mean():>10.4f}  {d.std():>10.4f}  {d.min():>10.4f}  {d.max():>10.4f}  {np.abs(d).max():>10.4f}")
    
    # Correlation between enhanced and clean
    corr = np.corrcoef(c, e)[0,1]
    corr_deg = np.corrcoef(c, d)[0,1]
    print(f"  corr(clean,enhanced)={corr:.4f}   corr(clean,degraded)={corr_deg:.4f}")
    
    # SI-SDR manual
    c_zm = c - c.mean(); e_zm = e - e.mean()
    alpha = np.dot(c_zm, e_zm) / (np.dot(c_zm, c_zm) + 1e-8)
    proj  = alpha * c_zm
    noise = e_zm - proj
    sdr   = 10*np.log10((np.dot(proj,proj)+1e-8)/(np.dot(noise,noise)+1e-8))
    print(f"  SI-SDR={sdr:.2f} dB  scale_factor(alpha)={alpha:.4f}")
    print()

print(f"\n[DIAG] Decoder output_scale parameter = {scale:.6f}")
print(f"[DIAG] If scale is near 0.1 and alpha is near 0, the decoder is producing near-zero output")
print(f"[DIAG] If correlation > 0.3 but SI-SDR < 0, it is a pure amplitude scale problem")