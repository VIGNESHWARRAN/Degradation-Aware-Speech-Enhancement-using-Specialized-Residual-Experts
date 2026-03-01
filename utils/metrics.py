"""
Evaluation metrics used for speech enhancement such as PESQ, STOI, SI-SDR, DNSMOS wrappers.
"""


def compute_pesq(clean, enhanced, fs=16000):
    """Return PESQ score (requires external package)."""
    pass


def compute_stoi(clean, enhanced, fs=16000):
    """Return STOI score."""
    pass


def compute_sisdr(clean, enhanced):
    """Return SI-SDR between signals."""
    pass
