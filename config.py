"""Shared constants and device detection."""

import torch

MIMI_SR = 24000   # Mimi / MMS-TTS native sample rate
MIMI_FR = 12.5    # Mimi tokens per second
ASR_SR  = 16000   # Whisper sample rate


def detect_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = detect_device()
