"""
ASR -> MT -> TTS translation cascade.
Whisper detects language automatically; routes FR<->EN accordingly.
"""

import time

import numpy as np
import torch

from config import DEVICE
from latency import LATENCY_HUB


def translate_and_speak(
    models: dict,
    audio_16k: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """
    Transcribe audio_16k (float32, 16 kHz) with Whisper, translate,
    and synthesise with MMS-TTS.

    Returns (wav_float32, sample_rate) or None if nothing to speak.
    """
    t0 = time.perf_counter()
    res = models["whisper"].transcribe(audio_16k, task="transcribe")
    asr_ms = (time.perf_counter() - t0) * 1000
    lang = res["language"]
    text = res["text"].strip()

    if not text:
        return None

    print(f"\n  [{lang.upper()}] {text}")

    if lang == "fr":
        mt_tok   = models["mt_fr_en_tok"]
        mt_model = models["mt_fr_en_model"]
        tts_tok  = models["tts_en_tok"]
        tts_mdl  = models["tts_en_model"]
        tgt_tag  = "EN"
        labels   = {
            "asr": "A Whisper FR ASR",
            "mt":  "B Helsinki-NLP FR->EN MT",
            "tts": "C MMS-TTS EN synthesis",
        }
    elif lang == "en":
        mt_tok   = models["mt_en_fr_tok"]
        mt_model = models["mt_en_fr_model"]
        tts_tok  = models["tts_fr_tok"]
        tts_mdl  = models["tts_fr_model"]
        tgt_tag  = "FR"
        labels   = {
            "asr": "A Whisper EN ASR",
            "mt":  "B Helsinki-NLP EN->FR MT",
            "tts": "C MMS-TTS FR synthesis",
        }
    else:
        print(f"  [SKIP] unsupported language: {lang}")
        return None

    t1 = time.perf_counter()
    enc = mt_tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        ids = mt_model.generate(**enc, max_new_tokens=256)
    translated = mt_tok.decode(ids[0], skip_special_tokens=True)
    mt_ms = (time.perf_counter() - t1) * 1000
    print(f"       -> [{tgt_tag}] {translated}")

    t2 = time.perf_counter()
    tts_in = tts_tok(translated, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        wav = tts_mdl(**tts_in).waveform[0].cpu().numpy().squeeze().astype(np.float32)
    tts_ms = (time.perf_counter() - t2) * 1000

    LATENCY_HUB.publish(
        {
            labels["asr"]: asr_ms,
            labels["mt"]: mt_ms,
            labels["tts"]: tts_ms,
        },
        meta={"language": lang, "text": text, "translated": translated},
    )

    return wav, tts_mdl.config.sampling_rate
