"""
Energy-based Voice Activity Detection accumulator.

Accepts float32 audio chunks in the range [-1, 1] at ASR_SR (16 kHz).
Emits complete speech segments to out_queue when silence follows speech.
Set .muted = True during TTS playback to prevent echo re-triggering.
"""

import os
import queue
from collections import deque

import numpy as np

from config import ASR_SR


class VADAccum:
    def __init__(
        self,
        out_queue:  queue.Queue,
        rms_thresh: float = 0.008,
        silence_ms: int   = 700,
        chunk_ms:   int   = 30,
        pre_frames: int   = 5,
        min_dur_s:  float = 0.4,
    ):
        self.q          = out_queue
        self.rms_thresh = rms_thresh
        self.sil_frames = max(1, int(silence_ms / chunk_ms))
        self.min_n      = int(min_dur_s * ASR_SR)
        self.debug      = os.getenv("VAD_DEBUG", "") != ""

        self.pre    = deque(maxlen=pre_frames)
        self.buf    = []
        self.sil    = 0
        self.active = False
        self.muted  = False

    def push(self, chunk_f32: np.ndarray) -> None:
        """chunk_f32: float32 array in [-1, 1] at ASR_SR."""
        if self.muted:
            return

        rms = float(np.sqrt(np.mean(chunk_f32 ** 2)))

        if rms > self.rms_thresh:
            if not self.active:
                if self.debug:
                    print(f"[vad] speech start rms={rms:.6f}")
                self.active = True
                self.buf    = [c.copy() for c in self.pre]
            self.buf.append(chunk_f32.copy())
            self.sil = 0

        elif self.active:
            self.buf.append(chunk_f32.copy())
            self.sil += 1
            if self.sil >= self.sil_frames:
                self.active = False
                audio = np.concatenate(self.buf).astype(np.float32)
                self.buf, self.sil = [], 0
                if len(audio) >= self.min_n:
                    if self.debug:
                        dur_ms = len(audio) / ASR_SR * 1000
                        print(f"[vad] speech end dur_ms={dur_ms:.0f}")
                    self.q.put(audio)

        else:
            self.pre.append(chunk_f32.copy())
