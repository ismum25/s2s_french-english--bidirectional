"""
Full-duplex translator.

Opens the microphone at its native sample rate, resamples to 16 kHz
for Whisper, routes each utterance through ASR->MT->TTS, and plays back.
Mic is muted during playback to prevent echo re-transcription.
"""

import queue
import threading
import time

import numpy as np
import scipy.signal
import sounddevice as sd

from config import ASR_SR
from translation import translate_and_speak
from vad import VADAccum

# Sample rates to try when the device doesn't support 16 kHz directly.
_CANDIDATE_RATES = [16000, 48000, 44100, 22050, 32000, 8000]


def _find_input_device(preferred: int | None) -> tuple[int, int]:
    """
    Return (device_index, native_sample_rate).

    Tries every candidate rate with sd.check_input_settings so we never
    pass an unsupported configuration to PortAudio.  Prefers WASAPI on
    Windows, falls back to DirectSound / MME.  WDM-KS is skipped because
    it reliably returns paInvalidDevice.
    """
    hostapis = sd.query_hostapis()

    def api_rank(api_name: str) -> int:
        n = api_name.lower()
        if "wasapi"      in n: return 0
        if "directsound" in n: return 1
        if "mme"         in n: return 2
        if "wdm"         in n: return 99
        return 50

    def try_device(dev_idx: int) -> int | None:
        """Return the first supported sample rate, or None."""
        for rate in _CANDIDATE_RATES:
            try:
                sd.check_input_settings(
                    device=dev_idx, channels=1,
                    dtype="float32", samplerate=rate,
                )
                return rate
            except Exception:
                continue
        return None

    if preferred is not None:
        rate = try_device(preferred)
        if rate is not None:
            return preferred, rate
        raise RuntimeError(
            f"Input device {preferred} does not support any of "
            f"{_CANDIDATE_RATES}. Run --list-devices to pick another."
        )

    # Enumerate all devices, rank by host API preference.
    devices = list(enumerate(sd.query_devices()))
    ranked  = sorted(
        [(i, info) for i, info in devices if info["max_input_channels"] > 0],
        key=lambda x: api_rank(hostapis[x[1]["hostapi"]]["name"]),
    )

    for dev_idx, info in ranked:
        rate = try_device(dev_idx)
        if rate is not None:
            api_name = hostapis[info["hostapi"]]["name"]
            print(f"[audio] input  device {dev_idx}: {info['name']}  ({api_name})  {rate} Hz")
            return dev_idx, rate

    raise RuntimeError(
        "No working input audio device found. "
        "Run --list-devices and pass --input-device N."
    )


def _find_output_device(preferred: int | None) -> int:
    """
    Return a device index suitable for sd.play().
    sd.play() handles arbitrary sample rates internally, so we only need
    to verify the device has output channels and isn't WDM-KS.
    """
    hostapis = sd.query_hostapis()

    def api_rank(api_name: str) -> int:
        n = api_name.lower()
        if "wasapi"      in n: return 0
        if "directsound" in n: return 1
        if "mme"         in n: return 2
        if "wdm"         in n: return 99
        return 50

    def try_device(dev_idx: int) -> bool:
        try:
            sd.check_output_settings(device=dev_idx, channels=1, dtype="float32")
            return True
        except Exception:
            return False

    if preferred is not None:
        if try_device(preferred):
            return preferred
        raise RuntimeError(
            f"Output device {preferred} is not usable. "
            "Run --list-devices to pick another."
        )

    devices = list(enumerate(sd.query_devices()))
    ranked  = sorted(
        [(i, info) for i, info in devices if info["max_output_channels"] > 0],
        key=lambda x: api_rank(hostapis[x[1]["hostapi"]]["name"]),
    )

    for dev_idx, info in ranked:
        if try_device(dev_idx):
            api_name = hostapis[info["hostapi"]]["name"]
            print(f"[audio] output device {dev_idx}: {info['name']}  ({api_name})")
            return dev_idx

    raise RuntimeError(
        "No working output audio device found. "
        "Run --list-devices and pass --output-device N."
    )


class FullDuplexTranslator:
    """
    Usage:
        t = FullDuplexTranslator(models)
        t.run_forever()          # blocks until Ctrl+C
        # --- or ---
        t.start()
        ...
        t.stop()
    """

    CHUNK_MS = 30

    def __init__(
        self,
        models:        dict,
        rms_thresh:    float = 0.008,
        silence_ms:    int   = 700,
        min_dur_s:     float = 0.4,
        input_device:  int | None = None,
        output_device: int | None = None,
    ):
        self.models        = models
        self.rms_thresh    = rms_thresh
        self.silence_ms    = silence_ms
        self.min_dur_s     = min_dur_s
        self.input_device  = input_device
        self.output_device = output_device

        self._q          = queue.Queue()
        self._stop_evt   = threading.Event()
        self._native_sr  = ASR_SR
        self._out_dev    = None
        self._out_sr     = None
        self._vad:    VADAccum | None        = None
        self._stream: sd.InputStream | None  = None
        self._thread: threading.Thread | None = None

    def _mic_cb(self, indata, frames, time_info, status):
        chunk = indata[:, 0].astype(np.float32)
        # Resample to ASR_SR (16 kHz) if the device runs at a different rate.
        if self._native_sr != ASR_SR:
            chunk = scipy.signal.resample_poly(
                chunk, ASR_SR, self._native_sr
            ).astype(np.float32)
        self._vad.push(chunk)

    def _worker(self):
        while not self._stop_evt.is_set():
            try:
                audio_16k = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                result = translate_and_speak(self.models, audio_16k)
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                continue

            if result is None:
                continue

            wav, sr_out = result
            if self._out_sr is not None and sr_out != self._out_sr:
                wav = scipy.signal.resample_poly(
                    wav.astype(np.float32), self._out_sr, sr_out
                ).astype(np.float32)
                sr_out = self._out_sr
            elapsed_ms  = (time.perf_counter() - t0) * 1000
            audio_ms    = len(wav) / sr_out * 1000
            print(f"  Latency {elapsed_ms:.0f} ms | Audio {audio_ms:.0f} ms")

            self._vad.muted = True
            sd.play(wav, samplerate=sr_out, device=self._out_dev)
            sd.wait()
            self._vad.muted = False

    def start(self) -> None:
        in_dev, native_sr = _find_input_device(self.input_device)
        self._native_sr   = native_sr
        self._out_dev     = _find_output_device(self.output_device)
        try:
            self._out_sr = int(sd.query_devices(self._out_dev)["default_samplerate"])
        except Exception:
            self._out_sr = None

        # Chunk size in native samples so CHUNK_MS stays constant in time.
        chunk_n = int(native_sr * self.CHUNK_MS / 1000)

        self._stop_evt.clear()
        self._vad = VADAccum(
            out_queue=self._q,
            rms_thresh=self.rms_thresh,
            silence_ms=self.silence_ms,
            chunk_ms=self.CHUNK_MS,
            min_dur_s=self.min_dur_s,
        )

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        self._stream = sd.InputStream(
            samplerate=native_sr, channels=1, dtype="float32",
            blocksize=chunk_n, callback=self._mic_cb,
            device=in_dev,
        )
        self._stream.start()

        print("━" * 54)
        print("  FULL-DUPLEX TRANSLATOR  ▶  RUNNING")
        print("  Speak French   ->  translated to English")
        print("  Speak English  ->  translated to French")
        print(f"  Mic RMS threshold : {self.rms_thresh}")
        print(f"  Silence cutoff    : {self.silence_ms} ms")
        print("  Press Ctrl+C to stop.")
        print("━" * 54)

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stop_evt.set()
        sd.stop()
        if self._thread:
            self._thread.join(timeout=3)
        print("Translator stopped.")

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.stop()
