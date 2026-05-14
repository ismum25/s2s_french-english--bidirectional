#!/usr/bin/env python3
"""
Full-Duplex FR/EN Translator — entry point.

Usage:
    python translator.py
    python translator.py --threshold 0.01 --silence 900
    python translator.py --list-devices

Install (once):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install transformers>=4.40.0 openai-whisper sentencepiece sounddevice librosa scipy numpy
"""

import argparse
import logging
import warnings

import sounddevice as sd

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from config import DEVICE
from api_server import start_api
from latency import LATENCY_HUB
from models import load_models
from duplex import FullDuplexTranslator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time FR<->EN full-duplex translator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--threshold", type=float, default=0.008,
        help="RMS energy VAD threshold (raise to reduce false triggers)")
    parser.add_argument(
        "--silence", type=int, default=700,
        help="Silence in ms that ends an utterance")
    parser.add_argument(
        "--min-dur", type=float, default=0.4,
        help="Minimum speech segment length in seconds")
    parser.add_argument(
        "--input-device", type=int, default=None,
        help="Mic device index (see --list-devices)")
    parser.add_argument(
        "--output-device", type=int, default=None,
        help="Speaker device index (see --list-devices)")
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio devices and exit")
    parser.add_argument(
        "--no-ws", action="store_true",
        help="Disable the WebSocket latency server")
    parser.add_argument(
        "--ws-host", type=str, default="127.0.0.1",
        help="WebSocket server host for UI")
    parser.add_argument(
        "--ws-port", type=int, default=8765,
        help="WebSocket server port for UI")
    parser.add_argument(
        "--no-api", action="store_true",
        help="Disable the HTTP API server")
    parser.add_argument(
        "--api-host", type=str, default="127.0.0.1",
        help="HTTP API server host for UI")
    parser.add_argument(
        "--api-port", type=int, default=8000,
        help="HTTP API server port for UI")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not args.no_api:
        start_api(host=args.api_host, port=args.api_port)
        print(f"[init] API server: http://{args.api_host}:{args.api_port}")

    if not args.no_ws:
        LATENCY_HUB.start(host=args.ws_host, port=args.ws_port)
        print(f"[init] WS latency server: ws://{args.ws_host}:{args.ws_port}")

    print(f"[init] Device: {DEVICE}")
    models = load_models()

    FullDuplexTranslator(
        models=models,
        rms_thresh=args.threshold,
        silence_ms=args.silence,
        min_dur_s=args.min_dur,
        input_device=args.input_device,
        output_device=args.output_device,
    ).run_forever()


if __name__ == "__main__":
    main()
