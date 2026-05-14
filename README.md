# Full-Duplex FR/EN Translator + Latency Dashboard

Real-time full-duplex speech translator (FR <-> EN) with a React UI that shows per-stage latency.

## Features
- Whisper ASR -> Helsinki-NLP MT -> MMS-TTS
- Mimi latent pipeline timing (encoder/transform/decoder stages)
- FastAPI endpoint for latest latency snapshot
- React (Vite) dashboard that polls the API

## Requirements
- Python 3.10+
- Node.js 18+ (for the UI)
- Working input/output audio devices

## Install

### 1) Python dependencies

Install PyTorch first (pick the line for your GPU):

```
# CUDA 12.8 (RTX 50xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.x (RTX 40xx)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.x (RTX 30xx)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
# pip install torch torchvision torchaudio
```

Then install the rest:

```
pip install -r requirements.txt
```

### 2) UI dependencies

```
cd ui
npm install
```

## Run

### 1) Start the translator (includes API server)

```
python translator.py --threshold 0.02 --input-device 12 --output-device 8
```

Optional flags:
- `--list-devices` to show audio devices
- `--silence 700` to adjust speech end
- `--min-dur 0.2` to allow short utterances
- `--no-api` to disable the HTTP API
- `--api-host 127.0.0.1 --api-port 8000` to change API host/port

The API will start at:
```
http://127.0.0.1:8000/api/latency
```

### 2) Start the UI

```
cd ui
npm run dev
```

Open:
```
http://localhost:5173
```

## API

- `GET /api/latency` -> latest latency snapshot
- `GET /api/history` -> recent latency items

## Notes
- The UI polls the API every ~1.2s.
- Latency values appear only after a full translation completes.

## Reproducible Instructions
1) Install Python + Node requirements (see Install).
2) Pick devices with `python translator.py --list-devices`.
3) Start backend:

```
python translator.py --threshold 0.02 --input-device <MIC_INDEX> --output-device <SPEAKER_INDEX>
```

4) Start UI:

```
cd ui
npm run dev
```

5) Open `http://localhost:5173` and speak; latency updates after each translation.

## Agent Prompt (OpenCode)
Copy/paste this prompt into OpenCode to reproduce the setup automatically:

```
You are an automation agent. Reproduce the project setup and run commands in this repo:
1) Install Python deps: first install torch/torchvision/torchaudio for CUDA (choose cu128 unless user says otherwise), then run `pip install -r requirements.txt`.
2) Install UI deps: `cd ui` then `npm install`.
3) List audio devices: `python translator.py --list-devices` and ask user for mic/speaker indices.
4) Run backend: `python translator.py --threshold 0.02 --input-device <MIC_INDEX> --output-device <SPEAKER_INDEX>`.
5) Run UI: `cd ui` then `npm run dev`.
6) Confirm the API responds at `http://127.0.0.1:8000/api/latency` and the UI is reachable at `http://localhost:5173`.
```
