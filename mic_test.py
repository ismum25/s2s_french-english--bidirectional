# save as mic_test.py and run: python mic_test.py
import time
import numpy as np
import sounddevice as sd

sr = 48000
dev = 12

def cb(indata, frames, time_info, status):
    if status:
        print("STATUS:", status)
    rms = float(np.sqrt(np.mean(indata[:,0]**2)))
    print(f"RMS: {rms:.6f}")

with sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32", callback=cb):
    print("Speak for 5 seconds...")
    time.sleep(5)