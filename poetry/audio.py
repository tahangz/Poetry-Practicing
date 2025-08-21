import sys, time, queue
import numpy as np
import sounddevice as sd
from collections import deque
import webrtcvad
from .config import *

class AudioProducer:
    def __init__(self, samplerate=TARGET_SR, channels=CHANNELS, blocksize=BLOCKSIZE):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.stream, self.running = None, False

    def _callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status, file=sys.stderr)
        arr = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.q.put(arr)

    def start(self):
        if self.running: return
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                     blocksize=self.blocksize, dtype='float32', callback=self._callback)
        self.stream.start()
        self.running = True

    def stop(self):
        if not self.running: return
        self.stream.stop(); self.stream.close(); self.running = False

    def read_seconds(self, seconds: float, timeout=5.0):
        samples_needed = int(self.samplerate * seconds)
        frames, got, start = [], 0, time.time()
        while got < samples_needed and (time.time() - start) < timeout:
            try:
                f = self.q.get(timeout=timeout)
                frames.append(f); got += f.shape[0]
            except queue.Empty:
                break
        if not frames: return np.array([], dtype=np.float32)
        arr = np.concatenate(frames).astype(np.float32)
        if arr.shape[0] > samples_needed: arr = arr[:samples_needed]
        elif arr.shape[0] < samples_needed: arr = np.pad(arr, (0, samples_needed - arr.shape[0]))
        return arr

class NoiseEstimator:
    def __init__(self, sr=TARGET_SR, history_seconds=NOISE_HISTORY_SECONDS):
        self.sr = sr
        self.history_frames = int((history_seconds * sr) / (VAD_FRAME_MS * sr / 1000))
        self.rms_history = deque(maxlen=max(10, self.history_frames))

    def update_if_silent(self, rms_value):
        self.rms_history.append(rms_value)

    def estimated_noise_floor(self):
        if not self.rms_history: return 1e-6
        vals = sorted(self.rms_history)
        idx = max(0, int(0.2 * len(vals)) - 1)
        return max(vals[idx], 1e-7)

def vad_decision(pcm16_bytes: bytes, sample_rate=TARGET_SR, aggressiveness=VAD_AGGRESSIVENESS, frame_ms=VAD_FRAME_MS):
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sample_rate * frame_ms / 1000) * 2
    n_total, n_voiced = 0, 0
    for i in range(0, len(pcm16_bytes), frame_size):
        frame = pcm16_bytes[i:i + frame_size]
        if len(frame) < frame_size: break
        n_total += 1
        try:
            if vad.is_speech(frame, sample_rate=sample_rate): n_voiced += 1
        except Exception: pass
    return n_voiced, n_total
