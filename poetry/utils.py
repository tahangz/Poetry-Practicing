import re
import numpy as np
import soundfile as sf

def float_to_int16(x: np.ndarray):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def write_wav(path: str, pcm16: np.ndarray, sr: int = 16000):
    sf.write(path, pcm16.astype('int16'), sr, subtype='PCM_16')

def rms_float32(x: np.ndarray):
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=float)))

def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(text.split())
