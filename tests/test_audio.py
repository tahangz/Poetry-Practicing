import numpy as np

from poetry.audio import NoiseEstimator, vad_decision

def test_noise_estimator():
    ne = NoiseEstimator()
    ne.update_if_silent(0.01)
    assert ne.estimated_noise_floor() > 0

def test_vad_decision_silence():
    silence = np.zeros(16000, dtype=np.int16).tobytes()
    voiced, total = vad_decision(silence)
    assert voiced <= total
