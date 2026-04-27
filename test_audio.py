import soundfile as sf
import numpy as np
import os

def create_test_wav(filename='test.wav', duration=2.0, freq=440.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, audio, sr)
    print(f"Generated {filename}")

if __name__ == "__main__":
    create_test_wav('test.wav')
    audio, sr = sf.read('test.wav')
    print('Loaded:', len(audio), 'samples at', sr, 'Hz')
