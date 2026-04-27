import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import json
from signal_scope import SignalScope

def analyze_audio(file_path, live_visual=False):
    # 1. Load audio
    audio, sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Mono mixdown
    
    # 2. Frame segmentation
    frame_size = 1024
    hop_size = 512
    frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, hop_size)]
    
    # Pre-compute Hilbert for phase
    analytic_signal = signal.hilbert(audio)
    unwrapped_phases = np.unwrap(np.angle(analytic_signal))
    
    scope = SignalScope(beta=0.1)
    history = []
    
    print(f"Processing {len(frames)} frames...")

    for i, frame in enumerate(frames):
        start_idx = i * hop_size
        
        # 3. Extract W1 amplitude (RMS)
        rms = np.sqrt(np.mean(frame**2))
        
        # 4. Extract W2 phase (phase progression)
        frame_phases = unwrapped_phases[start_idx:start_idx+frame_size]
        w2_val = np.mean(np.diff(frame_phases))
        
        # 5. Extract W3 frequency (spectral centroid)
        fft_data = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1/sr)
        if np.sum(fft_data) > 0:
            centroid = np.sum(freqs * fft_data) / np.sum(fft_data)
        else:
            centroid = 0

        # Normalize raw features for initial input
        # Note: SignalScope.normalize handles clamping and sum-to-1
        raw_features = np.array([rms, abs(w2_val), centroid / (sr/2)])
        
        # Update SignalScope
        frame_results = scope.update(raw_features)
        
        # Add timestamp, log events, and raw features
        for level in frame_results:
            frame_results[level]['t'] = i * hop_size / sr
            frame_results[level]['raw_features'] = raw_features.tolist()
            if frame_results[level]['events']:
                print(f"[{frame_results[level]['t']:.2f}s] {level.upper()} Events: {frame_results[level]['events']}")
            
        history.append(frame_results)

    # 6. Save logs (JSONL as per spec)
    os.makedirs('logs', exist_ok=True)
    with open('logs/signal_scope.jsonl', 'w') as f:
        for entry in history:
            f.write(json.dumps(entry) + '\n')
            
    # 7. Plotting
    generate_hierarchical_plots(history)
    
    if live_visual:
        print("Live visualization not implemented in this version. Check outputs/analysis_plot.png")

def generate_hierarchical_plots(history):
    levels = ['local', 'global', 'meta']
    num_levels = len(levels)
    
    fig, axes = plt.subplots(num_levels, 3, figsize=(18, 12), sharex=True)
    
    for i, level in enumerate(levels):
        t = [h[level]['t'] for h in history]
        W = np.array([h[level]['W'] for h in history])
        C = [h[level]['C'] for h in history]
        E = [h[level]['E_scope'] for h in history]
        
        # Participation Plot
        axes[i, 0].plot(t, W[:, 0], 'r', label='W1 (Amp)', alpha=0.7)
        axes[i, 1].plot(t, W[:, 1], 'g', label='W2 (Phase)', alpha=0.7)
        axes[i, 2].plot(t, W[:, 2], 'b', label='W3 (Freq)', alpha=0.7)
        axes[i, 0].set_ylabel(f'{level.capitalize()} Participation')
        
        # Coupling & Imbalance Plot (Overlay or separate?)
        # Let's use twinx for C and E
        ax_right = axes[i, 2].twinx()
        ax_right.plot(t, C, 'k--', label='Coupling (C)', alpha=0.5)
        ax_right.plot(t, E, 'm:', label='Imbalance (E)', alpha=0.5)
        
    axes[0, 0].set_title('W1 Participation (Red)')
    axes[0, 1].set_title('W2 Participation (Green)')
    axes[0, 2].set_title('W3 Participation (Blue) / C, E (Right Axis)')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/analysis_plot.png')
    print("Plot saved to outputs/analysis_plot.png")

if __name__ == "__main__":
    if not os.path.exists('test.wav'):
        print("Generating test.wav...")
        # Simple test signal generation if missing
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Chirp-like signal for dynamic activity
        audio = 0.5 * np.sin(2 * np.pi * (220 + 440 * t) * t) 
        sf.write('test.wav', audio, sr)
        
    analyze_audio('test.wav')
