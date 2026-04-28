import numpy as np

def normalize_signal(signal):
    """Normalize signal to range [-1, 1]"""
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal

def generate_alpha(sample_rate, duration, freq=10.0, noise=0.02):
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Alpha rhythm often has some amplitude modulation (waxing/waning)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)
    signal = am * np.sin(2 * np.pi * freq * t)
    signal += np.random.normal(0, noise, len(t))
    return normalize_signal(signal)

def generate_theta(sample_rate, duration, freq=6.0, noise=0.02):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)
    signal += np.random.normal(0, noise, len(t))
    return normalize_signal(signal)

def generate_beta(sample_rate, duration, freq=20.0, noise=0.02):
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * freq * t)
    signal += np.random.normal(0, noise, len(t))
    return normalize_signal(signal)

def generate_mixed_alpha_theta(sample_rate, duration):
    alpha = generate_alpha(sample_rate, duration, freq=10.0, noise=0.01)
    theta = generate_theta(sample_rate, duration, freq=6.0, noise=0.01)
    return normalize_signal(alpha + 0.5 * theta)

def generate_alpha_with_noise(sample_rate, duration, noise=0.15):
    return generate_alpha(sample_rate, duration, freq=10.0, noise=noise)

def generate_alpha_tail_removed(sample_rate, duration, keep_ratio=0.6):
    signal = generate_alpha(sample_rate, duration)
    cut_idx = int(len(signal) * keep_ratio)
    masked = signal.copy()
    masked[cut_idx:] = 0.0
    return masked

def generate_alpha_to_spike_burst(sample_rate, duration, transition_ratio=0.7):
    t = np.linspace(0, duration, int(sample_rate * duration))
    split_idx = int(len(t) * transition_ratio)
    
    alpha_part = generate_alpha(sample_rate, duration * transition_ratio, freq=10.0)
    
    # Generate spike burst (simulated undesired event)
    t_spike = t[split_idx:]
    # High frequency bursts with sharp spikes
    spike_part = np.sin(2 * np.pi * 30.0 * t_spike) * 2.0 
    spike_part += 3.0 * (np.random.rand(len(t_spike)) > 0.95).astype(float) # Random spikes
    
    signal = np.concatenate([alpha_part, spike_part])
    return normalize_signal(signal)

def generate_dropout(signal, dropout_ratio=0.1):
    new_signal = signal.copy()
    duration = len(signal)
    start = int(duration * 0.45)
    end = int(start + duration * dropout_ratio)
    new_signal[start:end] = 0.0
    return new_signal
