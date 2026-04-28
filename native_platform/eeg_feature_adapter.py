import numpy as np
from scipy import signal as signal_lib

def window_signal(sig, sample_rate, window_sec=1.0, overlap=0.5):
    """
    Splits signal into overlapping windows.
    Returns list of windows.
    """
    win_len = int(window_sec * sample_rate)
    step = int(win_len * (1.0 - overlap))
    
    windows = []
    for start in range(0, len(sig) - win_len + 1, step):
        windows.append(sig[start : start + win_len])
    return windows

def extract_window_features(win, sample_rate):
    """
    Extracts spectral and temporal features from a signal window.
    Returns a dictionary of features.
    """
    # 1. RMS Energy
    rms = np.sqrt(np.mean(win**2))
    
    # 2. Spectral features via FFT
    freqs, psd = signal_lib.welch(win, sample_rate, nperseg=len(win))
    
    # Avoid zero division
    psd_sum = np.sum(psd) if np.sum(psd) > 0 else 1.0
    
    # 3. Spectral Centroid
    centroid = np.sum(freqs * psd) / psd_sum
    
    # 4. Band Powers
    def get_band_power(f_low, f_high):
        idx = np.logical_and(freqs >= f_low, freqs <= f_high)
        return float(np.sum(psd[idx]))
    
    delta = get_band_power(0.5, 4.0)
    theta = get_band_power(4.0, 8.0)
    alpha = get_band_power(8.0, 13.0)
    beta = get_band_power(13.0, 30.0)
    
    # 5. Proxies
    # Phase proxy (rough estimate using angle of mean FFT)
    fft_val = np.fft.fft(win)
    phase_proxy = float(np.angle(np.mean(fft_val)))
    
    # Entropy proxy (Shannon entropy of normalized PSD)
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    
    return {
        "rms_energy": float(rms),
        "spectral_centroid": float(centroid),
        "band_power_delta": delta,
        "band_power_theta": theta,
        "band_power_alpha": alpha,
        "band_power_beta": beta,
        "phase_proxy": phase_proxy,
        "entropy_proxy": float(entropy)
    }

def features_to_scope_input(feats):
    """
    Simplified mapping for EEG: only use the most stable features.
    Produces 9 elements (3 per region) by repeating or grouping.
    """
    # Region 1: Magnitude/Rhythm
    r1 = [feats["rms_energy"], feats["band_power_alpha"], feats["rms_energy"]]
    
    # Region 2: Complexity
    r2 = [feats["entropy_proxy"] / 10.0, feats["spectral_centroid"] / 50.0, feats["entropy_proxy"] / 10.0]
    
    # Region 3: Phase/Temporal
    r3 = [feats["phase_proxy"] / np.pi, feats["rms_energy"], feats["phase_proxy"] / np.pi]
    
    return np.array(r1 + r2 + r3)

def signal_to_input_frames(sig, sample_rate, window_sec=2.0, overlap=0.9):
    """
    Full pipeline: Signal -> Windows -> Features -> Scope Inputs.
    Uses larger window for more stable spectral estimation.
    """
    windows = window_signal(sig, sample_rate, window_sec, overlap)
    raw_frames = []
    for win in windows:
        feats = extract_window_features(win, sample_rate)
        scope_input = features_to_scope_input(feats)
        raw_frames.append(scope_input)
    
    # Simple Moving Average smoothing (3-frame window)
    if len(raw_frames) < 3:
        return raw_frames
        
    smooth_frames = []
    for i in range(len(raw_frames)):
        start = max(0, i - 1)
        end = min(len(raw_frames), i + 2)
        window = raw_frames[start:end]
        smooth_frames.append(np.mean(window, axis=0))
        
    return smooth_frames
