"""
Debug script to print intermediate MFCC values for comparison with Arduino.
Run with a .wav file to see the exact values at each DSP stage.
"""
import os
import sys
import numpy as np
import librosa
from scipy.fftpack import dct

# Import the same params as training
from data_processing import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC

N_MELS = 40  # Match Arduino

def debug_mfcc_computation(wav_path: str, frame_idx: int = 0):
    """
    Compute MFCC step-by-step and print intermediate values.
    
    Args:
        wav_path: Path to .wav file
        frame_idx: Which frame to print detailed debug for (default: 0)
    """
    print(f"\n{'='*60}")
    print(f"DEBUG MFCC COMPUTATION")
    print(f"File: {wav_path}")
    print(f"{'='*60}\n")
    
    # Load audio
    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    print(f"=== Audio Loaded ===")
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio)/sr:.3f}s ({len(audio)} samples)")
    print(f"Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # Pad/trim to 1 second
    expected_samples = int(SAMPLE_RATE * 1.0)
    if len(audio) < expected_samples:
        audio = np.pad(audio, (0, expected_samples - len(audio)))
    else:
        audio = audio[:expected_samples]
    
    print(f"\nAfter pad/trim: {len(audio)} samples")
    
    # Compute number of frames
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    print(f"Number of frames: {n_frames}")
    print(f"\nAnalyzing frame {frame_idx}...")
    
    # Get the audio window for this frame
    start = frame_idx * HOP_LENGTH
    end = start + N_FFT
    audio_window = audio[start:end]
    
    print(f"\n=== Audio Window (frame {frame_idx}) ===")
    print(f"Sample indices: [{start}:{end}]")
    print(f"First 10 samples: {audio_window[:10]}")
    print(f"Last 10 samples: {audio_window[-10:]}")
    print(f"Window stats - Min: {audio_window.min():.6f}, Max: {audio_window.max():.6f}, RMS: {np.sqrt(np.mean(audio_window**2)):.6f}")
    
    # Apply Hamming window
    hamming = np.hamming(N_FFT)
    windowed = audio_window * hamming
    
    # Compute FFT
    fft_result = np.fft.rfft(windowed)
    
    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_result) ** 2
    
    print(f"\n=== FFT Power Spectrum ===")
    print(f"Number of bins: {len(power_spectrum)}")
    print(f"Bins 0-9: {power_spectrum[:10]}")
    print(f"Bins 100-109: {power_spectrum[100:110]}")
    max_bin = np.argmax(power_spectrum)
    max_freq = max_bin * SAMPLE_RATE / N_FFT
    print(f"Max power at bin {max_bin} ({max_freq:.1f} Hz): {power_spectrum[max_bin]:.6f}")
    
    # Create Mel filterbank (same as librosa)
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=SAMPLE_RATE/2,
        htk=False,
        norm='slaney'
    )
    
    # Apply mel filterbank
    mel_energies = np.dot(mel_basis, power_spectrum)
    
    print(f"\n=== Mel Energies (BEFORE log) ===")
    for i in range(0, N_MELS, 10):
        print(f"Mels {i}-{min(i+9, N_MELS-1)}: {mel_energies[i:i+10]}")
    
    # Apply log (same as Arduino: log(x + 1e-6))
    log_mel_energies = np.log(mel_energies + 1e-6)
    
    print(f"\n=== Mel Energies (AFTER log) ===")
    for i in range(0, N_MELS, 10):
        print(f"Mels {i}-{min(i+9, N_MELS-1)}: {log_mel_energies[i:i+10]}")
    
    # Create DCT matrix (same as Arduino)
    dct_matrix = np.zeros((N_MFCC, N_MELS), dtype=np.float32)
    for k in range(N_MFCC):
        for n in range(N_MELS):
            dct_matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / N_MELS)
    
    # Apply orthonormal normalization
    dct_matrix[0, :] *= 1.0 / np.sqrt(N_MELS)
    dct_matrix[1:, :] *= np.sqrt(2.0 / N_MELS)
    
    # Compute MFCCs
    mfcc = np.dot(dct_matrix, log_mel_energies)
    
    print(f"\n=== MFCC Coefficients ===")
    for k in range(N_MFCC):
        print(f"MFCC[{k}]: {mfcc[k]:.6f}")
    
    # Now compare with librosa's built-in MFCC
    print(f"\n=== Comparison with librosa.feature.mfcc ===")
    librosa_mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=SAMPLE_RATE, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    
    print(f"librosa MFCC shape: {librosa_mfcc.shape}")
    print(f"librosa MFCC for frame {frame_idx}:")
    for k in range(N_MFCC):
        diff = mfcc[k] - librosa_mfcc[k, frame_idx]
        print(f"  MFCC[{k}]: {librosa_mfcc[k, frame_idx]:.6f} (diff: {diff:.6f})")
    
    print(f"\n{'='*60}")
    print("SUMMARY: Values above should match Arduino debug output")
    print("If they don't match, the mismatch is in the DSP pipeline")
    print(f"{'='*60}\n")
    
    return mfcc, librosa_mfcc[:, frame_idx]


def main():
    if len(sys.argv) < 2:
        # Try to find a sample file from validation set
        config_dir = os.path.dirname(__file__)
        possible_paths = [
            os.path.join(config_dir, "../recorder/recordings/noise_val"),
            os.path.join(config_dir, "../recorder/recordings/go_val"),
            os.path.join(config_dir, "../recorder/recordings/stop_val"),
        ]
        
        wav_file = None
        for path in possible_paths:
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.wav')]
                if files:
                    wav_file = os.path.join(path, files[0])
                    break
        
        if wav_file is None:
            print("Usage: python debug_mfcc.py <path_to_wav_file>")
            print("Or place .wav files in recorder/recordings/*_val folders")
            sys.exit(1)
    else:
        wav_file = sys.argv[1]
    
    if not os.path.exists(wav_file):
        print(f"Error: File not found: {wav_file}")
        sys.exit(1)
    
    # Debug the first frame
    debug_mfcc_computation(wav_file, frame_idx=0)
    
    # Also debug a later frame to see variation
    print("\n\n--- Frame 15 (middle of audio) ---")
    debug_mfcc_computation(wav_file, frame_idx=15)


if __name__ == "__main__":
    main()
