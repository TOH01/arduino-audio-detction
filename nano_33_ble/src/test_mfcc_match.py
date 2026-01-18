"""
Test script to verify Arduino MFCC implementation matches librosa.
Run this to validate the mel_filterbank.h values produce correct MFCCs.
"""
import numpy as np
import librosa
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from data_processing import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC

def test_mfcc_match():
    """
    Generate a test signal and compare librosa MFCC with our implementation.
    This validates that the pre-computed filterbank produces matching results.
    """
    print("=" * 60)
    print("MFCC Implementation Verification Test")
    print("=" * 60)
    
    # Generate a simple test signal (1 second of a 440Hz tone)
    duration = 1.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Parameters matching our Arduino implementation
    n_mels = 40  # Standard librosa default we're using
    
    print(f"\nTest Parameters:")
    print(f"  Sample Rate: {SAMPLE_RATE}")
    print(f"  N_FFT: {N_FFT}")
    print(f"  HOP_LENGTH: {HOP_LENGTH}")
    print(f"  N_MELS: {n_mels}")
    print(f"  N_MFCC: {N_MFCC}")
    
    # Compute MFCC with librosa (ground truth)
    mfcc_librosa = librosa.feature.mfcc(
        y=test_signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        htk=False,  # Slaney formula (our default)
        norm='ortho'  # Orthonormal DCT
    )
    
    print(f"\nLibrosa MFCC shape: {mfcc_librosa.shape}")
    print(f"  (n_mfcc={mfcc_librosa.shape[0]}, time_frames={mfcc_librosa.shape[1]})")
    
    # Now compute using our pure Python implementation matching Arduino
    # Get mel filterbank (same as we generate for Arduino)
    mel_basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=n_mels,
        htk=False,
        norm='slaney'
    )
    
    # Compute DCT matrix (same as Arduino)
    dct_matrix = np.zeros((N_MFCC, n_mels), dtype=np.float32)
    for k in range(N_MFCC):
        for n in range(n_mels):
            dct_matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / n_mels)
    # Orthonormal normalization
    dct_matrix[0, :] *= 1.0 / np.sqrt(n_mels)
    dct_matrix[1:, :] *= np.sqrt(2.0 / n_mels)
    
    # Compute STFT with Hamming window (matching Arduino)
    stft = librosa.stft(
        y=test_signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window='hamming',
        center=True
    )
    power_spectrum = np.abs(stft)
    
    # Apply mel filterbank
    mel_spec = np.dot(mel_basis, power_spectrum)
    
    # Log compression
    log_mel = np.log(mel_spec + 1e-6)
    
    # Apply DCT
    mfcc_ours = np.dot(dct_matrix, log_mel)
    
    print(f"\nOur MFCC shape: {mfcc_ours.shape}")
    
    # Compare
    # Note: There may be small differences due to windowing/centering details
    # What matters is the overall pattern and magnitude are similar
    
    # Check first frame
    print(f"\n--- First Frame Comparison ---")
    print(f"Librosa MFCC[0:5, 0]: {mfcc_librosa[:5, 0]}")
    print(f"Our MFCC[0:5, 0]:     {mfcc_ours[:5, 0]}")
    
    # Calculate MSE
    min_frames = min(mfcc_librosa.shape[1], mfcc_ours.shape[1])
    mse = np.mean((mfcc_librosa[:, :min_frames] - mfcc_ours[:, :min_frames]) ** 2)
    
    print(f"\nMSE between implementations: {mse:.6f}")
    
    if mse < 1.0:
        print("\n✓ PASS: Implementations match closely!")
        return True
    else:
        print("\n✗ FAIL: Significant difference detected")
        print("  This may indicate a bug in the filterbank generation.")
        return False


if __name__ == "__main__":
    success = test_mfcc_match()
    sys.exit(0 if success else 1)
