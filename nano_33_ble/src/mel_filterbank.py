"""
Generate Mel filterbank and DCT coefficients matching librosa for Arduino deployment.
These are auto-generated during training and saved to the deploy folder.
"""
import numpy as np
import librosa
import os

def hz_to_mel(hz):
    """Convert Hz to Mel scale (matches librosa's htk=False default)."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    """Convert Mel to Hz scale."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def generate_mel_filterbank_header(
    out_dir: str,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    n_mels: int = 40,
    n_mfcc: int = 13,
    fmin: float = 0.0,
    fmax: float = None
):
    """
    Generate mel_filterbank.h with pre-computed filter weights matching librosa.
    
    Args:
        out_dir: Output directory for the header file
        sample_rate: Audio sample rate
        n_fft: FFT size
        n_mels: Number of Mel filter bands
        n_mfcc: Number of MFCC coefficients to extract
        fmin: Minimum frequency for Mel filters
        fmax: Maximum frequency (defaults to sample_rate/2)
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    
    # Generate Mel filterbank using librosa (this is the ground truth)
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,  # Use Slaney formula (librosa default)
        norm='slaney'
    )
    
    # Generate DCT-II matrix for MFCC computation
    # This matches librosa.filters.dct with norm='ortho'
    dct_matrix = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct_matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / n_mels)
    
    # Apply orthonormal normalization (matches librosa norm='ortho')
    dct_matrix[0, :] *= 1.0 / np.sqrt(n_mels)
    dct_matrix[1:, :] *= np.sqrt(2.0 / n_mels)
    
    # Sparse representation: for each Mel filter, store only non-zero entries
    # This saves a lot of memory on Arduino
    n_bins = n_fft // 2 + 1
    
    # Write header file
    file_path = os.path.join(out_dir, "mel_filterbank.h")
    
    with open(file_path, 'w') as f:
        f.write("// Auto-generated Mel filterbank matching librosa\n")
        f.write("// DO NOT EDIT - regenerate by running training pipeline\n\n")
        f.write("#ifndef MEL_FILTERBANK_H\n")
        f.write("#define MEL_FILTERBANK_H\n\n")
        
        f.write(f"#define N_MEL_FILTERS {n_mels}\n")
        f.write(f"#define N_FFT_BINS {n_bins}\n")
        f.write(f"#define N_MFCC_COEFFS {n_mfcc}\n\n")
        
        # Write Mel filterbank as dense array (simpler for Arduino)
        f.write("// Mel filterbank weights [n_mels][n_fft_bins]\n")
        f.write(f"static const float mel_filterbank[N_MEL_FILTERS][N_FFT_BINS] = {{\n")
        
        for i, row in enumerate(mel_basis):
            f.write("  {")
            # Format with limited precision to save space
            values = ", ".join(f"{v:.6f}f" for v in row)
            f.write(values)
            f.write("}" + ("," if i < n_mels - 1 else "") + "\n")
        
        f.write("};\n\n")
        
        # Write DCT matrix
        f.write("// DCT-II matrix for MFCC [n_mfcc][n_mels]\n")
        f.write(f"static const float dct_matrix[N_MFCC_COEFFS][N_MEL_FILTERS] = {{\n")
        
        for i, row in enumerate(dct_matrix):
            f.write("  {")
            values = ", ".join(f"{v:.6f}f" for v in row)
            f.write(values)
            f.write("}" + ("," if i < n_mfcc - 1 else "") + "\n")
        
        f.write("};\n\n")
        
        f.write("#endif // MEL_FILTERBANK_H\n")
    
    print(f"Generated {file_path}")
    print(f"  - Mel filters: {n_mels} x {n_bins}")
    print(f"  - DCT matrix: {n_mfcc} x {n_mels}")
    
    return file_path


if __name__ == "__main__":
    # Test generation
    from data_processing import SAMPLE_RATE, N_FFT, N_MFCC
    generate_mel_filterbank_header(
        out_dir="./audio_output",
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=40,
        n_mfcc=N_MFCC
    )
