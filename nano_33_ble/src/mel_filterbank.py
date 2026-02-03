import numpy as np
import librosa
import os


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
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

    if fmax is None:
        fmax = sample_rate / 2.0

    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=False,
        norm='slaney'
    )

    dct_matrix = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct_matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / n_mels)

    dct_matrix[0, :] *= 1.0 / np.sqrt(n_mels)
    dct_matrix[1:, :] *= np.sqrt(2.0 / n_mels)

    n_bins = n_fft // 2 + 1

    file_path = os.path.join(out_dir, "mel_filterbank.h")

    with open(file_path, 'w') as f:
        f.write("// Auto-generated Mel filterbank matching librosa\n")
        f.write("#ifndef MEL_FILTERBANK_H\n")
        f.write("#define MEL_FILTERBANK_H\n\n")

        f.write(f"#define N_MEL_FILTERS {n_mels}\n")
        f.write(f"#define N_FFT_BINS {n_bins}\n")
        f.write(f"#define N_MFCC_COEFFS {n_mfcc}\n\n")

        f.write("// Mel filterbank weights [n_mels][n_fft_bins]\n")
        f.write("static const float mel_filterbank[N_MEL_FILTERS][N_FFT_BINS] = {\n")

        for i, row in enumerate(mel_basis):
            f.write("  {")
            values = ", ".join(f"{v:.3f}f" for v in row)
            f.write(values)
            f.write("}" + ("," if i < n_mels - 1 else "") + "\n")

        f.write("};\n\n")

        f.write("// DCT-II matrix for MFCC [n_mfcc][n_mels]\n")
        f.write("static const float dct_matrix[N_MFCC_COEFFS][N_MEL_FILTERS] = {\n")

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
