import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(path):
    y, sr = librosa.load(path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)

    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spektrogramm')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    main(args.path)