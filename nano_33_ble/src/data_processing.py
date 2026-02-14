import os
import numpy as np
import librosa


def extract_features(audio, sample_rate, n_mfcc, n_fft, hop_length):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T


def get_padded(audio, expected_samples):
    if len(audio) >= expected_samples:
        return audio[:expected_samples]

    pad_len = expected_samples - len(audio)

    return np.pad(audio, (0, pad_len))


def load_dataset_from_config(sample_rate, expected_samples, n_mfcc, n_fft, hop_length, motions):
    X = []
    y = []

    for motion in motions:
        label = motion["label"]
        name = motion["name"]
        data_count = 0
        folder_path = os.path.join(os.getcwd(), motion["data_path"])

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Loading {name}: {len(files)} files found")

        for file in files:
            file_path = os.path.join(folder_path, file)

            raw_audio, _ = librosa.load(file_path, sr=sample_rate)

            # Original audio (center padded)
            variations = [get_padded(raw_audio, expected_samples)]

            # Time Stretching (https://arxiv.org/pdf/1608.04363)
            time_stretch_factors = [0.81, 0.93, 1.07, 1.23]
            for rate in time_stretch_factors:
                y_stretched = librosa.effects.time_stretch(raw_audio, rate=rate)
                variations.append(get_padded(y_stretched, expected_samples))

            # Pitch Shifting (https://arxiv.org/pdf/1608.04363)
            pitch_shift_semitones = [-2, -1, 1, 2]
            for steps in pitch_shift_semitones:
                y_shifted = librosa.effects.pitch_shift(raw_audio, sr=sample_rate, n_steps=steps)
                variations.append(get_padded(y_shifted, expected_samples))

            for v in variations:
                feat = extract_features(v, sample_rate, n_mfcc, n_fft, hop_length)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
                    data_count += 1

        print(f"Generated {data_count} samples for motion '{name}'")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    if len(X) > 0:
        X = X[..., np.newaxis]

    return X, y
