import os
import numpy as np
import librosa


def extract_features(audio, sample_rate, n_mfcc, n_fft, hop_length):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T


def get_padded(audio, expected_samples, mode='left'):
    if len(audio) >= expected_samples:
        return audio[:expected_samples]

    pad_len = expected_samples - len(audio)

    if mode == 'left':
        return np.pad(audio, (0, pad_len))
    elif mode == 'right':
        return np.pad(audio, (pad_len, 0))
    elif mode == 'center':
        half = pad_len // 2
        return np.pad(audio, (half, pad_len - half))
    elif mode == 'random':
        start = np.random.randint(0, pad_len + 1)
        return np.pad(audio, (start, pad_len - start))
    return audio


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

            # creates 5 different versions of each audio file
            # left, right, center and 2 random padding
            variations = []
            variations.append(get_padded(raw_audio, expected_samples, 'left'))
            variations.append(get_padded(raw_audio, expected_samples, 'right')) 
            variations.append(get_padded(raw_audio, expected_samples, 'center'))

            for _ in range(2):
                v = get_padded(raw_audio, expected_samples, 'random')
                noise_amp = 0.005 * np.random.uniform() * np.amax(v)
                v = v + noise_amp * np.random.normal(size=v.shape)
                variations.append(v)

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
