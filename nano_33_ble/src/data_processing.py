from typing import List, Tuple
import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 1.0 

N_MFCC = 13
N_FFT = 2048 
HOP_LENGTH = 512



def extract_features(audio: np.ndarray) -> np.ndarray:
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        return mfcc.T 
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def load_dataset_from_config(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    expected_samples = int(SAMPLE_RATE * DURATION)
    X = []
    y = []

    for motion in config["motions"]:
        label = motion["label"]
        name = motion["name"]
        data_count = 0
        folder_path = os.path.join(os.getcwd(), motion["data_path"])
        
        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Loading {name}: {len(files)} files found.")

        for file in files:
            file_path = os.path.join(folder_path, file)
            try:
                # Load raw audio (variable length)
                raw_audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # We need to create multiple 1.0s versions of this clip
                # to make the model "position invariant".
                # This is CRITICAL for low latency.
                
                variations = []
                
                # Helper to pad to specific length
                def get_padded(audio, mode='left'):
                    if len(audio) >= expected_samples:
                        return audio[:expected_samples]
                    
                    pad_len = expected_samples - len(audio)
                    if mode == 'left': # Audio at Start (Pad End) - High Latency
                        return np.pad(audio, (0, pad_len))
                    elif mode == 'right': # Audio at End (Pad Start) - Low Latency!
                        return np.pad(audio, (pad_len, 0))
                    elif mode == 'center':
                        half = pad_len // 2
                        return np.pad(audio, (half, pad_len - half))
                    elif mode == 'random':
                        start = np.random.randint(0, pad_len + 1)
                        return np.pad(audio, (start, pad_len - start))
                    return audio

                # 1. Generate Alignments
                # "Right" alignment matches the moment the word enters the buffer (Instant detection)
                # "Left" alignment matches the moment the word is about to leave (1s delay)
                variations.append(get_padded(raw_audio, 'left'))
                variations.append(get_padded(raw_audio, 'right')) 
                variations.append(get_padded(raw_audio, 'center'))
                
                # 2. Add some random shifts/noises for robustness
                for _ in range(2):
                     v = get_padded(raw_audio, 'random')
                     # Add Noise
                     noise_amp = 0.005 * np.random.uniform() * np.amax(v)
                     v = v + noise_amp * np.random.normal(size=v.shape)
                     variations.append(v)
                
                # Process all variations
                for v in variations:
                    feat = extract_features(v)
                    if feat is not None:
                        X.append(feat)
                        y.append(label)
                        data_count += 1
                        
            except Exception as e:
                print(f"Error processing {file}: {e}")
                
        print(f"  -> Generated {data_count} samples for motion '{name}' (Multi-alignment)")    

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    if len(X) > 0:
        X = X[..., np.newaxis]

    return X, y