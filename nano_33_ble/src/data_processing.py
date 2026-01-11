from typing import List, Tuple
import os
import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 1.0 

N_MFCC = 13
N_FFT = 2048 
HOP_LENGTH = 512


def extract_features(file_path: str, expected_samples: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

        if len(audio) < expected_samples:
            audio = np.pad(audio, (0, expected_samples - len(audio)))
        else:
            audio = audio[:expected_samples]

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        return mfcc.T 
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
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

        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                data_count+=1
                feat = extract_features(os.path.join(folder_path, file), expected_samples)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
                
        print(f"Added {data_count} samples for motion {name}")    
                    

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    if len(X) > 0:
        X = X[..., np.newaxis]

    return X, y