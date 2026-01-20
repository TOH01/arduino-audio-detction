import os
import json
import csv
import numpy as np
import tensorflow as tf
import argparse
import librosa
from data_processing import extract_features


def load_validation_samples(folder_path: str) -> list:
    samples = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return samples

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.wav'):
            samples.append(os.path.join(folder_path, file))
    return samples


def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output, input_details, output_details


def main():
    parser = argparse.ArgumentParser(description='Validate TFLite audio model')
    parser.add_argument('config_json', help='Path to config JSON file')
    args = parser.parse_args()

    with open(args.config_json, 'r') as f:
        config = json.load(f)

    out_dir = config.get("output_dir", "")
    model_name = config.get("name", "")
    sample_rate = config.get("sample_rate")
    duration = config.get("audio_duration")
    n_mfcc = config.get("n_mfcc")
    n_fft = config.get("n_fft")
    hop_length = config.get("hop_length")

    if not (n_mfcc and n_fft and hop_length):
        raise RuntimeError("Librosas mfcc params missing in config")

    if sample_rate and duration:
        expected_samples = int(sample_rate * duration)
    else:
        raise RuntimeError("Invalid config, check sample_rate and audio_duration")

    model_path = os.path.join(out_dir, f"{model_name}.tflite")
    print(f"Using model: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    print(f"Model expects input shape: {expected_shape}, dtype: {input_dtype}")

    results = []
    total_correct = 0
    total_samples = 0

    for motion in config['motions']:
        name = motion['name']
        label = motion['label']
        val_path = motion.get('validation_data_path')

        if val_path is None:
            print(f"Skipping motion '{name}': no validation_data_path")
            continue

        folder_path = os.path.join(os.path.dirname(args.config_json), val_path)

        wav_files = load_validation_samples(folder_path)

        if not wav_files:
            print(f"No .wav files for {name} found!")
            continue

        correct = 0
        total = 0

        for wav_path in wav_files:
            try:
                audio, _ = librosa.load(wav_path, sr=sample_rate)

                # Pad/Crop to exact length
                if len(audio) < expected_samples:
                    audio = np.pad(audio, (0, expected_samples - len(audio)))
                else:
                    audio = audio[:expected_samples]
                    
                mfcc = extract_features(audio, sample_rate, n_mfcc, n_fft, hop_length)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
            if mfcc is None:
                continue

            # Reshape: (time_steps, n_mfcc) -> (1, time_steps, n_mfcc, 1)
            input_data = mfcc[np.newaxis, ..., np.newaxis]

            if input_dtype == np.int8:
                scale, zero_point = input_details[0]['quantization']
                input_data = (input_data / scale + zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(np.float32)

            output, _, _ = run_inference(interpreter, input_data)

            if output_details[0]['dtype'] == np.int8:
                out_scale, out_zero_point = output_details[0]['quantization']
                output = out_scale * (output.astype(np.float32) - out_zero_point)

            predicted_label = int(np.argmax(output))
            confidence = float(np.max(output))
            is_correct = predicted_label == label

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'file': os.path.basename(wav_path),
                'motion': name,
                'expected_label': label,
                'predicted_label': predicted_label,
                'confidence': f"{confidence:.4f}",
                'correct': is_correct
            })

        acc = correct / total if total > 0 else 0
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
        total_correct += correct
        total_samples += total

    if total_samples > 0:
        overall_acc = total_correct / total_samples
        print(f"\n{'='*40}")
        print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{total_samples})")

    if results:
        csv_path = os.path.join(out_dir, f"{model_name}_evaluation.csv")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=';')
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
