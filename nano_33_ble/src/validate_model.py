import os
import json
import csv
import numpy as np
import tensorflow as tf
import argparse
import librosa
from data_processing import extract_features


def load_validation_samples(folder_path):
    samples = []
    if not os.path.exists(folder_path):
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


def main(config):
    out_dir = config["output_dir"]
    model_name = config["name"]
    sample_rate = config["sample_rate"]
    duration = config["audio_duration"]
    n_mfcc = config["n_mfcc"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    confidence = config["confidence"]
    motions = config["motions"]

    expected_samples = int(sample_rate * duration)
    model_path = os.path.join(out_dir, f"{model_name}.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    results = []
    total_correct = 0
    total_samples = 0

    for motion in motions:
        name = motion['name']
        label = motion['label']
        val_path = motion.get('validation_data_path')

        folder_path = os.path.join(os.getcwd(), val_path)

        wav_files = load_validation_samples(folder_path)

        correct = 0
        total = 0

        for wav_path in wav_files:
 
            audio, _ = librosa.load(wav_path, sr=sample_rate)

            if len(audio) < expected_samples:
                audio = np.pad(audio, (0, expected_samples - len(audio)))
            else:
                audio = audio[:expected_samples]
                    
            mfcc = extract_features(audio, sample_rate, n_mfcc, n_fft, hop_length)

            input_data = mfcc[np.newaxis, ..., np.newaxis]

            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.int8)

            output, _, _ = run_inference(interpreter, input_data)

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
        print(f"Accuracy: {acc:.2%} ({correct}/{total})")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
