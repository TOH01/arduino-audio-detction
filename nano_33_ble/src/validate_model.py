import os
import re
import json
import csv
import numpy as np
import tensorflow as tf
import argparse
from data_processing import load_motion_data

def load_normalization_params(header_path):
    params = {}
    with open(header_path, 'r') as f:
        content = f.read()
        params["ACC_MEAN"] = float(re.search(r"#define ACC_MEAN ([\d\.\-eE]+)", content).group(1))
        params["ACC_STD"] = float(re.search(r"#define ACC_STD ([\d\.\-eE]+)", content).group(1))
        params["GYRO_MEAN"] = float(re.search(r"#define GYRO_MEAN ([\d\.\-eE]+)", content).group(1))
        params["GYRO_STD"] = float(re.search(r"#define GYRO_STD ([\d\.\-eE]+)", content).group(1))
    return params

def normalize_data(imu_data, norm_params):
    normalized = imu_data.copy()
    normalized[:, 0:3] = (imu_data[:, 0:3] - norm_params["ACC_MEAN"]) / norm_params["ACC_STD"]
    normalized[:, 3:6] = (imu_data[:, 3:6] - norm_params["GYRO_MEAN"]) / norm_params["GYRO_STD"]
    return normalized

def split_into_windows(data, window_size_samples):
    num_windows = len(data) // window_size_samples
    return np.array_split(data[:num_windows * window_size_samples], num_windows)

def run_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

def main():
    parser = argparse.ArgumentParser(description='Batch test TFLite model with config')
    parser.add_argument('config_json', help='Path to config JSON file')
    args = parser.parse_args()

    # Load config
    with open(args.config_json, 'r') as f:
        config = json.load(f)

    out_dir = config.get('output_dir', '.')
    os.makedirs(out_dir, exist_ok=True)

    # Load normalization params
    norm_header_path = os.path.join(out_dir, 'normalization.h')
    norm_params = load_normalization_params(norm_header_path)

    # Find model file in output_dir
    tflite_files = [f for f in os.listdir(out_dir) if f.endswith('.tflite')]
    if not tflite_files:
        print("No .tflite model found in output_dir!")
        return
    model_path = os.path.join(out_dir, tflite_files[0])

    # Load model interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    results = []
    for motion in config['motions']:
        name = motion['name']
        label = motion['label']
        val_path = motion.get('validation_data_path')
        if val_path is None:
            print(f"Skipping motion '{name}': no validation_data_path")
            continue

        print(f"Evaluating motion '{name}'...")

        imu_data = np.array(load_motion_data(val_path))  # raw 6-axis IMU data
        norm_data = normalize_data(imu_data, norm_params)

        # Determine window size
        motion_duration = config['motion_duration']
        sample_rate = config['sample_rate']
        window_size_samples = int(motion_duration * sample_rate)

        # Split into windows
        windows = split_into_windows(norm_data, window_size_samples)

        # Prepare model input info
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']

        correct = 0
        total = 0

        for window in windows:
            input_data = window.reshape(1, window_size_samples, 6)

            if input_dtype == np.int8:
                scale, zero_point = input_details[0]['quantization']
                input_data = (input_data / scale + zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(np.float32)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            # Dequantize output if needed
            if output_details[0]['dtype'] == np.int8:
                out_scale, out_zero_point = output_details[0]['quantization']
                output = out_scale * (output.astype(np.float32) - out_zero_point)

            predicted_label = np.argmax(output)
            confidence = np.max(output)
            is_correct = predicted_label == label

            correct += int(is_correct)
            total += 1

            results.append({
                'motion': name,
                'expected_label': label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': is_correct
            })

        acc = correct / total if total > 0 else 0
        print(f"  Accuracy: {acc:.2%} ({correct}/{total})")
    

    csv_path = os.path.join(out_dir, f"{config.get('name', 'results')}_evaluation.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
