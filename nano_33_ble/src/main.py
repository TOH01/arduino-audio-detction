import tensorflow as tf
import os
import json
import argparse
from sklearn.model_selection import train_test_split
from data_processing import load_dataset_from_config
from model import train_model
from header_helper import generate_model_header, generate_model_config_header, generate_dsp_params_header, generate_mel_filterbank_header, convert_wav_to_header


def create_full_model_from_config(config):
    out_dir = config["output_dir"]
    model_name = config["name"]
    sample_rate = config["sample_rate"]
    duration = config["audio_duration"]
    n_mfcc = config["n_mfcc"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    motions = config["motions"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    confidence = config["confidence"]
    duration = config["audio_duration"]
    test_audio_path = config.get("inject_test_path", None)     # optional parameter

    X, y = load_dataset_from_config(sample_rate, int(sample_rate * duration), n_mfcc, n_fft, hop_length, motions)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1]]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # save model for validate_model.py
    with open(os.path.join(out_dir, f"{model_name}.tflite"), 'wb') as f:
        f.write(tflite_model)

    # auto header generation
    generate_model_header(out_dir, model_name, tflite_model)
    generate_dsp_params_header(out_dir, sample_rate, n_fft, hop_length, ((sample_rate * duration) // hop_length) + 1)
    generate_model_config_header(out_dir, model_name, motions, confidence, test_audio=test_audio_path)
    generate_mel_filterbank_header(out_dir, sample_rate, n_fft, n_mfcc)
    
    if test_audio_path:
        convert_wav_to_header(out_dir, test_audio_path, sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    create_full_model_from_config(config)
