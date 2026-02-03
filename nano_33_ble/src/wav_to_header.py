from logging import warning
import numpy as np
import librosa
import os
import argparse

def convert_wav_to_header(wav_path, output_header_path, sample_rate=16000):
    """
    Reads a WAV file, resamples it if necessary, converts to int16,
    and writes it as a C array to a header file.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(src_dir, "..", wav_path)
    wav_path = os.path.abspath(wav_path)
    
    output_header_path = os.path.join(src_dir, "..", output_header_path, "audio_inject.h")
    output_header_path = os.path.abspath(output_header_path)

    print(f"Converting {wav_path} -> {output_header_path}")

    # Load audio 
    # librosa.load resamples automatically if sr is provided
    # mono=True mixes down to mono
    audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)

    # Convert float [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)

    # Generate C header content
    var_name = "test_audio_data"
    len_name = "test_audio_len"
    
    with open(output_header_path, 'w') as f:
        f.write("#ifndef TEST_AUDIO_H\n")
        f.write("#define TEST_AUDIO_H\n\n")
        f.write(f"// Auto-generated from {os.path.basename(wav_path)}\n")
        f.write(f"const int {len_name} = {len(audio_int16)};\n")
        f.write(f"const short {var_name}[] = {{\n")
        
        # Write data in lines of 16 values
        for i, val in enumerate(audio_int16):
            f.write(f"{val}")
            if i < len(audio_int16) - 1:
                f.write(", ")
            if (i + 1) % 16 == 0:
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write("#endif // TEST_AUDIO_H\n")

    print(f"Done! Written {len(audio_int16)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WAV to C header for Arduino audio injection.")
    parser.add_argument("wav_path", type=str, help="Path to input .wav file")
    parser.add_argument("output_path", type=str, help="Path to output .h file")
    args = parser.parse_args()

    convert_wav_to_header(args.wav_path, args.output_path)
