import numpy as np
import librosa
import os

def convert_wav_to_header(wav_path, output_header_path, sample_rate=16000):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(src_dir, "..", wav_path)
    wav_path = os.path.abspath(wav_path)
    
    output_header_path = os.path.join(src_dir, "..", output_header_path, "audio_inject.h")
    output_header_path = os.path.abspath(output_header_path)

    print(f"Converting {wav_path} to {output_header_path}")

    audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)

    audio_int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)

    var_name = "test_audio_data"
    len_name = "test_audio_len"
    
    with open(output_header_path, 'w') as f:
        f.write("#ifndef TEST_AUDIO_H\n")
        f.write("#define TEST_AUDIO_H\n\n")
        f.write(f"// Auto-generated from {os.path.basename(wav_path)}\n")
        f.write(f"const int {len_name} = {len(audio_int16)};\n")
        f.write(f"const short {var_name}[] = {{\n")
        
        for i, val in enumerate(audio_int16):
            f.write(f"{val}")
            if i < len(audio_int16) - 1:
                f.write(", ")
            if (i + 1) % 16 == 0:
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write("#endif // TEST_AUDIO_H\n")

    print(f"Done, written {len(audio_int16)} samples.")
