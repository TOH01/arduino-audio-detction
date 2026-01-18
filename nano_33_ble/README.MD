# NANO 33 BLE Audio Keyword Detection

TinyML keyword detection for Arduino Nano 33 BLE Sense Rev 2.

## Workflow Overview

```
1. Record Data  →  2. Train Model  →  3. Validate  →  4. Deploy
```

## 1. Recording Data

### Setup
1. Flash `recorder/recorder.ino` to your Arduino Nano 33 BLE Sense Rev 2
2. Hold the microphone ~15 cm from your mouth

### Triggered Recorder
Use for clean keyword recordings with proper timing:

```bash
python recorder/recorder_triggered.py -p COM5 -d recordings/go -l go
```

- Press **SPACE** to start a 1-second recording
- Say the keyword immediately after the beep
- Press **Q** to quit

## 2. Training

Train a new model using VS Code:
- Open **Run and Debug** panel (Ctrl+Shift+D)
- Select **"Train Audio Model"**
- Press F5

Or via command line:
```bash
cd nano_33_ble
python src/main.py --config config.json
```

Output files are saved to `audio_output/`:
- `audio_model.tflite` - Quantized model
- `audio_model.h` - C header for Arduino
- `dsp_params.h` - Audio preprocessing constants
- `model_config.h` - Class labels and settings

## 3. Validation

Test the model on held-out validation data:
- Select **"Validate Audio Model"** in VS Code
- Press F5

Or via command line:
```bash
cd nano_33_ble
python src/validate_model.py config.json
```

Results are saved to `audio_output/audio_model_evaluation.csv`

## 4. Deployment

Copy the header files from `audio_output/` to `deploy/` and flash `deploy/deploy.ino`.

## Resources

- [TensorFlow Audio Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Understanding PDM Audio](https://users.ece.utexas.edu/~bevans/courses/rtdsp/lectures/10_Data_Conversion/AP_Understanding_PDM_Digital_Audio.pdf)
- [Mel Spectrograms Explained](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
