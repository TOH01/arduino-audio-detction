# NANO 33 BLE Audio Keyword Detection

TinyML audio keyword detection for Arduino Nano 33 BLE Sense Rev 2.

## Workflow Overview

```
1. Record Data  →  2. Train Model  →  3. Validate  →  4. Deploy
```

## 1. Recording Data

### Setup
1. Flash `recorder/recorder.ino` to your Arduino Nano 33 BLE Sense Rev 2
2. Hold the microphone ~15 cm from your mouth

### Triggered Recorder

Example (see argparse for details on params):

```bash
python recorder/recorder_triggered.py -p COM5 -d recordings/go -l go
```

## 2. Training

Train model using command line:
```bash
cd nano_33_ble
python src/main.py --config config.json
```

If needed, modify the provided json config. It is possible to change all training parameters without editing the code.

Output files are saved to specified output_dir in config:
- `audio_model.tflite` - Needed for validation
- `audio_model.h`, `dsp_params.h`, `model_config.h`, `mel_filterbank.h` - Needed for deploy.ino

## 3. Validation

Validate model using command line:
```bash
cd nano_33_ble
python src/validate_model.py config.json
```

Results will be saved in specified output dir. For validation to work properly `validation_data_path` needs to be set for each motion in the config.
Validation path should include new data, which the model will be validated against.

## 4. Deployment

Copy the required header files from Step 2 into the same location as deploy.ino.

## 5. Notes

Changing PDM mic gain needs to be done in multiple places `PDM.setGain(X);` in recorder.ino, `GAIN = X.X` in recorder.py
and lastly `#define GAIN_FACTOR 1.5f` aswell as `PDM.setGain(80);` in deploy.ino

## Resources

- [TensorFlow Audio Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)
- [Understanding PDM Audio](https://users.ece.utexas.edu/~bevans/courses/rtdsp/lectures/10_Data_Conversion/AP_Understanding_PDM_Digital_Audio.pdf)
- [Mel Spectrograms Explained](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)
