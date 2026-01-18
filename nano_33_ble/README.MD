## NANO 33 BLE Audio Recorder

### Setup
1. Flash the `recorder.ino` sketch to your Arduino Nano 33 BLE Sense Rev 2
2. Hold the microphone about 15 cm away from your mouth while recording

### Triggered Recorder 
Use this for clean keyword recordings with proper timing:

```bash
python recorder_triggered.py -p COM5 -d recordings/go -l go
```

- Press **SPACE** to start a 1-second recording
- You'll hear a beep and see a prompt - say the keyword immediately
- Press **Q** to quit

## Resources

* https://www.tensorflow.org/tutorials/audio/simple\_audio
* https://users.ece.utexas.edu/~bevans/courses/rtdsp/lectures/10\_Data\_Conversion/AP\_Understanding\_PDM\_Digital\_Audio.pdf
* https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53











