# Arduino Audio Detection

This guide covers training a model using [Arduino ML Tools](https://mltools.arduino.cc) and deploying it to an Arduino device.
On ML Tools website, a model can easily be trained. For settings, follow to screenshots in /edge_impuls
When done you will receive a folder with Syntiant firmware. To Deploy it on the Nicla Voice follow the next steps.

## Prerequisites

* **Arduino CLI v0.34.2**
  Install from the [GitHub release page](https://github.com/arduino/arduino-cli/releases) and add it to your PATH variable.

* **Python serial library**

```bash
pip install pyserial
```

## Arduino Firmware

1. Using the tools from /on_off_project install the NDP120 firmware (Created by training a model on Edge Impulse)

2. Install the firmware using the flash script for your OS, e.g.:

```
flash_windows.bat
```

3. After flashing, the NDP120 should contain the following files:

* `ei_model.synpkg`
* `mcu_fw_120_v91.synpkg`
* `dsp_firmware_v91.synpkg`

## Edge Impulse Firmware for Arduino Nicla Voice

1. Use the tools in /firmware-arduino-nicla-voice

2. To easily flash the firmware needed to interact with the NDP, run script for your os, e.g. ./arduino-win-build.bat --al

## Demo

1. Open the Arduino IDE.
2. Open serial monitor for the right port, set baud to 115200 and Both NL + CR
3. If everything worked correct, you can now interact with the NPD. Send "AT+HELP" to view list of commands

* The model should now recognize the keywords **on**, **off** and occasionally **noise**

## Data

The Data being used to train a model on ML Tools, is an open Dataset by Google.
Folders for keywords "on" and "off" were used. Both containt around 3750 .wav files.
For the noise classifier, the background_noise folder was used. Can just import those folder to
Edge Impulse Website, it will handle them.

https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
