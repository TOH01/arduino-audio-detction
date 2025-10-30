# Arduino Audio Detection

This guide covers training a model using [Arduino ML Tools](https://mltools.arduino.cc) and deploying it to an Arduino device.

## Prerequisites

* **Arduino CLI v0.34.2**
  Install from the [GitHub release page](https://github.com/arduino/arduino-cli/releases) and add it to your PATH variable.

* **Python serial library**

```bash
pip install pyserial
```

## Arduino Firmware

1. Download the Arduino firmware for your board (or use the ZIP from the repo):
   [Arduino Nicla Voice Firmware Documentation](https://docs.edgeimpulse.com/hardware/boards/arduino-nicla-voice)

2. Install the firmware using the flash script for your OS, e.g.:

```
flash_windows.bat
```

3. After flashing, the NDP120 should contain the following files:

* `ei_model.synpkg`
* `mcu_fw_120_v91.synpkg`
* `dsp_firmware_v91.synpkg`

## Edge Impulse Firmware for Arduino Nicla Voice

1. Clone or download the repository:
   [edgeimpulse/firmware-arduino-nicla-voice](https://github.com/edgeimpulse/firmware-arduino-nicla-voice)

2. Follow the instructions in `README.md` to both **build** and **flash** the firmware.

## Demo

1. Open the Arduino IDE.
2. Run the demo model.

* The model should now recognize the keywords **go** and **stop**

## Data
https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
