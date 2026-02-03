# Arduino Audio Detection

This guide covers training a model using [Arduino ML Tools](https://mltools.arduino.cc) and deploying it to an Arduino device.
On ML Tools website, a model can easily be trained. For settings, follow to screenshots in `/edge_impuls`. To add a custom model, 
with the structure of the nano_33_ble deployement enable expart mode and paste `edge_impuls/model.py`
When done you will receive a folder with Syntiant firmware. To Deploy it on the Nicla Voice follow the next steps.

## Prerequisites

* **Arduino CLI v0.34.2**
  Install from the [GitHub release page](https://github.com/arduino/arduino-cli/releases) and add it to your PATH variable.

* **Python serial library**

```bash
pip install pyserial
```

## Arduino Firmware

1. Using the tools from /stop_go_project install the NDP120 firmware (Created by training a model on Edge Impulse)

2. Format the flash of the NDP120:

```
format_windows_ext_flash.bat
```

3. Install the firmware using the flash script for your OS, e.g.:

```
flash_windows.bat
```

4. After flashing, the NDP120 should contain the following files:

* `ei_model.synpkg`
* `mcu_fw_120_v91.synpkg`
* `dsp_firmware_v91.synpkg`

## Demo

1. Open the Arduino IDE.
2. Open serial monitor for the right port, set baud to 115200 and Both NL + CR
3. If everything worked correct, you can now interact with the NPD. Send "AT+HELP" to view list of commands

* The model should now recognize the keywords **stop**, **go** and occasionally **noise**

## Data

The data used to train the model, was copied from the `nano_33_ble` project. Therefore the accuracy of the nicla voice deployement is bad, as its not trained on data from its on mic. Sadly it seems impossible to access the nicla voice mic, as its blocked behind the NDP120 chip.