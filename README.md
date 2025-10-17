# arduino-audio-detction

train model with : https://mltools.arduino.cc

How to deploy ei model

https://github.com/arduino/arduino-cli (v 0.34.2)

first install arduino-cli v 0.34.2 from gh release page, add to path variable
pip install pyserial

get arduino firmware from here : 
https://docs.edgeimpulse.com/hardware/boards/arduino-nicla-voice

to install it, using the flash script for desired os, e.g flash_windows.bat
now the ndp120 should be populated with

ei_model.synpkg
mcu_fw_120_v91.synpkg
dsp_firmware_v91.synpkg

then get
https://github.com/edgeimpulse/firmware-arduino-nicla-voice
follow the instruction from README.MD

need to both build and flash

then open arduino ide, demo model with go and stop keyword recognition should now run
