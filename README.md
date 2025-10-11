# arduino-audio-detction

Arduino Nicla Vision Setup

under board manager install arduino mbed os nicla boards

flash firmware from zip file:

first flash example/NDP/syntiant_upload_fw

then upload binaries:

./syntiant-uploader send -m "Y" -w "Y" -p $portName $filename

example for windows, using COM3 and provided zip file
./syntiant-uploader-win send -m "Y" -w "Y" -p COM3 mcu_fw_120_v91.synpkg
./syntiant-uploader-win send -m "Y" -w "Y" -p COM3 dsp_firmware_v91.synpkg
./syntiant-uploader-win send -m "Y" -w "Y" -p COM3 alexa_334_NDP120_B0_v11_v91.synpkg

To Record data, use sketch examples/NDP/Record_and_stream.ino

Will need to manually install 2 libraries

into arduino library folder
git clone https://github.com/pschatzmann/arduino-libg722
git clone https://github.com/pschatzmann/arduino-audio-tools/

for the audio tools library, version 0.9.6 is required.

in arduino-audio-tools library : git checkout tags/v0.9.6

now can start the record_and_stream.ino sketch
