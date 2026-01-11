import serial
import numpy as np
import wave
import struct
import argparse
import serial.tools.list_ports
import os
import datetime
import uuid

# Default values
DEFAULT_BAUD = 115200
DEFAULT_LABEL = "audio"

# Parse arguments
parser = argparse.ArgumentParser(description="Serial Audio Data Collection")
parser.add_argument('-p', '--port', dest='port', type=str, required=True, help="Serial port to connect to")
parser.add_argument('-b', '--baud', dest='baud', type=int, default=DEFAULT_BAUD, help="Baud rate")
parser.add_argument('-d', '--directory', dest='directory', type=str, default=".", help="Output directory for files")
parser.add_argument('-l', '--label', dest='label', type=str, default=DEFAULT_LABEL, help="Label for files")
args = parser.parse_args()

port = args.port
baud = args.baud
out_dir = args.directory
label = args.label

# Print available serial ports
print("\nAvailable serial ports:")
for port_info, desc, hwid in sorted(serial.tools.list_ports.comports()):
    print(f" {port_info} : {desc} [{hwid}]")

print(f"\nConnecting to {port} at {baud} baud...")
ser = serial.Serial(port, baud)
if not ser.is_open:
    ser.open()

# Make output directory
os.makedirs(out_dir, exist_ok=True)

# Audio configuration
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
SAMPLE_RATE = 16000
CHUNK_SIZE = 16000
RECORD_DURATION = 2  # seconds

# Noise gate and amplification
NOISE_GATE_THRESHOLD = 300
GAIN = 2.0  # amplify factor

def amplify(data, gain):
    return data * gain

def main():
    buffer = []

    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            # Read audio chunk from serial
            data = ser.read(CHUNK_SIZE * SAMPLE_WIDTH)
            if not data:
                continue

            samples = np.frombuffer(data, dtype=np.int16).copy()

            # Apply simple noise gate
            samples[np.abs(samples) < NOISE_GATE_THRESHOLD] = 0

            # Amplify
            samples = amplify(samples, GAIN)

            buffer.extend(samples.astype(np.int16))

            # Save to WAV if enough samples collected
            if len(buffer) >= SAMPLE_RATE * RECORD_DURATION * CHANNELS:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                uid = str(uuid.uuid4())[-12:]
                filename = f"{label}.{uid}.{timestamp}.wav"
                filepath = os.path.join(out_dir, filename)

                buffer_array = np.array(buffer, dtype=np.int16)
                with wave.open(filepath, "w") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(SAMPLE_WIDTH)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(buffer_array.tobytes())

                print(f"Saved: {filepath}")
                buffer = []

    except KeyboardInterrupt:
        print("\nStopped recording.")
        ser.close()

if __name__ == "__main__":
    main()
