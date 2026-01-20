import serial
import numpy as np
import wave
import argparse
import serial.tools.list_ports
import os
import datetime
import threading
import time
import msvcrt


BAUD = 115200
SAMPLE_RATE = 16000
DURATION = 1.0
GAIN = 1.5


def get_key():
    if msvcrt.kbhit():
        return msvcrt.getch().decode('utf-8', errors='ignore')


class Recorder:
    def __init__(self, port, out_dir, label):
        self.ser = serial.Serial(port, BAUD, timeout=0.1)
        self.out_dir = out_dir
        self.label = label
        self.running = True
        self.recording = False
        self.buffer = []
        self.samples_needed = int(SAMPLE_RATE * DURATION)
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)

    def read_loop(self):
        while self.running:
            data = self.ser.read(1024)
            if data:
                samples = np.frombuffer(data, dtype=np.int16).copy()
                samples = (samples * GAIN).clip(-32768, 32767).astype(np.int16)
                if self.recording:
                    self.buffer.extend(samples)
                    if len(self.buffer) >= self.samples_needed:
                        self.save()
                        self.recording = False

    def save(self):
        samples = np.array(self.buffer[:self.samples_needed], dtype=np.int16)
        ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        path = os.path.join(self.out_dir, f"{self.label}.{ts}.wav")
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())
        print(f"Saved: {path}")
        self.buffer = []

    def run(self):
        threading.Thread(target=self.read_loop, daemon=True).start()
        print(f"Press SPACE to record '{self.label}', Q to quit")
        try:
            while self.running:
                key = get_key()
                if key == ' ' and not self.recording:
                    self.count += 1
                    print(f"Recording {self.count}...")
                    # prevent noise from pressing keycap appearing in recording
                    time.sleep(0.15)
                    self.ser.reset_input_buffer()
                    self.buffer = []
                    self.recording = True
                elif key in ('q', 'Q'):
                    self.running = False
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        self.ser.close()
        print(f"Done. Recorded {self.count} samples.")


if __name__ == "__main__":
    print("Ports:", [p.device for p in serial.tools.list_ports.comports()])
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True)
    parser.add_argument('-d', '--directory', default="recordings")
    parser.add_argument('-l', '--label', default="audio")
    args = parser.parse_args()
    Recorder(args.port, args.directory, args.label).run()
