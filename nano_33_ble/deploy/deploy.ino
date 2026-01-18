/*
 * TinyML Audio Classifier (Keyword Spotting)
 * Production Version
 * Fixes included:
 * 1. Log Scaling (4.34x) to match Librosa dB scale
 * 2. Shadow Buffer to prevent input tensor corruption by TFLite
 */

#include <PDM.h>
#include <TensorFlowLite.h>
#include <arduinoFFT.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "dsp_params.h"
#include "mel_filterbank.h"
#include "model_config.h"

// ==========================================
// CONFIG
// ==========================================
// Software gain to match recorder.py (which uses GAIN = 1.5)
#define GAIN_FACTOR 1.5f

// ==========================================
// GLOBALS
// ==========================================

// Audio Input Buffers
// PDM usually provides ~256 samples per callback
short sampleBuffer[512];
volatile int samplesRead = 0; // Counter for the ISR

// Processing Buffers
float audioWindow[N_FFT];
float vReal[N_FFT];
float vImag[N_FFT];
int samplesSinceInference = 0; // Accumulator

// SHADOW BUFFER (CRITICAL FIX):
// Maintained separately from TFLite arena to prevent data corruption
int8_t spectrogram[EXPECTED_FRAMES][N_MFCC_COEFFS];

// DSP Objects
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, N_FFT, SAMPLE_RATE);

// TFLite Objects
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;

// Memory Arena
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

// ==========================================
// ISR (Interrupt Service Routine)
// ==========================================
void onPDMdata() {
  // Query bytes available
  int bytesAvailable = PDM.available();

  // Read into the buffer
  PDM.read(sampleBuffer, bytesAvailable);

  // Update count (16-bit samples = bytes / 2)
  samplesRead = bytesAvailable / 2;
}

// ==========================================
// DSP HELPERS
// ==========================================

void compute_features(float *input_audio, int8_t *output_features) {
  // 1. Copy to FFT buffers
  for (int i = 0; i < N_FFT; i++) {
    vReal[i] = input_audio[i];
    vImag[i] = 0.0f;
  }

  // 2. FFT with Hamming window
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // CRITICAL: librosa uses POWER spectrum (magnitude^2), not magnitude!
  for (int i = 0; i < N_FFT_BINS; i++) {
    vReal[i] = vReal[i] * vReal[i];
  }

  // 3. Apply Mel Filterbank
  float mel_energies[N_MEL_FILTERS];

  for (int m = 0; m < N_MEL_FILTERS; m++) {
    float sum = 0.0f;
    for (int k = 0; k < N_FFT_BINS; k++) {
      sum += vReal[k] * mel_filterbank[m][k];
    }

    // FIX MATCHING PYTHON:
    // Librosa uses 10 * log10(x) = 10 * ln(x) / ln(10)
    // Scaling factor 10 / ln(10) ~= 4.3429
    mel_energies[m] = 4.342944819f * logf(sum + 1e-6f);
  }

  // 4. DCT-II to get MFCCs
  float mfcc[N_MFCC_COEFFS];

  for (int k = 0; k < N_MFCC_COEFFS; k++) {
    float sum = 0.0f;
    for (int m = 0; m < N_MEL_FILTERS; m++) {
      sum += dct_matrix[k][m] * mel_energies[m];
    }
    mfcc[k] = sum;
  }

  // 5. Quantize to int8 for TFLite
  float scale = tflInputTensor->params.scale;
  int zero_point = tflInputTensor->params.zero_point;

  for (int i = 0; i < N_MFCC_COEFFS; i++) {
    int32_t val = (int32_t)(mfcc[i] / scale) + zero_point;
    if (val > 127)
      val = 127;
    if (val < -128)
      val = -128;
    output_features[i] = (int8_t)val;
  }
}

// ==========================================
// MAIN SETUP
// ==========================================
void setup() {
  Serial.begin(115200);

  // Wait for serial but timeout after 3s
  long start = millis();
  while (!Serial && (millis() - start < 3000))
    ;

  Serial.println("--- Starting Audio Classifier ---");

  // Initialize spectrogram to safe zero point
  memset(spectrogram, 0, sizeof(spectrogram));

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  // LED Check (Blue ON)
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);

  // 1. TFLite Init
  tflModel = tflite::GetModel(audio_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Model schema mismatch!");
    while (1) {
      digitalWrite(LEDR, !digitalRead(LEDR));
      delay(100);
    }
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver,
                                                tensorArena, kTensorArenaSize);

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: AllocateTensors failed!");
    while (1) {
      digitalWrite(LEDR, LOW);
      delay(500);
    }
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // Init spectrogram with actual zero point from model
  memset(spectrogram, tflInputTensor->params.zero_point, sizeof(spectrogram));

  Serial.println("TFLite Init OK");

  // 2. Microphone Init
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Error: PDM Start Failed!");
    while (1)
      ;
  }

  // Match the gain used during recording
  PDM.setGain(80);

  Serial.println("Microphone OK");
  Serial.println("--- Setup Complete ---");

  // LED Off
  digitalWrite(LEDB, HIGH);
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  // 1. Atomic Read of Interrupt Data
  int localSamplesRead = 0;
  short localBuffer[512];

  if (samplesRead > 0) {
    noInterrupts();
    localSamplesRead = samplesRead;
    for (int i = 0; i < localSamplesRead; i++) {
      localBuffer[i] = sampleBuffer[i];
    }
    samplesRead = 0;
    interrupts();
  }

  // 2. Process Audio if we got any
  if (localSamplesRead > 0) {

    // Slide the Audio Window
    memmove(audioWindow, &audioWindow[localSamplesRead],
            (N_FFT - localSamplesRead) * sizeof(float));

    // Add new data
    for (int i = 0; i < localSamplesRead; i++) {
      // Apply Gain & Normalize
      float sample = (float)localBuffer[i] * GAIN_FACTOR;
      if (sample > 32767.0f)
        sample = 32767.0f;
      if (sample < -32768.0f)
        sample = -32768.0f;

      int idx = (N_FFT - localSamplesRead) + i;
      audioWindow[idx] = sample / 32768.0f;
    } // Accumulate how much new data we have processed
    samplesSinceInference += localSamplesRead;

    // 3. Check if we have enough new data to run a HOP
    if (samplesSinceInference >= HOP_LENGTH) {

      // DEBUG: Simple VU Meter to make sure we are getting signal
      float max_amp = 0.0f;
      for (int i = 0; i < N_FFT; i++) {
        if (fabs(audioWindow[i]) > max_amp)
          max_amp = fabs(audioWindow[i]);
      }
      Serial.print("Vol:");
      Serial.print(max_amp, 2);
      Serial.print(" [");
      int bars = (int)(max_amp * 20.0f); // Scale to ~20 chars
      for (int i = 0; i < bars; i++)
        Serial.print("#");
      for (int i = bars; i < 20; i++)
        Serial.print(" ");
      Serial.print("]  ");

      // Blink Green briefly
      digitalWrite(LEDG, LOW);

      // A. Extract Features
      int8_t mfcc_col[N_MFCC_COEFFS];
      compute_features(audioWindow, mfcc_col);

      // B. Update SHADOW Spectrogram (FIXED)
      int time_steps = EXPECTED_FRAMES;

      // Shift logic
      for (int t = 0; t < time_steps - 1; t++) {
        for (int c = 0; c < N_MFCC_COEFFS; c++) {
          spectrogram[t][c] = spectrogram[t + 1][c];
        }
      }

      // Add new col at the end
      for (int c = 0; c < N_MFCC_COEFFS; c++) {
        spectrogram[time_steps - 1][c] = mfcc_col[c];
      }

      // C. Copy to TFLite Input Tensor (FIXED: Full Copy every time)
      int8_t *input_data = tflInputTensor->data.int8;
      memcpy(input_data, spectrogram, sizeof(spectrogram));

      // D. Run Inference
      TfLiteStatus invoke_status = tflInterpreter->Invoke();
      if (invoke_status == kTfLiteOk) {
        float scale = tflOutputTensor->params.scale;
        int zero = tflOutputTensor->params.zero_point;

        Serial.print("Probs: ");
        float max_conf = 0.0;
        int max_idx = -1;

        for (int i = 0; i < NUM_CLASSES; i++) {
          float prob = (tflOutputTensor->data.int8[i] - zero) * scale;
          Serial.print(class_labels[i]);
          Serial.print(":");
          Serial.print(prob, 2);
          Serial.print("  ");

          if (prob > max_conf) {
            max_conf = prob;
            max_idx = i;
          }
        }

        if (max_conf > CONFIDENCE_THRESHOLD && max_idx > 0) {
          Serial.print("  >>> DETECTED: ");
          Serial.print(class_labels[max_idx]);
        }
        Serial.println();
      }

      samplesSinceInference -= HOP_LENGTH;
      digitalWrite(LEDG, HIGH); // LED Off
    }
  }
}