#include <PDM.h>
#include <TensorFlowLite.h>
#include <arduinoFFT.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "dsp_params.h"
#include "mel_filterbank.h"
#include "model_config.h"

#ifdef INJECT_TEST_AUDIO
#include "audio_inject.h"
#endif

// Value from recorder.py (GAIN = 1.5)
#define GAIN_FACTOR 1.5f

#ifdef INJECT_TEST_AUDIO
size_t inject_idx = 0;
#endif

short sampleBuffer[512];
volatile int samplesRead = 0;

float audioWindow[N_FFT];
float vReal[N_FFT];
float vImag[N_FFT];
int samplesSinceInference = 0;

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
uint8_t tensorArena[kTensorArenaSize];

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);

  // Update count (16-bit samples = bytes / 2)
  samplesRead = bytesAvailable / 2;
}

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

  // librosa uses magnitude^2
  // https://librosa.org/doc/latest/generated/librosa.power_to_db.html
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

    // Librosa uses 10 * log10(x) = 10 * ln(x) / ln(10)
    // Scaling factor 10 / ln(10) ~= 4.3429
    // https://librosa.org/doc/latest/generated/librosa.power_to_db.html
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

void setup() {
  Serial.begin(115200);

  long start = millis();
  while (!Serial && (millis() - start < 3000))
    ;

  Serial.println("--- Start Init Audio Classifier ---");

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

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

  memset(spectrogram, tflInputTensor->params.zero_point, sizeof(spectrogram));
#ifndef INJECT_TEST_AUDIO
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Error: PDM Start Failed!");
    while (1)
      ;
  }

  // Must match value from recorder.ino
  PDM.setGain(80);
#endif

  Serial.println("--- Init Complete ---");
}

void loop() {
  int localSamplesRead = 0;
  short localBuffer[512];

#ifdef INJECT_TEST_AUDIO
  if (samplesRead == 0) {
    if (inject_idx >= test_audio_len)
      inject_idx = 0;
    int chunk = 512;
    int remaining = test_audio_len - inject_idx;
    int to_copy = (remaining < chunk) ? remaining : chunk;
    memcpy(sampleBuffer, &test_audio_data[inject_idx], to_copy * sizeof(short));
    inject_idx += to_copy;
    if (to_copy < chunk) {
      memcpy(&sampleBuffer[to_copy], &test_audio_data[0],
             (chunk - to_copy) * sizeof(short));
      inject_idx = chunk - to_copy;
    }
    samplesRead = chunk;
    delay(32);
  }
#endif

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

      int time_steps = EXPECTED_FRAMES;

      for (int t = 0; t < time_steps - 1; t++) {
        for (int c = 0; c < N_MFCC_COEFFS; c++) {
          spectrogram[t][c] = spectrogram[t + 1][c];
        }
      }

      for (int c = 0; c < N_MFCC_COEFFS; c++) {
        spectrogram[time_steps - 1][c] = mfcc_col[c];
      }

      int8_t *input_data = tflInputTensor->data.int8;
      memcpy(input_data, spectrogram, sizeof(spectrogram));

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