/* Includes */
#include <PDM.h>
#include <TensorFlowLite.h>
#include <arduinoFFT.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <limits.h>

#include "dsp_params.h"
#include "mel_filterbank.h"
#include "model_config.h"

#ifdef INJECT_TEST_AUDIO
#include "audio_inject.h"
#endif

/* Defines */
#define GAIN_FACTOR 1.5f  // Value from recorder.py (GAIN = 1.5)
#define PDM_GAIN 80       // Must match value from recorder.ino
#define TENSOR_ARENA_SIZE 60 * 1024
#define AUDIO_BLOCK_SIZE 512

#ifdef INJECT_TEST_AUDIO
#define INJECT_TEST_AUDIO_DELAY 32
#endif

/* Globals */
#ifdef INJECT_TEST_AUDIO
size_t Deploy_InjectIdx = 0;
#endif

short Deploy_SampleBuffer[AUDIO_BLOCK_SIZE];
volatile int Deploy_SamplesRead = 0;

float Deploy_AudioWindow[N_FFT];
float Deploy_vReal[N_FFT];
float Deploy_vImag[N_FFT];
int Deploy_SamplesSinceInference = 0;

int8_t Deploy_Spectogram[EXPECTED_FRAMES][N_MFCC_COEFFS];

ArduinoFFT<float> Deploy_FFT = ArduinoFFT<float>(Deploy_vReal, Deploy_vImag, N_FFT, SAMPLE_RATE);

tflite::AllOpsResolver Deploy_TflOpsResolve;
const tflite::Model *Deploy_TflModel = nullptr;
tflite::MicroInterpreter *Deploy_TflInterpreter = nullptr;
TfLiteTensor *Deploy_TflInputTensor = nullptr;
TfLiteTensor *Deploy_TflOutputTensor = nullptr;

uint8_t Deploy_TensorArena[TENSOR_ARENA_SIZE];

void onPDMdata(void) {
  int bytesAvailable = PDM.available();
  PDM.read(Deploy_SampleBuffer, bytesAvailable);
  Deploy_SamplesRead = bytesAvailable / 2;
}

void compute_features(float *input_audio, int8_t *output_features) {
  float melEnergies[N_MEL_FILTERS];
  float scale = Deploy_TflInputTensor->params.scale;
  int zeroPoint = Deploy_TflInputTensor->params.zero_point;
  float mfcc[N_MFCC_COEFFS];
  
  for (int i = 0; i < N_FFT; i++) {
    Deploy_vReal[i] = input_audio[i];
    Deploy_vImag[i] = 0.0f;
  }

  Deploy_FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  Deploy_FFT.compute(FFTDirection::Forward);
  Deploy_FFT.complexToMagnitude();

  // librosa uses magnitude^2
  // https://librosa.org/doc/latest/generated/librosa.power_to_db.html
  for (int i = 0; i < N_FFT_BINS; i++) {
    Deploy_vReal[i] = Deploy_vReal[i] * Deploy_vReal[i];
  }
  for (int m = 0; m < N_MEL_FILTERS; m++) {
    float sum = 0.0f;
    for (int k = 0; k < N_FFT_BINS; k++) {
      sum += Deploy_vReal[k] * mel_filterbank[m][k];
    }
    // Librosa uses 10 * log10(x) = 10 * ln(x) / ln(10)
    // Scaling factor 10 / ln(10) ~= 4.3429
    // https://librosa.org/doc/latest/generated/librosa.power_to_db.html
    melEnergies[m] = 4.342944819f * logf(sum + 1e-6f);
  }
  for (int k = 0; k < N_MFCC_COEFFS; k++) {
    float sum = 0.0f;
    for (int m = 0; m < N_MEL_FILTERS; m++) {
      sum += dct_matrix[k][m] * melEnergies[m];
    }
    mfcc[k] = sum;
  }
  for (int i = 0; i < N_MFCC_COEFFS; i++) {
    int32_t val = (int32_t)(mfcc[i] / scale) + zeroPoint;
    if (val > INT8_MAX)
      val = INT8_MAX;
    if (val < INT8_MIN)
      val = INT8_MIN;
    output_features[i] = (int8_t)val;
  }
}

void CriticalError_Handler(void){
  while(1){
    // do nothing
  }
}

void setup() {
  Serial.begin(115200);

  while (!Serial) {
    // do nothing
  };

  Deploy_TflModel = tflite::GetModel(audio_model);

  if (Deploy_TflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Model schema mismatch!");
    CriticalError_Handler();
  }

  Deploy_TflInterpreter = new tflite::MicroInterpreter(Deploy_TflModel, Deploy_TflOpsResolve, Deploy_TensorArena, TENSOR_ARENA_SIZE);

  if (Deploy_TflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: AllocateTensors failed!");
    CriticalError_Handler();
  }

  Deploy_TflInputTensor = Deploy_TflInterpreter->input(0);
  Deploy_TflOutputTensor = Deploy_TflInterpreter->output(0);

  memset(Deploy_Spectogram, Deploy_TflInputTensor->params.zero_point, sizeof(Deploy_Spectogram));

  #ifndef INJECT_TEST_AUDIO
    PDM.onReceive(onPDMdata);
    if (!PDM.begin(1, SAMPLE_RATE)) {
      CriticalError_Handler();
    }

    PDM.setGain(PDM_GAIN);
  #endif
}

void loop() {
  int localSamplesRead = 0;
  short localBuffer[AUDIO_BLOCK_SIZE];
  int8_t mfccCol[N_MFCC_COEFFS];
  float maxConf;
  int maxIdx;
  float prob;

  __WFI();

  #ifdef INJECT_TEST_AUDIO
    int remaining;
    int toCopy;

    if (Deploy_SamplesRead == 0) {
      if (Deploy_InjectIdx >= test_audio_len){
        Deploy_InjectIdx = 0;
      }
        
      remaining = test_audio_len - Deploy_InjectIdx;
      if (remaining < AUDIO_BLOCK_SIZE){
        toCopy = remaining;
      }
      else{
        toCopy = AUDIO_BLOCK_SIZE;
      }

      memcpy(Deploy_SampleBuffer, &test_audio_data[Deploy_InjectIdx], toCopy * sizeof(short));
      Deploy_InjectIdx += toCopy;

      if (toCopy < AUDIO_BLOCK_SIZE) {
        memcpy(&Deploy_SampleBuffer[toCopy], &test_audio_data[0], (AUDIO_BLOCK_SIZE - toCopy) * sizeof(short));
        Deploy_InjectIdx = AUDIO_BLOCK_SIZE - toCopy;
      }
      Deploy_SamplesRead = AUDIO_BLOCK_SIZE;
      delay(INJECT_TEST_AUDIO_DELAY);
    }
  #endif

  if (Deploy_SamplesRead > 0) {
    noInterrupts();
    localSamplesRead = Deploy_SamplesRead;
    for (int i = 0; i < localSamplesRead; i++) {
      localBuffer[i] = Deploy_SampleBuffer[i];
    }
    Deploy_SamplesRead = 0;
    interrupts();
  }

  if (localSamplesRead > 0) {

    memmove(Deploy_AudioWindow, &Deploy_AudioWindow[localSamplesRead], (N_FFT - localSamplesRead) * sizeof(float));

    for (int i = 0; i < localSamplesRead; i++) {
      float sample = (float)localBuffer[i] * GAIN_FACTOR;
      if (sample > SHRT_MAX){
        sample = SHRT_MAX;
      }
      else if(sample < SHRT_MIN){
        sample = SHRT_MIN;
      } 

      Deploy_AudioWindow[(N_FFT - localSamplesRead) + i] = sample / (float)(SHRT_MAX + 1);
    }
    
    Deploy_SamplesSinceInference += localSamplesRead;

    if (Deploy_SamplesSinceInference >= HOP_LENGTH) {
      compute_features(Deploy_AudioWindow, mfccCol);

      for (int t = 0; t < EXPECTED_FRAMES - 1; t++) {
        for (int c = 0; c < N_MFCC_COEFFS; c++) {
          Deploy_Spectogram[t][c] = Deploy_Spectogram[t + 1][c];
        }
      }

      for (int c = 0; c < N_MFCC_COEFFS; c++) {
        Deploy_Spectogram[EXPECTED_FRAMES - 1][c] = mfccCol[c];
      }

      memcpy(Deploy_TflInputTensor->data.int8, Deploy_Spectogram, sizeof(Deploy_Spectogram));

      TfLiteStatus invoke_status = Deploy_TflInterpreter->Invoke();
      if (invoke_status == kTfLiteOk) {
        maxConf = 0.0;
        maxIdx = -1;

        Serial.print("Probabilites: ");

        for (int i = 0; i < NUM_CLASSES; i++) {
          prob = (Deploy_TflOutputTensor->data.int8[i] - Deploy_TflOutputTensor->params.zero_point) * Deploy_TflOutputTensor->params.scale;
          Serial.print(class_labels[i]);
          Serial.print(":");
          Serial.print(prob, 2);
          Serial.print("  ");

          if (prob > maxConf) {
            maxConf = prob;
            maxIdx = i;
          }
        }

        if (maxConf > CONFIDENCE_THRESHOLD && maxIdx > 0) {
          Serial.print("  >>> DETECTED: ");
          Serial.print(class_labels[maxIdx]);
        }
        
        Serial.println();
      }

      Deploy_SamplesSinceInference -= HOP_LENGTH;
    }
  }
}