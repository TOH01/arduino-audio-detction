/*
 * TinyML Audio Classifier (Keyword Spotting)
 * FIXED: Buffer Logic & Debugging
 */

#include <PDM.h>
#include <arduinoFFT.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "dsp_params.h"
#include "model_config.h"

// ==========================================
// DEBUG & GAIN
// ==========================================
// Bitshift gain to boost volume (Mic is quiet!)
// 4 = multiply by 16. If still quiet, try 5 or 6.
#define GAIN_SHIFT 4 

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

// DSP Objects
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, N_FFT, SAMPLE_RATE);

// TFLite Objects
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

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
  // Note: We are assuming we process fast enough that we don't overwrite
  // unread data. In a robust app, use a RingBuffer.
  PDM.read(sampleBuffer, bytesAvailable);
  
  // Update count (16-bit samples = bytes / 2)
  samplesRead = bytesAvailable / 2;
}

// ==========================================
// DSP HELPERS
// ==========================================
void compute_dct(float* log_energies, float* dct_features) {
    for (int k = 0; k < N_MFCC; k++) {
        float sum = 0.0;
        for (int n = 0; n < N_MFCC; n++) {
            sum += log_energies[n] * cos(PI / N_MFCC * (n + 0.5) * k);
        }
        dct_features[k] = sum;
    }
}

void compute_features(float* input_audio, int8_t* output_features) {
    // 1. Copy to FFT buffers
    for (int i = 0; i < N_FFT; i++) {
        vReal[i] = input_audio[i];
        vImag[i] = 0.0f;
    }

    // 2. FFT
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();

    // 3. Mel Binning
    int bin_size = (N_FFT / 2) / N_MFCC;
    float mel_energies[N_MFCC];

    for (int i = 0; i < N_MFCC; i++) {
        float sum = 0.0f;
        for (int j = 0; j < bin_size; j++) {
            int idx = (i * bin_size) + j + 1;
            if (idx < N_FFT/2) sum += vReal[idx];
        }
        mel_energies[i] = log10(sum + 1e-6);
    }

    // 4. DCT
    float dct_energies[N_MFCC];
    compute_dct(mel_energies, dct_energies);

    // 5. Quantize
    float scale = tflInputTensor->params.scale;
    int zero_point = tflInputTensor->params.zero_point;
    
    for (int i = 0; i < N_MFCC; i++) {
        int32_t val = (dct_energies[i] / scale) + zero_point;
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        output_features[i] = (int8_t)val;
    }
}

// ==========================================
// MAIN SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  
  // Wait a bit for serial to connect, but don't hang forever
  // if you want to run it on battery later.
  long start = millis();
  while(!Serial && (millis() - start < 3000));

  Serial.println("--- Starting Setup ---");

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
    while(1) { digitalWrite(LEDR, !digitalRead(LEDR)); delay(100); } // Fast blink Red
  }

  tflInterpreter = new tflite::MicroInterpreter(
      tflModel, tflOpsResolver, tensorArena, kTensorArenaSize);

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: AllocateTensors failed!");
    while(1) { digitalWrite(LEDR, LOW); delay(500); } // Solid Red
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  
  Serial.println("TFLite Init OK");

  // 2. Microphone Init
  PDM.onReceive(onPDMdata);
  // Note: 1 channel, 16kHz
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Error: PDM Start Failed!");
    while(1);
  }
  
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
  // We need to grab the data safely so the ISR doesn't overwrite it while we copy
  int localSamplesRead = 0;
  short localBuffer[512]; // Temp buffer to copy into

  if (samplesRead > 0) {
    noInterrupts();
    localSamplesRead = samplesRead;
    // Copy only what was read
    for(int i=0; i<localSamplesRead; i++) {
      localBuffer[i] = sampleBuffer[i];
    }
    samplesRead = 0; // Reset ISR counter
    interrupts();
  }

  // 2. Process Audio if we got any
  if (localSamplesRead > 0) {
    
    // Slide the Audio Window back by 'localSamplesRead' amount
    // (Throw away oldest data)
    memmove(audioWindow, &audioWindow[localSamplesRead], (N_FFT - localSamplesRead) * sizeof(float));

    // Add new data to the end of Audio Window
    for (int i = 0; i < localSamplesRead; i++) {
        // Apply Gain & Normalize
        int32_t sample = localBuffer[i] << GAIN_SHIFT;
        // Clamp 16-bit
        if (sample > 32767) sample = 32767;
        if (sample < -32768) sample = -32768;
        
        int idx = (N_FFT - localSamplesRead) + i;
        audioWindow[idx] = (float)sample / 32768.0f;
    }

    // Accumulate how much new data we have processed
    samplesSinceInference += localSamplesRead;

    // 3. Check if we have enough new data to run a HOP
    if (samplesSinceInference >= HOP_LENGTH) {
        
        // Blink Green briefly to show "Brain is working"
        digitalWrite(LEDG, LOW);
        
        // A. Extract Features
        int8_t mfcc_col[N_MFCC];
        compute_features(audioWindow, mfcc_col);

        // B. Update TFLite Input (Rolling Spectrogram)
        int8_t* input_data = tflInputTensor->data.int8;
        int time_steps = tflInputTensor->dims->data[1];
        int n_mfccs = tflInputTensor->dims->data[2];

        // Shift existing image left
        int bytes_to_shift = (time_steps - 1) * n_mfccs;
        memmove(input_data, input_data + n_mfccs, bytes_to_shift);
        
        // Add new col at the end
        for (int k=0; k<n_mfccs; k++) input_data[bytes_to_shift + k] = mfcc_col[k];

        // C. Run Inference
        TfLiteStatus invoke_status = tflInterpreter->Invoke();
        if (invoke_status == kTfLiteOk) {
            
            // D. Get Results
            float max_conf = 0.0;
            int max_idx = -1;
            
            float scale = tflOutputTensor->params.scale;
            int zero = tflOutputTensor->params.zero_point;

            // Debug Print
            Serial.print("Conf: ");

            for (int i = 0; i < NUM_CLASSES; i++) {
                float prob = (tflOutputTensor->data.int8[i] - zero) * scale;
                Serial.print(prob); Serial.print(" ");
                if (prob > max_conf) {
                    max_conf = prob;
                    max_idx = i;
                }
            }
            Serial.println(); // Newline

            // Action based on result
            if (max_conf > CONFIDENCE_THRESHOLD && max_idx > 0) {
               Serial.print(">>> DETECTED CLASS: ");
               Serial.println(max_idx);
            }
        }
        
        // Reset accumulation logic
        // We subtract HOP_LENGTH to keep phase, rather than zeroing out
        samplesSinceInference -= HOP_LENGTH;
        
        digitalWrite(LEDG, HIGH); // LED Off
    }
  }
}