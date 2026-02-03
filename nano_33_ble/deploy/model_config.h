#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "audio_model.h"

#define CONFIDENCE_THRESHOLD 0.60
#define NUM_CLASSES 3
#define MODEL_NAME "audio_model"

static const char* class_labels[NUM_CLASSES] = {
  "noise",
  "open",
  "close",
};

#endif // MODEL_CONFIG_H
