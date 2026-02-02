import math, requests
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D, MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

sys.path.append('./resources/libraries')
import ei_tensorflow.training

# --- 1. DYNAMIC SHAPE CALCULATION ---
# Edge Impulse passes data as a flat array (1D). 
# We need to figure out the 2D shape (Height x Width) automatically.

input_data_size = X_train[0].shape[0]
print(f"Total input features per sample: {input_data_size}")

# OPTION A: If you know your data is Square (e.g., Spectrogram 40x40)
# width = int(math.sqrt(input_data_size))
# height = width

# OPTION B: Standard Audio (MFCC/MFE)
# Usually, Edge Impulse calculates: (Window size / Stride) x (Number of Filters)
# Example: 650 features might be 50 time steps x 13 filters.
# You must adjust these two numbers to match your DSP Block settings in Edge Impulse!
# If you are unsure, try printing the shape or checking the 'Create Impulse' page.

# LET'S TRY TO AUTO-DETECT SQUARE FIRST, ELSE FALLBACK TO COMMON CONFIG
if int(math.sqrt(input_data_size)) ** 2 == input_data_size:
    width = int(math.sqrt(input_data_size))
    height = width
    print(f"Auto-detected square input: {width}x{height}")
else:
    # If not square, we guess standard MFCC (13 filters)
    # CHANGE THIS IF YOU USE MFE (often 40 filters)
    n_filters = 13 
    if input_data_size % n_filters == 0:
        width = int(input_data_size / n_filters)
        height = n_filters
        print(f"Auto-detected Rectangular input (assuming {n_filters} filters): {width}x{height}")
    else:
        # Fallback for MFE default
        n_filters = 40
        width = int(input_data_size / n_filters)
        height = n_filters
        print(f"Auto-detected Rectangular input (assuming {n_filters} filters): {width}x{height}")

INPUT_SHAPE = (width, height, 1)

# --- 2. BUILD MODEL ---
model = Sequential()

# Reshape the flat input buffer to 2D image format
model.add(Reshape(INPUT_SHAPE, input_shape=(input_data_size,)))

# Conv Block 1
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

# Conv Block 2
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

# --- CLASSIFIER CHANGE ---
# Replaced Flatten() with GlobalAveragePooling2D()
# Why? Flatten() is rigid and crashes if dimensions don't match perfectly.
# GlobalAveragePooling2D() handles any input size, is mathematically cleaner, 
# and saves massive amounts of RAM on the Arduino.
model.add(GlobalAveragePooling2D()) 

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

# Output Layer
model.add(Dense(classes, activation='softmax'))

# --- 3. TRAINING CONFIG ---
EPOCHS = args.epochs or 30
LEARNING_RATE = args.learning_rate or 0.005
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*4)

prefetch_policy = 1 if ENSURE_DETERMINISM else tf.data.AUTOTUNE
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(prefetch_policy)

callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

# --- 4. COMPILE & FIT ---
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_dataset, 
          validation_data=validation_dataset, 
          epochs=EPOCHS, 
          verbose=2, 
          callbacks=callbacks, 
          class_weight=ei_tensorflow.training.get_class_weights(Y_train))