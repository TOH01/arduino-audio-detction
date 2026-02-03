import math
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

sys.path.append('./resources/libraries')
import ei_tensorflow.training

input_data_size = X_train[0].shape[0]
print(f"Total input features per sample: {input_data_size}")

if int(math.sqrt(input_data_size)) ** 2 == input_data_size:
    width = int(math.sqrt(input_data_size))
    height = width
    print(f"Auto-detected square input: {width}x{height}")
else:
    n_filters = 13 
    if input_data_size % n_filters == 0:
        width = int(input_data_size / n_filters)
        height = n_filters
        print(f"Auto-detected Rectangular input (assuming {n_filters} filters): {width}x{height}")
    else:
        n_filters = 40
        width = int(input_data_size / n_filters)
        height = n_filters
        print(f"Auto-detected Rectangular input (assuming {n_filters} filters): {width}x{height}")

INPUT_SHAPE = (width, height, 1)

# as close as possible to nano_33_ble/src/model.py
model = Sequential()

model.add(Reshape(INPUT_SHAPE, input_shape=(input_data_size,)))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(GlobalAveragePooling2D()) 

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(classes, activation='softmax'))

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

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_dataset, 
          validation_data=validation_dataset, 
          epochs=EPOCHS, 
          verbose=2, 
          callbacks=callbacks, 
          class_weight=ei_tensorflow.training.get_class_weights(Y_train))