from tensorflow.keras import layers, models

# model architecture is inspired from: https://www.tensorflow.org/tutorials/audio/simple_audio,
# but heavily compacted to run on embedded devices

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(X_train, y_train, X_test, y_test, epochs, batch_size):
    input_shape = X_train.shape[1:] 
    num_classes = len(set(y_train))

    model = build_model(input_shape, num_classes)

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test))

    return model
