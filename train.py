import numpy as np, os, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_CLASSES = 8
MODEL_PATH  = "model/blood_group_cnn.h5"

def build_model():
    m = Sequential([
        # Block 1 — simple
        Conv2D(32,(3,3), activation="relu", padding="same", input_shape=(128,128,1)),
        MaxPooling2D(2,2),
        Dropout(0.3),

        # Block 2
        Conv2D(64,(3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.3),

        # Block 3
        Conv2D(64,(3,3), activation="relu", padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    m.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)

    # Load data
    X_train = np.load("processed/X_train.npy")
    X_val   = np.load("processed/X_val.npy")
    y_train = to_categorical(np.load("processed/y_train.npy"), NUM_CLASSES)
    y_val   = to_categorical(np.load("processed/y_val.npy"),   NUM_CLASSES)

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # Data augmentation — creates more variety from same images
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train with augmented data
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=callbacks
    )

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    ax[0].plot(history.history["accuracy"],     label="Train")
    ax[0].plot(history.history["val_accuracy"], label="Val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(history.history["loss"],     label="Train")
    ax[1].plot(history.history["val_loss"], label="Val")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("model/training_history.png")
    plt.show()
    print(f"Done! Model saved to {MODEL_PATH}")