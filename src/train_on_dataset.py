import os

import tensorflow as tf
from keras.src.applications.efficientnet import EfficientNetB0
from tensorflow import keras
from tensorflow.keras import layers

## TODO: THIS SHOULD PROBABLY BE CONVERTED TO A NOTEBOOK - BETTER FOR VISUALIZATION ##

os.chdir("..")
current_working_directory = os.getcwd()
print(current_working_directory)

# Use most common image size for EfficientNetB0
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32  # Number of samples processed before the model is updated
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
# Number of pavement classes
NUM_CLASSES = 6
TRAIN_DIR = 'dataset/training_data'
VALID_DIR = 'dataset/validation_data'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    # Generate Labels from subdirectory names
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    crop_to_aspect_ratio=True,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMAGE_SIZE,
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),  # Rotate by up to 10% of 2*pi
        layers.RandomZoom(0.1),  # Zoom by up to 10%
        layers.RandomContrast(0.1),  # Adjust contrast by up to 10%
    ], name="data_augmentation"
)

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)  # 3 for RGB images
)

# Freeze the base model (so its weights don't change during initial training)
base_model.trainable = False

# Create the new model on top
inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

x = data_augmentation(inputs)  # Apply augmentation
# TODO: Check if preprocessing is needed for EfficientNetB0, probably not needed here
x = tf.keras.applications.efficientnet.preprocess_input(x)  # Preprocessing specific to EfficientNet

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)  # To convert features to a flat vector
x = layers.Dense(128, activation='relu')(x)  # Example intermediate dense layer
x = layers.Dropout(0.5)(x)  # Dropout for regularization
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax for multi-class

model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',  # For multi-class classification with one-hot labels
    metrics=['accuracy']
)

## TRAINING
EPOCHS = 20

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    'best_pavement_model.keras', save_best_only=True, monitor='val_accuracy'
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint]
)
