import os
import shutil
import random
import json
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define paths to the dataset
base_dir = 'D:/Users/270249563/Downloads/PlantDataset'  # Base directory containing all images
train_dir = os.path.join(base_dir, 'train')  # Directory for training data
valid_dir = os.path.join(base_dir, 'valid')  # Directory for validation data

# Function to split dataset into training and validation sets
def split_dataset(base_dir, train_dir, valid_dir, split_size=0.8):
    classes = os.listdir(base_dir)  # List all classes
    for cls in classes:
        class_path = os.path.join(base_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle images to randomize dataset
        
        train_size = int(len(images) * split_size)  # Calculate the number of training images
        train_images = images[:train_size]  # Select training images
        valid_images = images[train_size:]  # Select validation images
        
        # Copy training images to the training directory
        for img in train_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(train_dir, cls, img)
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            if not os.path.exists(dest):  # Avoid overwriting existing files
                shutil.copy(src, dest)
        
        # Copy validation images to the validation directory
        for img in valid_images:
            src = os.path.join(class_path, img)
            dest = os.path.join(valid_dir, cls, img)
            os.makedirs(os.path.join(valid_dir, cls), exist_ok=True)
            if not os.path.exists(dest):  # Avoid overwriting existing files
                shutil.copy(src, dest)

# Split the dataset into training and validation sets
split_dataset(base_dir, train_dir, valid_dir, split_size=0.8)

# Image data generator for preprocessing with augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # Rescale pixel values from [0,255] to [0,1]
    shear_range=0.2,           # Apply shear transformations
    zoom_range=0.2,            # Apply random zoom
    horizontal_flip=True,      # Randomly flip images horizontally
    rotation_range=40,         # Apply random rotations
    width_shift_range=0.2,     # Randomly shift images horizontally
    height_shift_range=0.2     # Randomly shift images vertically
)

# Image data generator for preprocessing validation set (without augmentation)
valid_datagen = ImageDataGenerator(rescale=1.0/255)  # Only rescale for validation data

# Prepare training dataset generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),   # Resize images to 128x128 pixels
    batch_size=32,            # Number of images to be yielded from the generator per batch
    class_mode='categorical'  # Set class mode to categorical for multi-class classification
)

# Prepare validation dataset generator
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),   # Resize images to 128x128 pixels
    batch_size=32,            # Number of images to be yielded from the generator per batch
    class_mode='categorical'  # Set class mode to categorical for multi-class classification
)

# Save class indices to a JSON file for later use (useful for inference)
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Get the number of classes for the final output layer
num_classes = len(train_generator.class_indices)

# Build the Convolutional Neural Network (CNN) model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # First convolutional layer
    BatchNormalization(),  # Normalize the activations of the previous layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling to reduce spatial dimensions
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    BatchNormalization(),  # Normalize the activations of the previous layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling to reduce spatial dimensions
    Conv2D(128, (3, 3), activation='relu'),  # Third convolutional layer
    BatchNormalization(),  # Normalize the activations of the previous layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling to reduce spatial dimensions
    Flatten(),  # Flatten the 3D output to 1D for the fully connected layers
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dropout(0.5),  # Dropout for regularization to prevent overfitting
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation for multi-class classification
])

# Compile the model with the Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop training when validation loss stops improving
checkpoint = ModelCheckpoint('best_plant_disease_detector.keras', monitor='val_accuracy', save_best_only=True)  # Save only the best model

# Train the model with the training and validation data generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of batches per epoch
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size,  # Number of batches per validation step
    epochs=20,  # Number of epochs to train
    callbacks=[early_stopping, checkpoint]  # Use the early stopping and model checkpoint callbacks
)

# Save the final trained model to a file
model.save('plant_disease_detector.keras')

# Load the trained model
model = load_model('best_plant_disease_detector.keras')

# Save the training history (loss and accuracy over epochs) to a JSON file
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
