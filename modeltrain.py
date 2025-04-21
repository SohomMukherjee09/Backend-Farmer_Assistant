import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load and preprocess the test image for alignment (optional, for debugging)
test_image_path = 'path_to_your_apple_image.jpg'  # Replace with the actual path
test_image = load_img(test_image_path, target_size=(160, 160))
test_image_array = img_to_array(test_image) / 255.0
print("Test image shape:", test_image_array.shape)

# Load pre-trained ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base layers initially

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Increased units for better capacity
x = Dropout(0.5)(x)
predictions = Dense(39, activation='softmax')(x)

# Create the model
new_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Data generator with targeted augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.6],  # Adjusted for rust colors
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,  # Align with ResNet
    fill_mode='nearest'
)

# Load and prepare data generators
train_generator = datagen.flow_from_directory(
    'dataset/plant_disease_images',
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    'dataset/plant_disease_images',
    target_size=(160, 160),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Calculate and adjust class weights
class_weights = dict()
total_samples = train_generator.samples
class_indices = train_generator.class_indices
target_class = 'Apple___Cedar_apple_rust'
target_index = class_indices[target_class]
class_counts = np.bincount(train_generator.classes)
min_count = min(class_counts[class_counts > 0])
for i, count in enumerate(class_counts):
    weight = max(1.0, 8.0 * min_count / count if count > 0 else 1.0)  # Stronger weight up to 8x
    class_weights[i] = weight
print("Adjusted class weights:", class_weights)

# Calculate steps
train_samples = train_generator.samples
val_samples = validation_generator.samples
steps_per_epoch = train_samples // train_generator.batch_size
validation_steps = val_samples // validation_generator.batch_size

print(f"Training samples: {train_samples}, Validation samples: {val_samples}")
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
print("Training classes:", train_generator.class_indices)
print(f"Number of classes: {len(train_generator.class_indices)}")

# Check class distribution
class_counts = np.bincount(train_generator.classes)
print("Class distribution (training):", dict(zip(train_generator.class_indices.keys(), class_counts)))

# Initial training with frozen layers (reduced to 5 epochs)
history = new_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=15,
    class_weight=class_weights
)

# Unfreeze more layers for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Fine-tune the model (reduced to 5 epochs)
history_fine = new_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=10,  # Total 10 epochs (5 initial + 5 fine-tuning)
    initial_epoch=5,
    class_weight=class_weights
)

# Save the model
new_model.save("Models/new_plant_disease_recog_model_pwp.keras")
print("Transfer learning model with reduced epochs saved as 'new_plant_disease_model_transfer_learning_reduced_epochs.keras'.")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()