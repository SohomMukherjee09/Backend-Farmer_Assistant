import tensorflow as tf
import pickle
import numpy as np

# Step 1: Load the .keras model
model = tf.keras.models.load_model('Models/plant_disease_recog_model_pwp (1).keras')

# Step 2: Extract model architecture and weights
model_json = model.to_json()  # Serialize model architecture to JSON
model_weights = model.get_weights()  # Get model weights as a list of NumPy arrays

# Step 3: Create a dictionary to store model components
model_dict = {
    'architecture': model_json,
    'weights': model_weights
}

# Step 4: Save the dictionary to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

# Step 5 (Optional): Load the pickled model back
with open('model.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# Reconstruct the model
loaded_model = tf.keras.models.model_from_json(loaded_dict['architecture'])
loaded_model.set_weights(loaded_dict['weights'])

# Compile the model (required before training or inference)
loaded_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Verify the model (e.g., by checking its summary)
loaded_model.summary()