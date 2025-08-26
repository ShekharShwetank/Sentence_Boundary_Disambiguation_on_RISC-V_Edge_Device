import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split

# --- Constants from your dataset generation script ---
WINDOW_RADIUS = 10
WINDOW_LEN = 2 * WINDOW_RADIUS + 1
# Vocabulary size: alphabet size + PAD token + 1 (since char2id starts from 1)
VOCAB_SIZE = 95 + 1 + 1 # Based on string.ascii_lowercase + string.digits + string.punctuation + " "
EMBEDDING_DIM = 8

# --- Load Balanced Dataset ---
# Using the strictly balanced dataset for better training on a simple model
X = np.load(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X_bal_strict.npy")
y = np.load(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\y_bal_strict.npy")

print(f"Loaded dataset shapes: X={X.shape}, y={y.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define the Model ---
model = Sequential([
    # The Embedding layer converts integer indices (character IDs) into dense vectors of fixed size.
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=WINDOW_LEN),
    # Flattens the 3D tensor from Embedding into a 2D tensor for the Dense layers.
    Flatten(),
    # A fully connected layer with ReLU activation to learn non-linear patterns.
    Dense(16, activation='relu'),
    # Output layer with a single neuron and sigmoid activation for binary classification (0 or 1).
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train the Model ---
print("\nStarting model training...")
history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    batch_size=128)

# --- Save the Trained Model ---
model.save("sbd_model.h5")
print("\nModel training complete. Saved to sbd_model.h5")