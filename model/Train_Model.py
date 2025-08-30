import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split

# --- Constants from your dataset generation script ---
WINDOW_RADIUS = 10
WINDOW_LEN = 2 * WINDOW_RADIUS + 1
VOCAB_SIZE = 97
EMBEDDING_DIM = 8

# --- Load Balanced Dataset ---
X = np.load(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\X_bal_strict.npy")
y = np.load(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\y_bal_strict.npy")

print(f"Loaded dataset shapes: X={X.shape}, y={y.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define the Model ---
model = Sequential([
    # This is the new input layer. It takes the one-hot encoded tensor and flattens it.
    Flatten(input_shape=(WINDOW_LEN, VOCAB_SIZE)),
    # A fully connected layer with ReLU activation.
    Dense(16, activation='relu'),
    # Output layer with a single neuron and sigmoid activation for binary classification.
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,  # You can adjust this
    validation_data=(X_val, y_val),
    verbose=2
)

# --- Save the trained model ---
model.save("sbd_model.h5")
print("\nModel trained and saved to sbd_model.h5")