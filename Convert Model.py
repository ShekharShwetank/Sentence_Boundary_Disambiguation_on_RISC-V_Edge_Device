import numpy as np
import tensorflow as tf

# --- Load the saved Keras model ---
model = tf.keras.models.load_model("sbd_model.h5")
print("Keras model loaded successfully.")

# --- Create a representative dataset for quantization ---
# Quantization requires a small sample of your training data to determine the range of weights and activations.
X = np.load(r"C:\Users\shwet\OneDrive\Desktop\NLP_PROJECT\X_bal_strict.npy")
def representative_dataset_gen():
    for i in range(100): # Using 100 samples is usually sufficient
        yield [X[i:i+1].astype(np.float32)]

# --- Convert the model to TFLite with full integer quantization ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# Enforce integer-only quantization for microcontroller compatibility
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# --- Save the TFLite model to a file ---
with open("sbd_model.tflite", "wb") as f:
    f.write(tflite_model)

print("\nModel converted to TFLite format and saved as sbd_model.tflite")
print(f"TFLite model size: {len(tflite_model)} bytes")