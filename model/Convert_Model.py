import numpy as np
import tensorflow as tf

# --- Load the saved Keras model ---
model = tf.keras.models.load_model("sbd_model.h5")
print("Keras model loaded successfully.")

# --- Create a representative dataset for quantization ---
# The representative dataset must match the new data shape.
X = np.load(r"C:\VSD_Sqd_Project\SBD_VER_2.3\model\data_gen\X_bal_strict.npy")

def representative_dataset_gen():
    for i in range(100): # Using 100 samples is usually sufficient
        # Yield a single example with the correct data type (float32 is fine for conversion).
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

print("TFLite model saved as sbd_model.tflite")

# --- Convert the TFLite model to a C++ header file ---
# This part is from your original script and remains the same.
# You will use this to generate the sbd_model.h file.