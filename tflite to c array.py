import os

# --- Configuration ---
TFLITE_FILE = "sbd_model.tflite"
HEADER_FILE = "sbd_model.h"
ARRAY_NAME = "sbd_model_tflite"

# --- Read the TFLite file ---
try:
    with open(TFLITE_FILE, "rb") as f:
        tflite_content = f.read()
except FileNotFoundError:
    print(f"Error: Could not find '{TFLITE_FILE}'. Make sure it's in the same directory.")
    exit()

# --- Write the C header file ---
with open(HEADER_FILE, "w") as f:
    f.write(f"// Converted from {TFLITE_FILE} using a Python script\n\n")
    f.write(f"#ifndef SBD_MODEL_H\n")
    f.write(f"#define SBD_MODEL_H\n\n")
    f.write(f"const unsigned char {ARRAY_NAME}[] = {{\n  ")

    # Write the byte array
    line_len = 0
    for byte in tflite_content:
        f.write(f"0x{byte:02x}, ")
        line_len += 1
        if line_len >= 16:
            f.write("\n  ")
            line_len = 0

    f.write("\n};\n\n")

    # Write the array length
    f.write(f"const unsigned int {ARRAY_NAME}_len = {len(tflite_content)};\n\n")
    f.write(f"#endif // SBD_MODEL_H\n")

print(f"Successfully converted '{TFLITE_FILE}' to '{HEADER_FILE}'")
print(f"C array name: {ARRAY_NAME}")
print(f"Array length: {len(tflite_content)}")