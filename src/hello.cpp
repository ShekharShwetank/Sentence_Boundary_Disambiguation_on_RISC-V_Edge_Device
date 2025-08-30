#include <stdint.h>
#include <string.h>

// TFLM C++ Headers
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h" // For supported operators

// Your model's C header
#include "sbd_model.h"

// --- Peripheral Definitions ---
#define GPIO_BASE      0x10012000UL
#define GPIO_IOF_EN    (*(volatile uint32_t *)(GPIO_BASE + 0x38))
#define GPIO_IOF_SEL   (*(volatile uint32_t *)(GPIO_BASE + 0x3C))

#define UART0_BASE     0x10013000UL
#define UART_TXDATA    (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART_RXDATA    (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART_TXCTRL    (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_RXCTRL    (*(volatile uint32_t *)(UART0_BASE + 0x0C))
#define UART_DIV       (*(volatile uint32_t *)(UART0_BASE + 0x18))

// --- Application Constants ---
constexpr int kWindowLen = 21;
constexpr int kVocabSize = 97;
char g_input_buffer[kWindowLen];

extern "C" void *__dso_handle;
void *__dso_handle;

// --- UART Functions ---
void uart_init_115200(void) {
    GPIO_IOF_SEL &= ~((1u << 16) | (1u << 17));
    GPIO_IOF_EN  |=  ((1u << 16) | (1u << 17));
    UART_DIV = 138; // Baud rate for ~115200 with 16MHz clock
    UART_TXCTRL = 1; // TX Enable
    UART_RXCTRL = 1; // RX Enable
}

void uart_write_char(char c) {
    while (UART_TXDATA & (1u << 31));
    UART_TXDATA = c;
}

char uart_read_char(void) {
    uint32_t rxdata;
    do {
        rxdata = UART_RXDATA;
    } while (rxdata & (1u << 31)); // Loop until the RX FIFO is not empty
    return (char)(rxdata & 0xFF);
}

void uart_write_str(const char* s) {
    while (*s) {
        if (*s == '\n') uart_write_char('\r');
        uart_write_char(*s++);
    }
}

class UartErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    // Just dump the message literally without formatting
    uart_write_str(format);
    uart_write_str("\r\n");
    return 0;
  }
};

// --- TFLM Globals ---
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create a tensor arena to hold the model's tensors.
constexpr int kTensorArenaSize = 40 * 1024; // Increased size for safety
uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// This function performs the one-hot encoding directly on the microcontroller.
void preprocess_input(const char* raw_input, int8_t* model_input) {
    // Clear the input buffer to all zeros for one-hot encoding
    memset(model_input, 0, kWindowLen * kVocabSize * sizeof(int8_t));

    // A correct character-to-ID mapping
    auto char_to_id = [](char c) -> int {
        if (c >= 'a' && c <= 'z') return (c - 'a' + 1);
        if (c >= 'A' && c <= 'Z') return (c - 'A' + 1); // Add support for uppercase
        if (c >= '0' && c <= '9') return (c - '0' + 27);
        // The punctuation mapping must match your Python script
        switch (c) {
            case ' ': return 37; case '!': return 38; case '"': return 39; case '#': return 40;
            case '$': return 41; case '%': return 42; case '&': return 43; case '\'': return 44;
            case '(': return 45; case ')': return 46; case '*': return 47; case '+': return 48;
            case ',': return 49; case '-': return 50; case '.': return 51; case '/': return 52;
            case ':': return 53; case ';': return 54; case '<': return 55; case '=': return 56;
            case '>': return 57; case '?': return 58; case '@': return 59; case '[': return 60;
            case '\\': return 61; case ']': return 62; case '^': return 63; case '_': return 64;
            case '`': return 65; case '{': return 66; case '|': return 67; case '}': return 68;
            case '~': return 69;
            default: return 0; // Unknown characters map to <PAD>
        }
    };

    // Populate the flattened one-hot encoded vector
    // The input tensor is a flattened array of size kWindowLen * kVocabSize
    for (int i = 0; i < kWindowLen; ++i) {
        int char_id = char_to_id(raw_input[i]);
        if (char_id < kVocabSize) {
            // Apply one-hot encoding
            model_input[i * kVocabSize + char_id] = 1;
        }
    }
}

// --- TFLM Setup ---
void setup_tflm() {
	static UartErrorReporter uart_error_reporter;
	error_reporter = &uart_error_reporter;
    model = tflite::GetModel(sbd_model_tflite);

    // This is the key change: a mutable op resolver with supported ops
    static tflite::MicroMutableOpResolver<3> resolver(error_reporter);

    if (resolver.AddFullyConnected() != kTfLiteOk) {
        error_reporter->Report("Failed to add FULLY_CONNECTED op.");
        return;
    }
    if (resolver.AddReshape() != kTfLiteOk) {
        error_reporter->Report("Failed to add RESHAPE op."); // Used for the Flatten layer
        return;
    }
    if (resolver.AddLogistic() != kTfLiteOk) {
        error_reporter->Report("Failed to add LOGISTIC op."); // Used for the Sigmoid activation
        return;
    }

    // Build an interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, nullptr, nullptr);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed.");
        return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
}

// --- Main Program ---
int main(void) {
    uart_init_115200();
    uart_write_str("SBD Model Ready debug 1.\r\n");
    setup_tflm();
    uart_write_str("SBD Model Ready.\r\n");

    for (;;) {
        // 1. Read a full window of characters from UART
        for (int i = 0; i < kWindowLen; ++i) {
            g_input_buffer[i] = uart_read_char();
        }

        // 2. Pre-process the raw input buffer and load it into the model's input tensor
        preprocess_input(g_input_buffer, input->data.int8);

        // 3. Run inference on the model
        if (interpreter->Invoke() != kTfLiteOk) {
            error_reporter->Report("Invoke failed");
            uart_write_str("E\r\n"); // Send 'E' for error and a newline
            continue;
        }

        // 4. Post-process the output and send the result
        int8_t result_quantized = output->data.int8[0];
        int8_t threshold_quantized = output->params.zero_point;

        if (result_quantized > threshold_quantized) {
            uart_write_char('1'); // End of Sentence
        } else {
            uart_write_char('0'); // Not End of Sentence
        }
    }
    return 0;
}
