#include <stdint.h>

#define GPIO_BASE      0x10012000UL
#define GPIO_INPUT_EN  (*(volatile uint32_t *)(GPIO_BASE + 0x04))
#define GPIO_OUTPUT_EN (*(volatile uint32_t *)(GPIO_BASE + 0x08))
#define GPIO_IOF_EN    (*(volatile uint32_t *)(GPIO_BASE + 0x38))
#define GPIO_IOF_SEL   (*(volatile uint32_t *)(GPIO_BASE + 0x3C))

#define UART0_BASE     0x10013000UL
#define UART_TXDATA    (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART_RXDATA    (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART_TXCTRL    (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_RXCTRL    (*(volatile uint32_t *)(UART0_BASE + 0x0C))
#define UART_IE        (*(volatile uint32_t *)(UART0_BASE + 0x10))
#define UART_IP        (*(volatile uint32_t *)(UART0_BASE + 0x14))
#define UART_DIV       (*(volatile uint32_t *)(UART0_BASE + 0x18))

static inline void uart_init_115200_from_hfrosc_14p4mhz(void) {
    // GPIO16=RX, GPIO17=TX => select IOF0 function and enable it
    const uint32_t RX_BIT = (1u << 16);
    const uint32_t TX_BIT = (1u << 17);

    // Route pins to IOF0 (UART0 uses IOF0 on 16/17)
    GPIO_IOF_SEL &= ~(RX_BIT | TX_BIT); // IOF0
    GPIO_IOF_EN  |=  (RX_BIT | TX_BIT);

    // Ensure GPIO input enabled for RX, output enabled for TX not required when IOF drives pins,
    // but harmless if set:
    GPIO_INPUT_EN  |= RX_BIT;
    GPIO_OUTPUT_EN |= TX_BIT;

    // Baud: ~14.4 MHz / (124+1) approx 115.2 kbaud
    UART_DIV   = 138u;

    // 8N1, tx enable, rx enable, watermark 0
    UART_TXCTRL = (1u << 0); // txen
    UART_RXCTRL = (1u << 0); // rxen

    // No interrupts for polling
    UART_IE = 0;
}

static inline void uart_write_char(char c) {
    // Wait while TX FIFO full (bit31==1)
    while (UART_TXDATA & (1u << 31)) { }
    UART_TXDATA = (uint8_t)c;
}

static inline int uart_read_char_nonblock(void) {
    uint32_t v = UART_RXDATA;
    if (v & (1u << 31)) return -1; // empty
    return (int)(v & 0xFF);
}

static void uart_write_str(const char *s) {
    while (*s) {
        if (*s == '\n') uart_write_char('\r');
        uart_write_char(*s++);
    }
}

int main(void) {
    uart_init_115200_from_hfrosc_14p4mhz();
    uart_write_str("UART0 echo ready\r\n");

    for (;;) {
        int ch = uart_read_char_nonblock();
        if (ch >= 0) {
            // simple echo
            uart_write_char((char)ch);
        }
    }
    return 0;
}
