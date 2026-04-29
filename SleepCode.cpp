#include <Arduino.h>
#include "esp_sleep.h"

// Change this if you want periodic wakeups.
// 24 hours is long, but still wake-capable.
static constexpr uint64_t SLEEP_US = 24ULL * 60ULL * 60ULL * 1000000ULL;

static void minimize_gpio_leakage() {
  // Put any pins used by your old circuit into high-impedance input mode.
  // Keep only the pins your board really needs.
  pinMode(17, INPUT);
  pinMode(18, INPUT);

  // Stop Serial so USB/UART activity does not keep anything awake.
  Serial.end();
}

void setup() {
  minimize_gpio_leakage();

  // No BLE, no sensor init, no loop logic.
  // Go to deep sleep immediately.
  esp_sleep_enable_timer_wakeup(SLEEP_US);
  esp_deep_sleep_start();
}

void loop() {
  // Never reached.
}
