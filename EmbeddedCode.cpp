#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Arduino.h>
#include "SparkFun_Bio_Sensor_Hub_Library.h"
#include "Wire.h"

const int RESET_PIN   = 17;
const int MFIO_PIN    = 18;
const int I2C_ADDRESS = 0x55;

// ─── Timing constants ─────────────────────────────────────────────────────────
const unsigned long MEASURE_DURATION_MS  = 30UL * 1000UL;        // 30 seconds
const unsigned long MEASURE_INTERVAL_MS  = 15UL * 60UL * 1000UL; // 15 minutes

// ─── Device state machine ─────────────────────────────────────────────────────
typedef enum {
    STATE_IDLE,
    STATE_MEASURING,
} DeviceState;

volatile DeviceState deviceState = STATE_IDLE;

// Tracks when the last measurement session started.
// Initialize to a value that guarantees a reading fires immediately on boot.
unsigned long lastMeasureTime = ULONG_MAX - MEASURE_INTERVAL_MS;

SparkFun_Bio_Sensor_Hub bioHub(RESET_PIN, MFIO_PIN, I2C_ADDRESS);

// ─── BLE UUIDs ────────────────────────────────────────────────────────────────
#define SERVICE_UUID             "12345678-1234-1234-1234-123456789012"
#define CHARACTERISTIC_UUID      "87654321-4321-4321-4321-210987654321"
#define COMMAND_CHARACTERISTIC_UUID "11223344-5566-7788-9900-aabbccddeeff"

BLECharacteristic *pCharacteristic;
BLECharacteristic *pCommandCharacteristic;
bool deviceConnected = false;

// ─── BLE Server Callbacks ─────────────────────────────────────────────────────
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("Device connected!");
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("Device disconnected!");
    delay(500);
    BLEDevice::startAdvertising();
  }
};

// ─── BLE Command Callbacks ────────────────────────────────────────────────────
// Handles write commands from the web app (e.g. "START" for manual reading)
class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) override {
    std::string value = pCharacteristic->getValue();

    if (value == "START") {
      if (deviceState == STATE_IDLE) {
        Serial.println("[CMD] Manual START received — beginning session.");
        deviceState = STATE_MEASURING;
      } else {
        Serial.println("[CMD] START ignored — already measuring.");
      }
    }
  }
};

void setup() {
  Serial.begin(115200);
  Serial.println("PPG BLE Logger");

  // ─── BLE setup ───────────────────────────────────────────────────────────
  BLEDevice::init("QT_Py_ESP32S3");
  BLEDevice::setMTU(512);

  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Data characteristic — notifies web app with IR and Red LED values
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  pCharacteristic->setValue("0,0");

  // Command characteristic — receives write commands from the web app
  pCommandCharacteristic = pService->createCharacteristic(
    COMMAND_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCommandCharacteristic->setCallbacks(new CommandCallbacks());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();
  Serial.println("BLE advertising...");

  // ─── Sensor setup ────────────────────────────────────────────────────────
  Wire1.begin();
  Wire1.setClock(400000);

  int result = bioHub.begin(Wire1, RESET_PIN, MFIO_PIN);
  Serial.println(result == 0 ? "Sensor initialized!" : "Sensor init failed!");

  bioHub.configSensor();
  bioHub.setSampleRate(100);
  bioHub.setPulseWidth(69);
}

// ─── Read one sample and notify over BLE ──────────────────────────────────────
void ReadAndSendBLEData()
{
  bioData body = bioHub.readSensor();

  if (deviceConnected) {
    String msg = String(body.irLed) + "," + String(body.redLed);
    pCharacteristic->setValue(msg.c_str());
    pCharacteristic->notify();

    Serial.print("IR: ");
    Serial.print(body.irLed);
    Serial.print(" | Red: ");
    Serial.println(body.redLed);
  }
}

// ─── Blocking 30-second read + BLE send session ───────────────────────────────
void ReadAndSend30Sec()
{
  Serial.println("[MEASURING] Starting 30-second PPG session...");
  unsigned long sessionStart = millis();

  while (millis() - sessionStart < MEASURE_DURATION_MS) {
    ReadAndSendBLEData();

    // Early exit if client disconnects mid-session
    if (!deviceConnected) {
      Serial.println("[MEASURING] Client disconnected — ending session early.");
      break;
    }

    delay(10); // ~100 Hz, matches setSampleRate(100)
  }

  Serial.println("[MEASURING] Session complete.");
}

// ─── Main loop ────────────────────────────────────────────────────────────────
void loop() {

  if (deviceState == STATE_IDLE) {
    unsigned long now     = millis();
    unsigned long elapsed = now - lastMeasureTime;
    unsigned long remaining = MEASURE_INTERVAL_MS - elapsed;

    // Log countdown every 60 seconds so you can monitor over Serial
    static unsigned long lastLog = 0;
    if (now - lastLog >= 60000UL) {
      Serial.printf("[IDLE] Next auto-reading in %lu min %lu sec\n",
                    remaining / 60000UL, (remaining % 60000UL) / 1000UL);
      lastLog = now;
    }

    // Auto-trigger every 15 minutes (also fires immediately on first boot)
    if (elapsed >= MEASURE_INTERVAL_MS) {
      Serial.println("[IDLE] Auto-trigger fired.");
      deviceState = STATE_MEASURING;
    }
  }

  if (deviceState == STATE_MEASURING) {
    // Stamp before the session so the 15-min interval is wall-clock accurate
    // regardless of whether this was triggered manually or automatically
    lastMeasureTime = millis();
    ReadAndSend30Sec();
    deviceState = STATE_IDLE;
  }

  delay(100);
}
