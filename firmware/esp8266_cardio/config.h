// config.h -- edit these for your network and laptop before flashing.

// Wi-Fi (2.4 GHz only: the ESP8266 cannot join 5 GHz networks)
#define WIFI_SSID      "YOUR_WIFI_NAME"
#define WIFI_PASSWORD  "YOUR_WIFI_PASSWORD"

// Laptop running uvicorn. Find it with `ipconfig` (Windows) or `ip addr` (Linux).
// Give the laptop a fixed IP in your router, or the device will stop finding it
// when the address changes.
#define SERVER_HOST    "192.168.1.50"
#define SERVER_PORT    8000

// Must match the CARDIO_DEVICE_KEY environment variable on the laptop.
#define DEVICE_KEY     "choose-a-long-secret"

// Shown on the dashboard. Keep the same patient ID you use in the form.
#define DEVICE_ID      "esp8266-01"
#define PATIENT_ID     "patient_001"
