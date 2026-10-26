/*
  CardioPredictor+ sensor node (NodeMCU ESP8266)

  Reads heart rate and an approximate SpO2 from a MAX30102, shows them on a
  0.96" SSD1306 OLED, and POSTs a reading to the laptop every 5 seconds:

      POST http://<laptop>:8000/device/reading
      X-Device-Key: <DEVICE_KEY>
      {"device_id":..., "patient_id":..., "heart_rate":72, "spo2":97, "signal_ok":true}

  When there is no finger (or no clear pulse yet) it sends signal_ok=false and
  null values, so the dashboard shows "No finger" instead of a wrong number.

  Libraries (Arduino IDE > Library Manager):
    - SparkFun MAX3010x Pulse and Proximity Sensor Library
    - Adafruit SSD1306 and Adafruit GFX
    - ArduinoJson (version 6.x)
  Board: "NodeMCU 1.0 (ESP-12E Module)", upload speed 115200.

  SpO2 note: this is a rough ratio-of-ratios estimate, not a clinical oximeter.
  The MAX30102 module is hobby-grade. The model does not use SpO2.
*/

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"   // SparkFun library; also works with the MAX30102 chip
#include "heartRate.h"  // checkForBeat()
#include "config.h"

Adafruit_SSD1306 display(128, 64, &Wire, -1);
MAX30105 sensor;

// ---- timing ----
const unsigned long POST_EVERY_MS = 5000;
const unsigned long WIFI_RETRY_MS = 10000;
unsigned long lastPost = 0;
unsigned long lastWifiTry = 0;
int lastHttpCode = 0;

// ---- finger / beat detection ----
const long FINGER_IR_MIN = 30000;   // IR below this = no finger on the sensor
const byte RATE_SIZE = 4;           // average of the last 4 beat intervals
int rates[RATE_SIZE];
byte rateCount = 0;
byte rateSpot = 0;
long lastBeatMs = 0;
int beatAvg = 0;                    // 0 = no valid heart rate yet

// ---- approximate SpO2 (ratio of ratios) ----
double dcRed = 0, dcIr = 0, acRed = 0, acIr = 0;
double spo2Smooth = -1;
int spo2Out = 0;                    // 0 = no valid SpO2 yet

void resetVitals() {
  rateCount = 0;
  rateSpot = 0;
  beatAvg = 0;
  spo2Out = 0;
  spo2Smooth = -1;
  dcRed = dcIr = acRed = acIr = 0;
}

void processSample(long ir, long red) {
  bool fingerOn = ir > FINGER_IR_MIN;
  if (!fingerOn) {
    resetVitals();
    return;
  }

  if (checkForBeat(ir)) {
    long now = millis();
    long intervalMs = now - lastBeatMs;
    lastBeatMs = now;
    if (intervalMs > 0) {
      int bpm = (int)(60000L / intervalMs);
      if (bpm >= 30 && bpm <= 220) {
        rates[rateSpot] = bpm;
        rateSpot = (rateSpot + 1) % RATE_SIZE;
        if (rateCount < RATE_SIZE) rateCount++;
        long sum = 0;
        for (byte i = 0; i < rateCount; i++) sum += rates[i];
        beatAvg = (int)(sum / rateCount);
      }
    }
  }

  // Slow DC level and running AC amplitude for red and IR.
  dcRed += (red - dcRed) * 0.01;
  dcIr  += (ir  - dcIr)  * 0.01;
  acRed += (fabs(red - dcRed) - acRed) * 0.05;
  acIr  += (fabs(ir  - dcIr)  - acIr)  * 0.05;

  if (rateCount >= 2 && dcRed > 0 && dcIr > 0 && acIr > 0) {
    double R = (acRed / dcRed) / (acIr / dcIr);
    double est = 110.0 - 25.0 * R;
    if (spo2Smooth < 0) spo2Smooth = est;
    else spo2Smooth = 0.9 * spo2Smooth + 0.1 * est;
    if (spo2Smooth > 100) spo2Smooth = 100;
    if (spo2Smooth < 70) spo2Smooth = 70;
    spo2Out = (int)(spo2Smooth + 0.5);
  }
}

void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println(WiFi.status() == WL_CONNECTED ? " connected" : " not yet");
}

void postReading(bool signalOk, int hr, int spo2) {
  if (WiFi.status() != WL_CONNECTED) { lastHttpCode = -1; return; }

  char url[80];
  snprintf(url, sizeof(url), "http://%s:%d/device/reading", SERVER_HOST, SERVER_PORT);

  StaticJsonDocument<256> doc;
  doc["device_id"] = DEVICE_ID;
  doc["patient_id"] = PATIENT_ID;
  doc["signal_ok"] = signalOk;
  if (signalOk && hr > 0) doc["heart_rate"] = hr; else doc["heart_rate"] = nullptr;
  if (signalOk && spo2 >= 70) doc["spo2"] = spo2; else doc["spo2"] = nullptr;

  char body[256];
  serializeJson(doc, body, sizeof(body));

  WiFiClient client;
  HTTPClient http;
  http.setTimeout(3000);
  http.begin(client, url);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Device-Key", DEVICE_KEY);
  lastHttpCode = http.POST(body);
  http.end();
  Serial.printf("POST %s -> %d\n", body, lastHttpCode);
}

void drawScreen(bool signalOk) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("CardioPredictor+");

  display.setTextSize(2);
  display.setCursor(0, 14);
  if (signalOk && beatAvg > 0) {
    display.print("HR ");
    display.print(beatAvg);
  } else {
    display.print("HR --");
  }

  display.setCursor(0, 34);
  if (signalOk && spo2Out > 0) {
    display.print("O2 ");
    display.print(spo2Out);
    display.print("%");
  } else {
    display.print("O2 --");
  }

  display.setTextSize(1);
  display.setCursor(0, 56);
  if (!signalOk) {
    display.print("Place finger");
  } else {
    display.print(WiFi.status() == WL_CONNECTED ? "WiFi OK " : "WiFi down ");
    display.print("HTTP ");
    display.print(lastHttpCode);
  }
  display.display();
}

void setup() {
  Serial.begin(115200);
  Wire.begin(D2, D1);  // SDA = D2, SCL = D1

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found at 0x3C");
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Starting...");
  display.display();

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found at 0x57. Check wiring.");
    display.clearDisplay();
    display.setCursor(0, 0);
    display.print("Sensor not found");
    display.display();
    while (true) delay(1000);
  }
  // LED brightness 60, average 4 samples, red+IR, 100 Hz, pulse width 411, ADC range 4096.
  sensor.setup(60, 4, 2, 100, 411, 4096);

  connectWifi();
  lastPost = millis();
}

void loop() {
  // Drain the sensor FIFO so no samples are lost.
  sensor.check();
  while (sensor.available()) {
    processSample(sensor.getFIFOIR(), sensor.getFIFORed());
    sensor.nextSample();
  }

  unsigned long now = millis();

  if (WiFi.status() != WL_CONNECTED && now - lastWifiTry > WIFI_RETRY_MS) {
    lastWifiTry = now;
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }

  if (now - lastPost >= POST_EVERY_MS) {
    lastPost = now;
    bool signalOk = (beatAvg > 0);
    postReading(signalOk, beatAvg, spo2Out);
    drawScreen(signalOk);
  }
}
