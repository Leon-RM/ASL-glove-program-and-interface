#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Wire.h>

// --- Configuration ---
const int PIN_SDA = 21;
const int PIN_SCL = 1;

const int THUMB  = 4;
const int INDEX  = 5;
const int MIDDLE = 7;
const int RING   = 8; 
const int PINKY  = 9;
const int LED_PIN = 2; 

const int MPU_ADDR = 0x68; 

// --- BLE UUIDs (มาตรฐานสำหรับโปรไฟล์ UART/Serial) ---
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLECharacteristic *pCharacteristicTX;
bool deviceConnected = false;

// --- Moving Average Window ---
const int WINDOW_SIZE = 20; 
int ringReadings[WINDOW_SIZE];
int readIndex = 0;
long ringTotal = 0;
int ringAverage = 0;

// ตัวจัดการสถานะการเชื่อมต่อ Bluetooth
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      digitalWrite(2, HIGH); // ไฟติดค้างเมื่อ Python เชื่อมเข้ามาติดแล้ว
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      digitalWrite(2, LOW);
      pServer->getAdvertising()->start(); // เปิดรับการเชื่อมต่อใหม่ทันทีที่หลุด
    }
};

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  analogSetAttenuation(ADC_11db);
  for (int thisReading = 0; thisReading < WINDOW_SIZE; thisReading++) { ringReadings[thisReading] = 0; }

  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); 
  Wire.write(0);    
  Wire.endTransmission(true);

  // --- เริ่มต้นระบบ Bluetooth BLE ---
  BLEDevice::init("Leon_Glove_BLE"); // ชื่อบลูทูธที่จะโผล่ให้คอมค้นหา
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristicTX = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pCharacteristicTX->addDescriptor(new BLE2902());

  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("[OK] BLE Bluetooth Active! Waiting for Python to connect...");
}

void loop() {
  int t = analogRead(THUMB);
  int i = analogRead(INDEX);
  int m = analogRead(MIDDLE);
  int r_raw = analogRead(RING); 
  int p = analogRead(PINKY);

  ringTotal = ringTotal - ringReadings[readIndex]; 
  ringReadings[readIndex] = r_raw;                
  ringTotal = ringTotal + ringReadings[readIndex]; 
  readIndex = readIndex + 1;
  if (readIndex >= WINDOW_SIZE) { readIndex = 0; }
  ringAverage = ringTotal / WINDOW_SIZE; 

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); 
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);
  
  int16_t rawAccX = Wire.read() << 8 | Wire.read();
  int16_t rawAccY = Wire.read() << 8 | Wire.read();
  int16_t rawAccZ = Wire.read() << 8 | Wire.read();
  int16_t rawTemp = Wire.read() << 8 | Wire.read(); 
  int16_t rawGyrX = Wire.read() << 8 | Wire.read();
  int16_t rawGyrY = Wire.read() << 8 | Wire.read();
  int16_t rawGyrZ = Wire.read() << 8 | Wire.read();

  float accX = rawAccX / 16384.0;
  float accY = rawAccY / 16384.0;
  float accZ = rawAccZ / 16384.0;
  float gyroX = rawGyrX / 131.0;
  float gyroY = rawGyrY / 131.0;
  float gyroZ = rawGyrZ / 131.0;

  // ประกอบร่างข้อมูลตัวเลขล้วนคั่นคอมมา
  String trainableString = String(t) + "," + String(i) + "," + String(m) + "," + 
                           String(ringAverage) + "," + String(p) + "," + 
                           String(accX, 2) + "," + String(accY, 2) + "," + String(accZ, 2) + "," +
                           String(gyroX, 2) + "," + String(gyroY, 2) + "," + String(gyroZ, 2);
  
  // ส่งข้อมูลออกทาง Bluetooth เฉพาะตอนที่มีคอมพิวเตอร์มาเชื่อมต่อแล้วเท่านั้น
  if (deviceConnected) {
    pCharacteristicTX->setValue(trainableString.c_str());
    pCharacteristicTX->notify();
  }

  // มอนิเตอร์ทาง Serial ดูความนิ่งได้เหมือนเดิม
  char monitorBuffer[150];
  sprintf(monitorBuffer, "FINGERS -> T:%4d | I:%4d | M:%4d | R_AVG:%4d | R_Raw:%4d | P:%4d", t, i, m, ringAverage, r_raw, p);
  Serial.print(monitorBuffer);
  Serial.print("  ||  ACCEL -> X:"); Serial.print(accX, 2);  Serial.print(" Y:"); Serial.print(accY, 2);  Serial.print(" Z:"); Serial.print(accZ, 2);
  Serial.print("  ||  GYRO -> X:");  Serial.print(gyroX, 2);  Serial.print(" Y:"); Serial.print(gyroY, 2);  Serial.print(" Z:"); Serial.println(gyroZ, 2);

  delay(25); // ปรับเป็น 25ms เพื่อความสอดคล้องกับข้อจำกัดความเร็วการส่ง Packet ของ BLE (ประมาณ 40 FPS ซึ่งเหลือเฟือสำหรับ Train)
}