#include <SPI.h>

// 引脚定义
#define ADS1256_CS_PIN   5
#define ADS1256_DRDY_PIN 17

// ADS1256寄存器地址
#define REG_STATUS  0x00
#define REG_MUX     0x01
#define REG_ADCON   0x02
#define REG_DRATE   0x03
#define REG_IO      0x04

// SPI设置
SPISettings ADS1256_SPI(1920000, MSBFIRST, SPI_MODE1); // SPI模式1，1.92MHz

void writeRegister(uint8_t reg, uint8_t data) {
  SPI.beginTransaction(ADS1256_SPI);
  digitalWrite(ADS1256_CS_PIN, LOW);
  delayMicroseconds(5);
  SPI.transfer(0x50 | reg); // 写寄存器命令
  SPI.transfer(0x00);       // 写入1字节
  SPI.transfer(data);
  delayMicroseconds(5);
  digitalWrite(ADS1256_CS_PIN, HIGH);
  SPI.endTransaction();
}

void setup() {
  Serial.begin(115200);
  pinMode(ADS1256_CS_PIN, OUTPUT);
  pinMode(ADS1256_DRDY_PIN, INPUT);
  digitalWrite(ADS1256_CS_PIN, HIGH);
  
  SPI.begin();

  // 初始化ADS1256
  writeRegister(REG_MUX, 0x01);      // AIN0和AIN1（通道1）
  writeRegister(REG_ADCON, 0x20);    // PGA=1, 缓冲使能
  writeRegister(REG_DRATE, 0xF0);    // 1000SPS (0xF0)
  
  // 发送同步命令
  SPI.beginTransaction(ADS1256_SPI);
  digitalWrite(ADS1256_CS_PIN, LOW);
  delayMicroseconds(5);
  SPI.transfer(0xFC); // SYNC命令
  delayMicroseconds(5);
  digitalWrite(ADS1256_CS_PIN, HIGH);
  SPI.endTransaction();
  
  // 发送WAKEUP命令开始连续转换
  SPI.beginTransaction(ADS1256_SPI);
  digitalWrite(ADS1256_CS_PIN, LOW);
  delayMicroseconds(5);
  SPI.transfer(0x00); // WAKEUP
  delayMicroseconds(5);
  digitalWrite(ADS1256_CS_PIN, HIGH);
  SPI.endTransaction();
}

int32_t readADC() {
  while (digitalRead(ADS1256_DRDY_PIN) == HIGH); // 等待数据就绪

  SPI.beginTransaction(ADS1256_SPI);
  digitalWrite(ADS1256_CS_PIN, LOW);
  delayMicroseconds(5);
  
  SPI.transfer(0x01); // 读取数据命令
  delayMicroseconds(5);
  
  // 读取3字节数据
  uint8_t b1 = SPI.transfer(0);
  uint8_t b2 = SPI.transfer(0);
  uint8_t b3 = SPI.transfer(0);
  
  digitalWrite(ADS1256_CS_PIN, HIGH);
  SPI.endTransaction();

  // 组合为32位有符号整数
  int32_t value = ((int32_t)b1 << 16) | ((int32_t)b2 << 8) | b3;
  if (value & 0x00800000) { // 符号扩展
    value |= 0xFF000000;
  }
  return value;
}

void loop() {
  static uint32_t lastTime = 0;
  int32_t adcValue = readADC();
  
  // 按1000Hz输出数据
  if (millis() - lastTime >= 1) {
    Serial.println(adcValue);
    lastTime = millis();
  }
}