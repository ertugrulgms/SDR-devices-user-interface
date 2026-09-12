/*
 * ============================================================================
 *  SDR EH Kontrol Paneli — ESP32-S3 + AS5600 ANTEN AÇI ENKODERİ (firmware)
 * ============================================================================
 *  Görev: Yönlü anten ELLE (ya da servo ile) döndürülürken AS5600 manyetik
 *  enkoderden anlık açıyı (0-360°) okuyup USB seri porttan bilgisayara yollar.
 *
 *  PROTOKOL (PC tarafı bunu bekliyor — DEĞİŞTİRME):
 *    - Baud: 115200
 *    - Çıkış satırı:  "ANGLE:245.5\n"   (float derece, "ANGLE:" ile başlar)
 *    - Gelen komutlar: "CMD:SCAN\n"  -> otonom tarama başlat (servo varsa)
 *                      "CMD:STOP\n"  -> taramayı durdur
 *    - EK (kurulum kolaylığı): "CMD:ZERO\n" -> şu anki yönü KUZEY (0°) yap.
 *      (PC göndermez; kurulumda Arduino Seri Monitör'den bir kez gönderebilirsin.)
 *
 *  DONANIM:
 *    - Kart: ESP32-S3 (ör. S3 Mini). Arduino IDE'de kartı seç, USB'den yükle.
 *    - AS5600 <-> ESP32 (I2C, 3.3V!):
 *          AS5600 VCC -> 3V3    (ASLA 5V değil)
 *          AS5600 GND -> GND
 *          AS5600 SDA -> SDA_PIN (aşağıda tanımlı)
 *          AS5600 SCL -> SCL_PIN (aşağıda tanımlı)
 *          AS5600 DIR -> GND  (dönüş yönü saat yönü; ters isterse 3V3'e bağla)
 *      Mıknatıs, AS5600 çipinin ortasına ~1-3 mm mesafede, anten miline sabit.
 *
 *  NOT: ESP32-S3'te I2C pinleri sabit değildir; AŞAĞIDAKİ SDA_PIN/SCL_PIN'i
 *  kendi lehimlediğin pinlere göre ayarla.
 * ============================================================================
 */

#include <Wire.h>

// ----------------------------------------------------------------------------
//  AYARLAR — kendi donanımına göre düzenle
// ----------------------------------------------------------------------------
#define SDA_PIN            9        // AS5600 SDA -> GPIO9 (kablolamaya göre: SDA=pin9)
#define SCL_PIN            8        // AS5600 SCL -> GPIO8 (kablolamaya göre: SCL=pin8)
#define SERIAL_BAUD        115200   // PC ile aynı olmalı (HardwareController: 115200)
#define OUTPUT_INTERVAL_MS 33       // ~30 Hz açı yayını (elle döndürmede akıcı)

// Açı hizalama:
float  ZERO_OFFSET_DEG = 0.0;       // KUZEY ofseti (CMD:ZERO ile de ayarlanır)
#define DIRECTION_SIGN   (+1)       // dönüş yönü: +1 saat yönü; -1 ters (mekanik ters ise değiştir)
#define SMOOTH_ALPHA     0.35       // 0.0-1.0 hafif yumuşatma (1.0 = yumuşatma yok, ham)

// Servo (otonom tarama) — İSTEĞE BAĞLI. Servo bağlıysa 1 yap ve ESP32Servo kütüphanesini kur.
#define ENABLE_SERVO     0
#define SERVO_PIN        4
#define SERVO_STEP_DEG   1
#define SERVO_STEP_MS    30

// ----------------------------------------------------------------------------
//  AS5600 sabitleri
// ----------------------------------------------------------------------------
#define AS5600_ADDR      0x36
#define AS5600_RAW_ANGLE 0x0C   // 12-bit ham açı (0-4095) — üst bayt 0x0C, alt 0x0D
#define AS5600_STATUS    0x0B   // MD (bit5)=mıknatıs var, ML(bit4)=zayıf, MH(bit3)=güçlü
#define AS5600_AGC       0x1A   // Otomatik Kazanç (3.3V'de 0-128 ideal ~orta; 0/255'e dayanınca kötü)
#define AS5600_MAGNITUDE 0x1B   // CORDIC alan büyüklüğü (12-bit) — mıknatıs alan gücü

#if ENABLE_SERVO
  #include <ESP32Servo.h>
  Servo scanServo;
  int   servoPos = 0;
  int   servoDir = SERVO_STEP_DEG;
#endif

bool          scanning     = false;
float         smoothAngle  = 0.0;
bool          haveAngle    = false;
unsigned long lastOutput   = 0;
unsigned long lastServo    = 0;
unsigned long lastHealth   = 0;      // periyodik AS5600 sağlık teşhisi zamanlayıcısı
int           readFailCount= 0;      // ardışık I2C okuma hatası sayısı (açı donması teşhisi)
String        rxBuf        = "";

// ----------------------------------------------------------------------------
//  AS5600 okuma
// ----------------------------------------------------------------------------
// Ham 12-bit açıyı okur (0-4095). Başarısızsa -1 döner.
int readRawAngle() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(AS5600_RAW_ANGLE);
  if (Wire.endTransmission(false) != 0) return -1;      // I2C hatası
  if (Wire.requestFrom(AS5600_ADDR, 2) != 2) return -1;
  int hi = Wire.read();
  int lo = Wire.read();
  return ((hi << 8) | lo) & 0x0FFF;                     // 12-bit
}

// Tek baytlık AS5600 register okuma (STATUS/AGC vb.). Başarısızsa 0xFF.
uint8_t readReg8(uint8_t reg) {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom(AS5600_ADDR, 1) != 1) return 0xFF;
  return Wire.read();
}

// AS5600 STATUS bayrağı (mıknatıs teşhisi). Başarısızsa 0xFF.
uint8_t readStatus() { return readReg8(AS5600_STATUS); }

// 12-bit alan büyüklüğü (MAGNITUDE). Başarısızsa -1.
int readMagnitude() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(AS5600_MAGNITUDE);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(AS5600_ADDR, 2) != 2) return -1;
  int hi = Wire.read();
  int lo = Wire.read();
  return ((hi << 8) | lo) & 0x0FFF;
}

// İki açı arası en kısa yol (-180..+180) — 359->0 sıçramasını doğru yumuşatmak için.
float shortestDelta(float target, float current) {
  float d = fmodf(target - current + 540.0, 360.0) - 180.0;
  return d;
}

// ----------------------------------------------------------------------------
//  setup
// ----------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);   // AS5600 400 kHz fast-mode destekler

  Serial.println("INFO: ESP32 AS5600 enkoder basladi.");

  uint8_t st = readStatus();
  if (st == 0xFF) {
    Serial.println("INFO: AS5600 bulunamadi! Kablolamayi (SDA/SCL/3V3/GND) ve pinleri kontrol et.");
  } else if (!(st & 0x20)) {
    Serial.println("INFO: Miknatis algilanmadi (MD=0). Miknatisi cipe ~1-3mm yaklastir/ortala.");
  } else {
    Serial.println("INFO: Miknatis OK, aci yayini basliyor.");
  }

#if ENABLE_SERVO
  scanServo.attach(SERVO_PIN);
  scanServo.write(servoPos);
#endif

  int raw = readRawAngle();
  if (raw >= 0) { smoothAngle = raw * 360.0 / 4096.0; haveAngle = true; }
}

// ----------------------------------------------------------------------------
//  Gelen komutları işle (satır satır: "CMD:SCAN", "CMD:STOP", "CMD:ZERO")
// ----------------------------------------------------------------------------
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd == "CMD:SCAN") {
    scanning = true;
    Serial.println("INFO: Tarama BASLADI.");
  } else if (cmd == "CMD:STOP") {
    scanning = false;
    Serial.println("INFO: Tarama DURDU.");
  } else if (cmd == "CMD:ZERO") {
    // Su anki fiziksel yonu 0° (Kuzey) kabul et.
    int raw = readRawAngle();
    if (raw >= 0) {
      float phys = raw * 360.0 / 4096.0;
      ZERO_OFFSET_DEG = phys;   // bu fiziksel aci artik 0 sayilir
      Serial.print("INFO: KUZEY ayarlandi (ofset=");
      Serial.print(ZERO_OFFSET_DEG, 1);
      Serial.println(").");
    }
  }
}

void pollSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxBuf.length() > 0) { handleCommand(rxBuf); rxBuf = ""; }
    } else {
      rxBuf += c;
      if (rxBuf.length() > 32) rxBuf = "";   // taşma koruması
    }
  }
}

// ----------------------------------------------------------------------------
//  loop
// ----------------------------------------------------------------------------
void loop() {
  pollSerial();

  // --- Açıyı oku + hizala + yumuşat ---
  int raw = readRawAngle();
  if (raw >= 0) {
    readFailCount = 0;                                  // I2C sağlıklı: hata sayacını sıfırla
    float phys = raw * 360.0 / 4096.0;                 // ham fiziksel açı
    // Kuzey ofseti + yön işareti uygula
    float ang = DIRECTION_SIGN * (phys - ZERO_OFFSET_DEG);
    ang = fmodf(ang + 360.0, 360.0);                   // 0-360'a sar

    if (!haveAngle) { smoothAngle = ang; haveAngle = true; }
    else {
      // Sarma-duyarlı yumuşatma (359<->0 sıçramasını doğru geçer)
      smoothAngle += SMOOTH_ALPHA * shortestDelta(ang, smoothAngle);
      smoothAngle = fmodf(smoothAngle + 360.0, 360.0);
    }
  } else {
    readFailCount++;                                   // I2C okuması başarısız (SDA/SCL/VCC gevşek olabilir)
  }

  unsigned long now = millis();

  // --- SAĞLIK TEŞHİSİ (~1 sn): AÇI DONMASINI SESSİZ BIRAKMA ---
  // I2C okuması başarısızsa firmware eski açıyı yollamaya devam eder -> arayüzde açı DONMUŞ görünür
  // ama sebebi belli olmaz. Burada arızayı AÇIKÇA bildiririz (arayüz "ANGLE:" dışını yok sayar;
  // Arduino Seri Monitör'de görürsün). Böylece "kod mu donanım mı" sorusu anında yanıtlanır.
  if (now - lastHealth >= 1000) {
    lastHealth = now;
    if (readFailCount > 0) {
      Serial.println("INFO: AS5600 I2C OKUNAMIYOR — aci DONMUS. SDA/SCL/VCC(3V3)/GND kablolarini kontrol et.");
    } else {
      uint8_t st  = readStatus();
      uint8_t agc = readReg8(AS5600_AGC);
      int     mag = readMagnitude();
      if (st == 0xFF) {
        Serial.println("INFO: AS5600 I2C hatasi (STATUS okunamadi) — kablolari kontrol et.");
      } else {
        // MD=mıknatıs var, ML=zayıf/uzak, MH=çok güçlü/yakın. AGC 3.3V'de ~orta ideal;
        // 0'a veya 255'e dayanmışsa mıknatıs mesafesi kötü. Bu satırı Seri Monitör'de izle:
        // mıknatısı, MD=1 & ML=0 & MH=0 & AGC ORTA olacak şekilde konumlandir.
        bool md = st & 0x20, ml = st & 0x10, mh = st & 0x08;
        Serial.print("TESHIS: MD="); Serial.print(md);
        Serial.print(" ML=");        Serial.print(ml);
        Serial.print(" MH=");        Serial.print(mh);
        Serial.print(" AGC=");       Serial.print(agc);
        Serial.print(" MAG=");       Serial.print(mag);
        if (!md)      Serial.println("  -> MIKNATIS YOK/UZAK! (aci COP okur) miknatisi yaklastir/ortala.");
        else if (ml)  Serial.println("  -> ZAYIF/UZAK: cipe biraz yaklastir.");
        else if (mh)  Serial.println("  -> COK YAKIN/GUCLU: biraz uzaklastir.");
        else          Serial.println("  -> MIKNATIS IYI.");
      }
    }
  }

  // --- Belirli aralıkla "ANGLE:xxx.x" yay ---
  if (haveAngle && (now - lastOutput) >= OUTPUT_INTERVAL_MS) {
    lastOutput = now;
    Serial.print("ANGLE:");
    Serial.println(smoothAngle, 1);     // tek ondalık (ör. ANGLE:245.5)
  }

  // --- Otonom tarama (servo) — sadece ENABLE_SERVO ise ---
#if ENABLE_SERVO
  if (scanning && (now - lastServo) >= SERVO_STEP_MS) {
    lastServo = now;
    servoPos += servoDir;
    if (servoPos >= 180) { servoPos = 180; servoDir = -SERVO_STEP_DEG; }
    if (servoPos <= 0)   { servoPos = 0;   servoDir =  SERVO_STEP_DEG; }
    scanServo.write(servoPos);
  }
#endif
}
