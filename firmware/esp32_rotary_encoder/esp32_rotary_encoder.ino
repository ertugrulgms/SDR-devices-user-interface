/*
 * ============================================================================
 *  SDR EH Kontrol Paneli — ESP32 + ARTIMLI (INCREMENTAL) ROTARY ENCODER
 *  Enkoder: E38S6G5-600B-G24N  (600 PPR, kuadratür A/B + Z index, NPN açık-kollektör)
 * ============================================================================
 *  Görev: Yönlü anten ELLE döndürülürken optik enkoderden anlık açıyı (0-360°)
 *  okuyup USB seri porttan bilgisayara yollar. Manyetik enkoderin (AS5600) mıknatıs
 *  eksantriklik/kayma sorunları BURADA YOKTUR — optik, kesin, doğrusal.
 *
 *  PROTOKOL (PC tarafı bunu bekliyor — DEĞİŞTİRME, AS5600 ile AYNI):
 *    - Baud: 115200
 *    - Çıkış satırı:  "ANGLE:245.5\n"   (float derece, "ANGLE:" ile başlar)
 *    - Gelen komutlar:
 *        "CMD:ZERO\n" -> şu anki yönü KUZEY (0°) yap. (Artımlı enkoder MUTLAK açı bilmez;
 *                        her açılışta anteni Kuzey'e çevirip bir kez bunu gönder.)
 *        "CMD:INFO\n" -> anlık sayım/açı/ayar teşhisi yaz (kablolama testi).
 *        "CMD:SCAN\n" / "CMD:STOP\n" -> (opsiyonel servo; ENABLE_SERVO ile).
 *
 *  ============================ DONANIM / KABLOLAMA ==========================
 *  E38S6G5-600B-G24N tipik kablo renkleri (ÜRETİCİYE GÖRE DEĞİŞİR — datasheet'ini DOĞRULA):
 *      Kırmızı  -> VCC  (+5V ... +24V)        [5V kullan]
 *      Siyah    -> GND
 *      Yeşil    -> A fazı
 *      Beyaz    -> B fazı
 *      Sarı     -> Z fazı (index, tur başına 1 puls)  [opsiyonel]
 *
 *  ⚠️ GERİLİM UYARISI (KRİTİK — ESP32'yi yakmamak için):
 *      Çıkışlar NPN AÇIK-KOLLEKTÖR: transistor GND'ye çeker, boştayken YÜZER.
 *      => A/B/Z hatlarını 3.3V'a PULL-UP direnç ile çek (2.2k–4.7k), ASLA 5V'a değil!
 *      Enkoderi 5V ile besle, ama sinyal seviyesini pull-up 3.3V belirlesin (açık-kollektör bunu sağlar).
 *      Harici pull-up yoksa aşağıda USE_INTERNAL_PULLUP=1 ile ESP32 dahili pull-up (zayıf ~45k)
 *      kullanılır — kısa kablo/yavaş dönüşte iş görür; uzun kabloda HARİCİ 3.3V pull-up şart.
 *      NOT: Enkoderin "push-pull/voltaj çıkış" versiyonuysa (5V sürer) DOĞRUDAN BAĞLAMA ->
 *      seviye çeviriciyle veya gerilim böleriyle 3.3V'a indir.
 *
 *  ÇÖZÜNÜRLÜK: 600 PPR × 4 (kuadratür X4) = 2400 sayım/tur -> 360/2400 = 0.15°/sayım.
 * ============================================================================
 */

// ----------------------------------------------------------------------------
//  AYARLAR — kendi kartına/kablolamana göre düzenle
// ----------------------------------------------------------------------------
#define PIN_A               8        // Enkoder A fazı (yeşil) -> GPIO8  (kesme destekli pin)
#define PIN_B               9        // Enkoder B fazı (beyaz) -> GPIO9  (kesme destekli pin)
#define PIN_Z               6        // Enkoder Z index (sarı) -> GPIO6  (USE_Z_INDEX=0 ise boş bırak)

#define PPR                 600      // Enkoder etiketindeki puls/tur (E38S6G5-600B -> 600)
#define COUNTS_PER_REV      (4 * PPR)// Kuadratür X4 -> 2400 sayım/tur (A ve B'nin her kenarı sayılır)
#define DIRECTION_SIGN      (+1)     // Dönüş yönü: +1 saat yönü artan; ters okuyorsa -1 yap
#define SERIAL_BAUD         115200   // PC ile aynı olmalı
#define OUTPUT_INTERVAL_MS  33       // ~30 Hz açı yayını (elle döndürmede akıcı)

#define USE_INTERNAL_PULLUP 1        // 1: ESP32 dahili pull-up (harici yoksa). Uzun kabloda 0 + harici 3.3V pull-up.
#define USE_Z_INDEX         0        // 1: Z index'i kullan (tur başına sayım hatasını sıfırlar; aşağıya bak)

// Servo (otonom tarama) — İSTEĞE BAĞLI (genelde kullanılmaz).
#define ENABLE_SERVO        0
#define SERVO_PIN           7
#define SERVO_STEP_DEG      1
#define SERVO_STEP_MS       30

// ----------------------------------------------------------------------------
//  Durum değişkenleri
// ----------------------------------------------------------------------------
volatile long    encCount   = 0;     // ham kuadratür sayımı (ISR günceller)
volatile uint8_t abState    = 0;     // son A/B durumu (2-bit): (A<<1)|B
volatile bool    pulseSeen  = false; // hiç puls geldi mi (kablolama teşhisi)

long          zeroCount    = 0;      // KUZEY referansı (CMD:ZERO ile ayarlanır)
long          lastAngleCnt = -999999;
unsigned long lastOutput   = 0;
unsigned long lastDebug    = 0;
bool          debugPins    = false;  // CMD:DEBUG ile açılır: A/B ham seviyelerini yaz (kanal testi)
String        rxBuf        = "";

#if USE_Z_INDEX
volatile bool    zSeen      = false;
volatile long    zCountRef  = 0;     // Z ilk görüldüğündeki sayım (tur-hizası referansı)
volatile bool    zHaveRef   = false;
#endif

#if ENABLE_SERVO
  #include <ESP32Servo.h>
  Servo scanServo;
  int   servoPos = 0, servoDir = SERVO_STEP_DEG;
  bool  scanning = false;
  unsigned long lastServo = 0;
#endif

// ----------------------------------------------------------------------------
//  KUADRATÜR ÇÖZÜCÜ (X4) — kesme (interrupt) tabanlı, sayım kaçırmaz
// ----------------------------------------------------------------------------
// Durum geçiş tablosu: index = (eskiDurum<<2)|yeniDurum, değer = +1 / -1 / 0(geçersiz).
// A/B Gray kodu sırayla değişir; her geçerli kenar +/-1 sayılır (X4 -> 2400/tur).
static const int8_t QDEC_TABLE[16] = {
   0, +1, -1,  0,
  -1,  0,  0, +1,
  +1,  0,  0, -1,
   0, -1, +1,  0
};

void IRAM_ATTR onEncoderAB() {
  uint8_t s = (uint8_t)((digitalRead(PIN_A) << 1) | digitalRead(PIN_B));
  int8_t d = QDEC_TABLE[(abState << 2) | s];
  if (d != 0) { encCount += d; pulseSeen = true; }
  abState = s;
}

#if USE_Z_INDEX
// Z index: tur başına bir kez tetiklenir. Mekanik olarak SABİT bir açıya karşılık gelir; buradaki
// sayımın (COUNTS_PER_REV modülünde) HER TURDA AYNI olması gerekir. Küçük sayım hataları birikirse
// Z anında düzeltiriz (uzun süreli sürüklenmeyi (drift) sıfırlar). Kuzey'i CMD:ZERO belirler.
void IRAM_ATTR onEncoderZ() {
  if (!zHaveRef) { zCountRef = encCount; zHaveRef = true; }
  else {
    long expectedMod = ((zCountRef % COUNTS_PER_REV) + COUNTS_PER_REV) % COUNTS_PER_REV;
    long actualMod   = ((encCount  % COUNTS_PER_REV) + COUNTS_PER_REV) % COUNTS_PER_REV;
    long err = actualMod - expectedMod;
    if (err >  COUNTS_PER_REV / 2) err -= COUNTS_PER_REV;
    if (err < -COUNTS_PER_REV / 2) err += COUNTS_PER_REV;
    encCount -= err;                       // birikmiş küçük sayım hatasını düzelt
  }
  zSeen = true;
}
#endif

// Ham sayımdan 0-360° açı (yön işareti + Kuzey ofseti uygulanmış).
float countToAngle(long cnt) {
  long c = (cnt - zeroCount) * DIRECTION_SIGN;
  c = ((c % COUNTS_PER_REV) + COUNTS_PER_REV) % COUNTS_PER_REV;   // 0..CPR-1
  return (float)c * 360.0f / (float)COUNTS_PER_REV;
}

// ----------------------------------------------------------------------------
//  Komutlar
// ----------------------------------------------------------------------------
void handleCommand(String cmd) {
  cmd.trim();
  if (cmd == "CMD:ZERO") {
    noInterrupts(); zeroCount = encCount; interrupts();
    Serial.println("INFO: KUZEY ayarlandi (bu yon = 0 derece).");
  } else if (cmd == "CMD:INFO") {
    long c; bool ps;
    noInterrupts(); c = encCount; ps = pulseSeen; interrupts();
    Serial.print("INFO: sayim="); Serial.print(c);
    Serial.print(" aci=");        Serial.print(countToAngle(c), 1);
    Serial.print(" CPR=");        Serial.print(COUNTS_PER_REV);
    Serial.print(" puls_geldi=");Serial.print(ps ? "EVET" : "HAYIR(A/B/GND/pull-up kontrol)");
    Serial.println();
  } else if (cmd == "CMD:DEBUG") {
    debugPins = !debugPins;
    Serial.println(debugPins ? "INFO: DEBUG ACIK — yavasca cevir; A ve B'nin IKISI de 0/1 degismeli."
                             : "INFO: DEBUG kapali.");
  }
#if ENABLE_SERVO
  else if (cmd == "CMD:SCAN") { scanning = true;  Serial.println("INFO: Tarama BASLADI."); }
  else if (cmd == "CMD:STOP") { scanning = false; Serial.println("INFO: Tarama DURDU."); }
#endif
}

void pollSerial() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (rxBuf.length() > 0) { handleCommand(rxBuf); rxBuf = ""; }
    } else {
      rxBuf += ch;
      if (rxBuf.length() > 32) rxBuf = "";   // taşma koruması
    }
  }
}

// ----------------------------------------------------------------------------
//  setup
// ----------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(300);

  int mode = (USE_INTERNAL_PULLUP ? INPUT_PULLUP : INPUT);
  pinMode(PIN_A, mode);
  pinMode(PIN_B, mode);
  abState = (uint8_t)((digitalRead(PIN_A) << 1) | digitalRead(PIN_B));
  attachInterrupt(digitalPinToInterrupt(PIN_A), onEncoderAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_B), onEncoderAB, CHANGE);

#if USE_Z_INDEX
  pinMode(PIN_Z, mode);
  attachInterrupt(digitalPinToInterrupt(PIN_Z), onEncoderZ, RISING);
#endif

#if ENABLE_SERVO
  scanServo.attach(SERVO_PIN);
  scanServo.write(servoPos);
#endif

  Serial.println("INFO: ESP32 ROTARY ENCODER (E38S6G5-600B) basladi.");
  Serial.print  ("INFO: PPR="); Serial.print(PPR);
  Serial.print  (" X4 sayim/tur="); Serial.print(COUNTS_PER_REV);
  Serial.println(" (0.15 derece/sayim).");
  Serial.println("INFO: Anteni KUZEY'e cevirip 'CMD:ZERO' gonder. Test icin 'CMD:INFO'.");
}

// ----------------------------------------------------------------------------
//  loop
// ----------------------------------------------------------------------------
void loop() {
  pollSerial();

  unsigned long now = millis();

  // KANAL TEŞHİSİ (CMD:DEBUG): A/B ham seviyeleri + sayım. Yavaşça çevirince A ve B'nin İKİSİ de
  // 0<->1 değişmeli. Biri SABİT kalıyorsa o kanal gelmiyor (kablo/pull-up/temas) -> sayım yerinde
  // sayar (0 ve 359.9). ~10 Hz yaz.
  if (debugPins && (now - lastDebug) >= 100) {
    lastDebug = now;
    long c; noInterrupts(); c = encCount; interrupts();
    Serial.print("DEBUG: A="); Serial.print(digitalRead(PIN_A));
    Serial.print(" B=");       Serial.print(digitalRead(PIN_B));
    Serial.print(" sayim=");   Serial.println(c);
  }

  // ~30 Hz "ANGLE:xxx.x" yay (değişmese de gönder; PC bağlantı canlılığını görür)
  if ((now - lastOutput) >= OUTPUT_INTERVAL_MS) {
    lastOutput = now;
    long c;
    noInterrupts(); c = encCount; interrupts();
    float ang = countToAngle(c);
    Serial.print("ANGLE:");
    Serial.println(ang, 1);
  }

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
