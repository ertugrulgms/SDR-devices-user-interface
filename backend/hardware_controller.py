import os
import json
import serial
import time
import threading
import logging
import numpy as np

# Modül loglayıcısı: seri-port hataları artık sessizce yutulmaz, görünür kılınır (bulgu #12).
# DF/servo mantığı DEĞİŞMEDİ; yalnızca hatalar teşhis için loglanır.
logger = logging.getLogger(__name__)

# Enkoder açı-doğrusallık DÜZELTME tablosu (mıknatıs eksantrikliği/eğikliği kaynaklı hata için).
# tools/calibrate_encoder.py ile üretilir: bilinen açılarda encoder ne okuyor kaydedilir; canlı açı
# bu tablodan interpolasyonla GERÇEK açıya çevrilir. Yoksa düzeltme uygulanmaz (ham açı döner).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENC_CAL_PATH = os.path.join(_PROJECT_ROOT, "data", "encoder_cal.json")


class HardwareController:
    """
    ESP32 ile Seri Port (USB) üzerinden haberleşerek:
    1. Servo motorlara tarama/dönme komutu gönderir.
    2. AS5600 manyetik enkoderden gelen gerçek zamanlı açı verilerini okur.
    """
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.current_angle = 0.0
        self._read_thread = None
        self._running = False
        # Açı düzeltme tablosu (mıknatıs eksantrikliği): _cal_m=ölçülen, _cal_t=gerçek (monoton açılmış).
        self._cal_m = None
        self._cal_t = None
        self._load_calibration()

    def _load_calibration(self, path: str = None):
        """data/encoder_cal.json varsa açı düzeltme tablosunu yükler. Şema:
        {"measured_deg": [...], "true_deg": [...]}. En az 3 nokta gerekir; yoksa düzeltme kapalı."""
        path = path or _ENC_CAL_PATH
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            meas = [float(x) % 360.0 for x in d["measured_deg"]]
            tru = [float(x) % 360.0 for x in d["true_deg"]]
            if len(meas) >= 3 and len(meas) == len(tru):
                idx = np.argsort(meas)
                m = np.array(meas)[idx]
                t = np.array(tru)[idx]
                # true'yu ölçülen sırasında MONOTON yap (360° sıçramasını aç) -> dairesel interpolasyon
                self._cal_t = np.degrees(np.unwrap(np.radians(t)))
                self._cal_m = m
                logger.info("Enkoder açı düzeltmesi yüklendi: %d nokta (%s)", len(m), path)
        except (OSError, ValueError, KeyError, TypeError):
            self._cal_m = self._cal_t = None      # tablo yok/bozuk -> ham açı (düzeltmesiz)

    def _apply_calibration(self, reported_deg: float) -> float:
        """Ham (encoder) açısını düzeltme tablosundan interpolasyonla GERÇEK açıya çevirir.
        Tablo yoksa açıyı olduğu gibi döner (geriye uyumlu). Dairesel (0/360 sarma-duyarlı)."""
        if self._cal_m is None:
            return reported_deg
        r = float(reported_deg) % 360.0
        m_ext = np.concatenate([self._cal_m, [self._cal_m[0] + 360.0]])
        t_ext = np.concatenate([self._cal_t, [self._cal_t[0] + 360.0]])
        if r < m_ext[0]:
            r += 360.0
        return float(np.interp(r, m_ext, t_ext)) % 360.0

    def connect(self, verify_angle: bool = False, verify_timeout: float = 1.2) -> bool:
        """Seri porta bağlanır. verify_angle=True ise, portun GERÇEKTEN 'ANGLE:' verisi gönderdiğini
        DOĞRULAR (aksi halde bağlanmaz). KRİTİK: ESP32-S3 native USB İKİ CDC arayüzü yaratır — biri
        upload/JTAG (ANGLE yok), diğeri açı gönderir. Eski kod 'açılan ilk porta' bağlanıyordu ->
        yanlış (sessiz) porta düşünce 'bağlı' görünüp AÇI GELMİYORDU. Artık ANGLE gören porta bağlanır."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
        except Exception as e:
            logger.warning("HardwareController: '%s' portuna bağlanılamadı: %s", self.port, e)
            self.is_connected = False
            return False

        if verify_angle:
            # Portu ~verify_timeout sn dinle; "ANGLE:xxx" satırı görürsek bu DOĞRU porttur.
            try:
                self.serial_conn.reset_input_buffer()
            except Exception:
                pass
            got = False
            deadline = time.time() + verify_timeout
            while time.time() < deadline:
                try:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                except Exception:
                    break
                if line.startswith("ANGLE:"):
                    try:
                        self.current_angle = float(line.split(":")[1])
                        got = True
                        break
                    except (ValueError, IndexError):
                        pass
            if not got:
                logger.warning("HardwareController: '%s' açıldı ama ANGLE verisi GELMİYOR "
                               "(yanlış CDC arayüzü olabilir) -> atlanıyor.", self.port)
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
                self.is_connected = False
                return False

        self.is_connected = True
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        return True

    def _read_loop(self):
        while self._running and self.is_connected:
            try:
                # GECİKMESİZ OKUMA: tampondaki TÜM bekleyen satırları oku, yalnızca EN SON açıyı tut.
                # Eskiden her iterasyonda TEK satır okunup 10ms uyunuyordu; hızlı dönüşte seri tampon
                # birikince eski açılar gösterilip GECİKME oluşuyordu (Serial Monitor akıcı, uygulama
                # takik). Tamponu her turda boşaltmak -> current_angle her zaman ANLIK değeri taşır.
                latest = None
                while self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("ANGLE:"):     # Beklenen format: "ANGLE:245.5"
                        latest = line
                if latest is not None:
                    try:
                        self.current_angle = float(latest.split(":")[1])
                    except (ValueError, IndexError):
                        pass
            except Exception as e:
                logger.warning("HardwareController: seri okuma hatası, bağlantı düşürülüyor: %s", e)
                self.is_connected = False
                break
            time.sleep(0.005)   # 10ms -> 5ms: daha sık kontrol, daha düşük gecikme

    def start_scan(self):
        """Servo motora 360 derece tarama komutu gönder."""
        if self.is_connected:
            try:
                self.serial_conn.write(b"CMD:SCAN\n")
            except Exception as e:
                logger.warning("HardwareController: SCAN komutu gönderilemedi: %s", e)

    def stop_scan(self):
        """Servo motor taramasını durdur."""
        if self.is_connected:
            try:
                self.serial_conn.write(b"CMD:STOP\n")
            except Exception as e:
                logger.warning("HardwareController: STOP komutu gönderilemedi: %s", e)

    def get_angle(self) -> float:
        """Anten açısı (derece). Kalibrasyon tablosu varsa mıknatıs eksantriklik hatası düzeltilir."""
        return self._apply_calibration(self.current_angle)

    def disconnect(self):
        self._running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
