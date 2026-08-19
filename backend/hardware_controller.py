import serial
import time
import threading
import logging

# Modül loglayıcısı: seri-port hataları artık sessizce yutulmaz, görünür kılınır (bulgu #12).
# DF/servo mantığı DEĞİŞMEDİ; yalnızca hatalar teşhis için loglanır.
logger = logging.getLogger(__name__)


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

    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            self._running = True
            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._read_thread.start()
            return True
        except Exception as e:
            logger.warning("HardwareController: '%s' portuna bağlanılamadı: %s", self.port, e)
            self.is_connected = False
            return False

    def _read_loop(self):
        while self._running and self.is_connected:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    # Beklenen format: "ANGLE:245.5"
                    if line.startswith("ANGLE:"):
                        try:
                            self.current_angle = float(line.split(":")[1])
                        except ValueError:
                            pass
            except Exception as e:
                logger.warning("HardwareController: seri okuma hatası, bağlantı düşürülüyor: %s", e)
                self.is_connected = False
                break
            time.sleep(0.01)

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
        return self.current_angle

    def disconnect(self):
        self._running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
