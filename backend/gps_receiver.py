"""GPS alıcısı: USB/seri (u-blox vb.) bir GPS modülünden NMEA cümlelerini okur ve anlık konumu
(enlem/boylam/irtifa) sağlar. Hareketli-alıcı konum belirleme (spec 5.1.5) bunu kullanır.

Tasarım HardwareController ile aynı: modül yoksa/ bağlanamıyorsa nesne yine kurulur ama
is_connected=False ve has_fix=False kalır — SAHTE konum ÜRETİLMEZ. Konum yalnızca gerçek bir GPS
fix'ten gelir. u-blox USB dongle tipik olarak /dev/ttyACM0'da 9600 baud NMEA akıtır.

Ayrıştırma (parse_gpgga) saf ve donanımsız birim-test edilebilir.
"""
import time
import threading
import logging

try:
    import serial
except Exception:  # pragma: no cover - ortam bağımlı
    serial = None

logger = logging.getLogger(__name__)


def _nmea_checksum_ok(line: str) -> bool:
    """NMEA satırının '*XX' sağlama toplamını doğrular. '*' yoksa True (bazı modüller vermez)."""
    if "*" not in line:
        return True
    try:
        body, cks = line[1:].split("*", 1)  # baştaki '$' at
    except ValueError:
        return False
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cks[:2], 16)
    except ValueError:
        return False


def _dm_to_deg(dm: str, hemi: str) -> float:
    """NMEA ddmm.mmmm (derece-dakika) -> ondalık derece. Güney/Batı negatif."""
    if not dm:
        return None
    dot = dm.find(".")
    deg_len = dot - 2 if dot >= 2 else len(dm) - 2      # enlem 2, boylam 3 haneli derece
    deg = float(dm[:deg_len])
    minutes = float(dm[deg_len:])
    val = deg + minutes / 60.0
    if hemi in ("S", "W"):
        val = -val
    return val


def parse_gpgga(line: str):
    """$GPGGA / $GNGGA cümlesini ayrıştırır. Dönüş: {lat,lon,alt,fix_quality,num_sats} veya None.
    fix_quality 0 -> geçersiz fix (konum yok). Sağlama toplamı hatalıysa None."""
    line = line.strip()
    if "GGA" not in line[:7]:
        return None
    if not _nmea_checksum_ok(line):
        return None
    core = line.split("*", 1)[0]
    f = core.split(",")
    # 0:$..GGA 1:time 2:lat 3:N/S 4:lon 5:E/W 6:fixq 7:numsat 8:HDOP 9:alt 10:M ...
    if len(f) < 10:
        return None
    try:
        fix_quality = int(f[6]) if f[6] else 0
    except ValueError:
        fix_quality = 0
    if fix_quality == 0:
        return {"fix_quality": 0, "lat": None, "lon": None, "alt": None,
                "num_sats": int(f[7]) if f[7].isdigit() else 0}
    lat = _dm_to_deg(f[2], f[3])
    lon = _dm_to_deg(f[4], f[5])
    try:
        alt = float(f[9]) if f[9] else 0.0
    except ValueError:
        alt = 0.0
    if lat is None or lon is None:
        return None
    return {"fix_quality": fix_quality, "lat": lat, "lon": lon, "alt": alt,
            "num_sats": int(f[7]) if f[7].isdigit() else 0}


class GPSReceiver:
    """USB/seri GPS modülünden NMEA okur; anlık konumu (enlem/boylam/irtifa) sağlar."""

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.has_fix = False
        self._lat = None
        self._lon = None
        self._alt = None
        self._num_sats = 0
        self._last_fix_ts = 0.0
        self._lock = threading.Lock()
        self._thread = None
        self._running = False

    def connect(self) -> bool:
        if serial is None:
            logger.warning("GPSReceiver: pyserial yok — GPS devre dışı.")
            return False
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            logger.warning("GPSReceiver: '%s' portuna bağlanılamadı: %s", self.port, e)
            self.is_connected = False
            return False

    def _read_loop(self):
        while self._running and self.is_connected:
            try:
                line = self.serial_conn.readline().decode("ascii", errors="ignore")
                if line:
                    self._ingest(line)
            except Exception as e:
                logger.warning("GPSReceiver: seri okuma hatası, bağlantı düşürülüyor: %s", e)
                self.is_connected = False
                break

    def _ingest(self, line: str):
        """Bir NMEA satırını işler (test edilebilir: seri portsuz doğrudan çağrılabilir)."""
        rec = parse_gpgga(line)
        if rec is None:
            return
        with self._lock:
            self._num_sats = rec.get("num_sats", 0)
            if rec["fix_quality"] > 0 and rec["lat"] is not None:
                self._lat, self._lon, self._alt = rec["lat"], rec["lon"], rec["alt"]
                self._last_fix_ts = time.time()
                self.has_fix = True
            else:
                self.has_fix = False

    def get_position(self):
        """Anlık konum (lat, lon, alt) veya fix yoksa None. Sahte değer ÜRETİLMEZ."""
        with self._lock:
            if not self.has_fix or self._lat is None:
                return None
            return (self._lat, self._lon, self._alt)

    def status(self) -> dict:
        with self._lock:
            return {"connected": self.is_connected, "has_fix": self.has_fix,
                    "num_sats": self._num_sats,
                    "lat": self._lat, "lon": self._lon, "alt": self._alt}

    def disconnect(self):
        self._running = False
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.is_connected = False
        self.has_fix = False
