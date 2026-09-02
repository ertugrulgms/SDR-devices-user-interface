#!/usr/bin/env python3
"""YARDIMCI DF İSTASYONU — Elektronik Harp Yön Bulma / Konum Köşe İstasyonu.

Üçgen dizilimdeki 3 istasyondan KÖŞE (yardımcı) istasyonlarında çalışır:
  * Köşe-1: ADALM-Pluto SDR   (NODE_ID = "NODE-2")
  * Köşe-2: USRP N210         (NODE_ID = "NODE-3")
(Merkez/tepe köşe ana bilgisayardır — asıl sdr_panel uygulamasını çalıştırır, bu dosyayı DEĞİL.)

GÖREV: Operatör LPDA anteni ELLE 0–360° döndürür. Altındaki AS5600 manyetik enkoder açıyı ESP32
üzerinden USB ile bu bilgisayara verir ("ANGLE:245.5"). Yazılım, ayarlı hedef frekansındaki alınan
GÜCÜ (dBm) sürekli ölçer ve HANGİ AÇIDA EN YÜKSEK GÜCÜ aldığını izler (genlik-tabanlı DF). Bu tepe
açı = kaynağın geliş yönü (kerteriz). Kerteriz, JSON olarak UDP ile merkez bilgisayara (Cat6/statik
IP) gönderilir; merkez, 3 köşenin kerterizlerini üçgenleyerek kaynağın konumunu bulur.

ARAYÜZDE YALNIZCA: RF (frekans/kazanç) · Spektrum · Açı · dBm  (+ kerteriz ve bağlantı durumu).

Kurulum (her yardımcı laptopta):  pip install PyQt6 pyqtgraph pyserial numpy soapysdr
  Pluto için:  SoapyPlutoSDR    |    N210/B200 için:  SoapyUHD  (SoapySDR eklentileri)

Çalıştırma:
  # Pluto istasyonu:
  python aux_station.py --id NODE-2 --sdr "driver=plutosdr" --host 192.168.1.10 --enc /dev/ttyUSB0
  # N210 istasyonu:
  python aux_station.py --id NODE-3 --sdr "driver=uhd,addr=192.168.10.2" --host 192.168.1.10 --enc /dev/ttyUSB0
  # Donanımsız arayüz testi (sahte sinyal + sahte açı):
  python aux_station.py --id NODE-2 --sim
"""
import sys
import json
import time
import socket
import argparse
import threading

import numpy as np

# ----------------------------------------------------------------------------- #
#  VARSAYILAN KONFIGÜRASYON (CLI ile override edilebilir — aşağıdaki argparse'a bak)
# ----------------------------------------------------------------------------- #
DEFAULTS = {
    "id": "NODE-2",                 # merkezdeki data/df_nodes.json ile AYNI olmalı (NODE-2 / NODE-3)
    "sdr": "driver=plutosdr",       # Pluto: "driver=plutosdr" | N210: "driver=uhd,addr=192.168.10.2"
    "host": "192.168.1.10",         # MERKEZ bilgisayarın IP'si (Cat6/statik)
    "port": 5005,                   # merkez UDP dinleme portu (sdr_panel ile aynı)
    "enc": "/dev/ttyUSB0",          # ESP32 (AS5600 enkoder) seri portu
    "baud": 115200,                 # ESP32 baud (mevcut firmware ile aynı)
    "freq": 433.0,                  # başlangıç hedef frekansı (MHz)
    "rate": 2.0,                    # örnekleme hızı (Msps)
    "gain": 40.0,                   # RX kazancı (dB)
    "rate_hz": 10.0,                # merkeze saniyede kaç kerteriz gönderilsin
}

FFT_POINTS = 2048

# --- İsteğe bağlı bağımlılıklar (yoksa nazikçe bildir) ---
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
    _HAVE_SOAPY = True
except Exception:
    _HAVE_SOAPY = False

try:
    import adi as _adi                 # pyadi-iio: ADALM-Pluto (libiio) — SoapyPluto apt'te yoksa (24.04)
    _HAVE_ADI = True
except Exception:
    _HAVE_ADI = False

try:
    import serial as _pyserial
    _HAVE_SERIAL = True
except Exception:
    _HAVE_SERIAL = False

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QDoubleSpinBox, QPushButton, QTextEdit, QGridLayout, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg


# ----------------------------------------------------------------------------- #
#  GENLİK-TABANLI KERTERİZ KESTİRİCİ (anten elle döndürülürken tepe açıyı bulur)
#  (Merkezdeki backend.direction_finding.AmplitudeDFEstimator'ın kompakt, bağımsız kopyası.)
# ----------------------------------------------------------------------------- #
class AmplitudeDF:
    """(açı, güç) örneklerinden en yüksek gücün alındığı azimutu (kerteriz) verir.
    Tepe etrafında güç-ağırlıklı dairesel merkez ile alt-derece hassasiyet. Örnekler zamanla söner
    (yeniden tarama / kaynak değişimi)."""

    def __init__(self, bin_deg=1.0, window_deg=25.0, decay_sec=12.0):
        self.bin_deg = bin_deg
        self.window_deg = window_deg
        self.decay_sec = decay_sec
        self._bins = {}   # az_bin -> (amp_dbm, ts)

    def reset(self):
        self._bins.clear()

    def update(self, azimuth_deg, amp_dbm, now=None):
        now = time.time() if now is None else now
        nb = int(round(360.0 / self.bin_deg))
        key = int(round((azimuth_deg % 360.0) / self.bin_deg)) % nb
        prev = self._bins.get(key)
        if prev is None or amp_dbm >= prev[0] or (now - prev[1]) > self.decay_sec:
            self._bins[key] = (amp_dbm, now)

    def _prune(self, now):
        for k in [k for k, (_, ts) in self._bins.items() if (now - ts) > self.decay_sec]:
            del self._bins[k]

    def bearing(self, now=None):
        """(azimut|None, tepe_dBm, güven[0..1], örnek_sayısı)."""
        now = time.time() if now is None else now
        self._prune(now)
        if len(self._bins) < 3:
            return None, -120.0, 0.0, len(self._bins)
        azs = np.array([k * self.bin_deg for k in self._bins.keys()])
        amps = np.array([v[0] for v in self._bins.values()])
        pk = int(np.argmax(amps))
        peak_az, peak_amp, floor = float(azs[pk]), float(amps[pk]), float(np.min(amps))
        offs = ((azs - peak_az + 180.0) % 360.0) - 180.0
        mask = np.abs(offs) <= self.window_deg
        w = np.power(10.0, (amps[mask] - floor) / 10.0)
        centroid = float(np.sum(offs[mask] * w) / (np.sum(w) + 1e-12))
        bearing = (peak_az + centroid) % 360.0
        conf = float(min(1.0, max(0.0, (peak_amp - floor) / 20.0)))
        return round(bearing, 2), round(peak_amp, 1), round(conf, 2), len(self._bins)


# ----------------------------------------------------------------------------- #
#  ESP32 / AS5600 ENKODER OKUYUCU (USB seri, "ANGLE:245.5" satır formatı)
# ----------------------------------------------------------------------------- #
class EncoderReader:
    def __init__(self, port, baud):
        self.port, self.baud = port, baud
        self.conn = None
        self.connected = False
        self.raw_angle = 0.0          # ham enkoder açısı
        self.north_offset = 0.0       # Kuzey referansı (raw - offset = azimut)
        self._running = False
        self._thread = None

    def connect(self):
        if not _HAVE_SERIAL:
            return False
        try:
            self.conn = _pyserial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True
        except Exception:
            self.connected = False
            return False

    def _loop(self):
        while self._running and self.connected:
            try:
                if self.conn.in_waiting > 0:
                    line = self.conn.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("ANGLE:"):
                        try:
                            self.raw_angle = float(line.split(":")[1])
                        except ValueError:
                            pass
            except Exception:
                self.connected = False
                break
            time.sleep(0.005)

    def set_north(self):
        """Şu anki ham açıyı Kuzey (0°) kabul et."""
        self.north_offset = self.raw_angle

    def azimuth(self):
        """Kuzeye göre azimut (0–360)."""
        return (self.raw_angle - self.north_offset) % 360.0

    def stop(self):
        self._running = False
        if self.conn and self.conn.is_open:
            self.conn.close()
        self.connected = False


# ----------------------------------------------------------------------------- #
#  SDR ALICI (SoapySDR — hem Pluto hem N210/UHD ile çalışır)
# ----------------------------------------------------------------------------- #
class SoapyRx:
    def __init__(self, args, rate_hz, freq_hz, gain_db):
        self.args, self.rate, self.freq, self.gain = args, rate_hz, freq_hz, gain_db
        self.dev = None
        self.stream = None
        self.connected = False
        self.error = None

    def open(self):
        if not _HAVE_SOAPY:
            self.error = "SoapySDR bulunamadı (pip install soapysdr + sürücü eklentisi)."
            return False
        try:
            self.dev = SoapySDR.Device(dict(kv.split("=", 1) for kv in self.args.split(",")))
            self.dev.setSampleRate(SOAPY_SDR_RX, 0, self.rate)
            self.dev.setFrequency(SOAPY_SDR_RX, 0, self.freq)
            self.dev.setGain(SOAPY_SDR_RX, 0, self.gain)
            self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.dev.activateStream(self.stream)
            self.connected = True
            return True
        except Exception as e:
            self.error = f"SDR açılamadı: {e}"
            self.connected = False
            return False

    def set_frequency(self, freq_hz):
        self.freq = freq_hz
        if self.connected:
            try:
                self.dev.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
            except Exception:
                pass

    def set_gain(self, gain_db):
        self.gain = gain_db
        if self.connected:
            try:
                self.dev.setGain(SOAPY_SDR_RX, 0, gain_db)
            except Exception:
                pass

    def read(self, n=FFT_POINTS):
        """n örnek IQ oku (complex64). Hata/az veri -> None."""
        if not self.connected:
            return None
        buff = np.zeros(n, np.complex64)
        try:
            sr = self.dev.readStream(self.stream, [buff], n, timeoutUs=200000)
            if sr.ret <= 0:
                return None
            return buff[:sr.ret]
        except Exception:
            return None

    def close(self):
        try:
            if self.stream is not None:
                self.dev.deactivateStream(self.stream)
                self.dev.closeStream(self.stream)
        except Exception:
            pass
        self.connected = False


# ----------------------------------------------------------------------------- #
#  ADALM-PLUTO ALICI (pyadi-iio / libiio) — SoapyPlutoSDR apt'te yoksa (Ubuntu 24.04).
#  Kurulum SADECE:  pip install pyadi-iio   (derleme YOK). --sdr pluto  /  --sdr ip:192.168.2.1
# ----------------------------------------------------------------------------- #
def _pluto_uri(arg):
    for part in arg.replace(",", " ").split():
        if part.startswith("ip:") or part.startswith("usb:"):
            return part
    return "ip:192.168.2.1"            # USB üstünden varsayılan Pluto adresi


class PlutoRx:
    def __init__(self, arg, rate_hz, freq_hz, gain_db):
        self.uri = _pluto_uri(arg)
        self.rate, self.freq, self.gain = rate_hz, freq_hz, gain_db
        self.dev = None
        self.connected = False
        self.error = None

    def open(self):
        if not _HAVE_ADI:
            self.error = "pyadi-iio yok — kur:  pip install pyadi-iio"
            return False
        try:
            self.dev = _adi.Pluto(uri=self.uri)
            self.dev.sample_rate = int(self.rate)
            self.dev.rx_rf_bandwidth = int(self.rate)
            self.dev.rx_lo = int(self.freq)
            self.dev.gain_control_mode_chan0 = "manual"
            self.dev.rx_hardwaregain_chan0 = float(self.gain)
            self.dev.rx_buffer_size = FFT_POINTS
            self.connected = True
            return True
        except Exception as e:
            self.error = f"Pluto açılamadı ({self.uri}): {e}"
            return False

    def set_frequency(self, freq_hz):
        self.freq = freq_hz
        if self.connected:
            try:
                self.dev.rx_lo = int(freq_hz)
            except Exception:
                pass

    def set_gain(self, gain_db):
        self.gain = gain_db
        if self.connected:
            try:
                self.dev.rx_hardwaregain_chan0 = float(gain_db)
            except Exception:
                pass

    def read(self, n=FFT_POINTS):
        if not self.connected:
            return None
        try:
            return np.asarray(self.dev.rx(), np.complex64)
        except Exception:
            return None

    def close(self):
        self.connected = False


def make_sdr(cfg):
    """--sdr argümanına göre alıcı arka ucunu seçer. 'pluto'/'ip:'/'usb:' -> pyadi-iio (derlemesiz);
    aksi halde SoapySDR (N210/UHD 'driver=uhd,...'). Pluto'da pyadi yoksa SoapySDR'a düşer."""
    low = cfg["sdr"].lower()
    is_pluto = ("pluto" in low) or low.startswith("ip:") or low.startswith("usb:")
    rate, freq, gain = cfg["rate"] * 1e6, cfg["freq"] * 1e6, cfg["gain"]
    if is_pluto and _HAVE_ADI:
        return PlutoRx(cfg["sdr"], rate, freq, gain)
    return SoapyRx(cfg["sdr"], rate, freq, gain)


# ----------------------------------------------------------------------------- #
#  İŞ PARÇACIĞI: IQ oku -> spektrum + dBm ölç -> açı oku -> kerteriz izle -> UDP gönder
# ----------------------------------------------------------------------------- #
class AuxWorker(QThread):
    data_ready = pyqtSignal(dict)     # {fft_dbm, freqs, dbm, azimuth, bearing, conf, n, sdr_ok, enc_ok, sent}
    log = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self.freq_hz = cfg["freq"] * 1e6
        self.gain_db = cfg["gain"]
        self.cal_offset_db = 0.0        # dBFS -> dBm kaba düzeltme (opsiyonel)
        self.sending = True             # merkeze otomatik gönderim
        self.df = AmplitudeDF()
        self.enc = EncoderReader(cfg["enc"], cfg["baud"])
        self.sdr = make_sdr(cfg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_send = 0.0

    # --- arayüzden gelen kontroller ---
    def set_frequency_mhz(self, mhz):
        self.freq_hz = mhz * 1e6
        self.sdr.set_frequency(self.freq_hz)
        self.df.reset()                 # frekans değişti -> eski açı-güç haritası geçersiz
        self.log.emit(f"Frekans -> {mhz:.4f} MHz (kerteriz sıfırlandı)")

    def set_gain(self, g):
        self.gain_db = g
        self.sdr.set_gain(g)

    def set_north(self):
        self.enc.set_north()
        self.log.emit("Kuzey ayarlandı (mevcut açı = 0°).")

    def reset_bearing(self):
        self.df.reset()
        self.log.emit("Kerteriz sıfırlandı — yeni tarama için anteni döndürün.")

    def set_sending(self, on):
        self.sending = on
        self.log.emit("Merkeze gönderim: " + ("AÇIK" if on else "KAPALI"))

    def run(self):
        self._running = True
        # Donanımları aç
        if self.enc.connect():
            self.log.emit(f"Enkoder bağlı: {self.cfg['enc']}")
        else:
            self.log.emit(f"⚠️ Enkoder BAĞLANAMADI ({self.cfg['enc']}) — açı 0 kalır." +
                          ("" if _HAVE_SERIAL else " (pyserial yok)"))
        if self.cfg.get("sim"):
            self.log.emit("SİMÜLASYON modu: sahte sinyal + sahte açı.")
        elif self.sdr.open():
            self.log.emit(f"SDR bağlı: {self.cfg['sdr']}  ({self.cfg['rate']} Msps)")
        else:
            self.log.emit(f"⚠️ {self.sdr.error}")

        win = np.hanning(FFT_POINTS)
        sim_peak_az = 137.0
        while self._running:
            # 1) IQ al (sim veya gerçek)
            if self.cfg.get("sim"):
                az = self.enc.azimuth() if self.enc.connected else (time.time() * 40.0) % 360.0
                # sahte: hedef sim_peak_az'da; anten ona yaklaştıkça güç artar
                d = ((az - sim_peak_az + 180) % 360) - 180
                snr = 25.0 * np.exp(-(d ** 2) / (2 * 20.0 ** 2))
                t = np.arange(FFT_POINTS)
                iq = (10 ** (snr / 20) * np.exp(1j * 2 * np.pi * 0.13 * t)).astype(np.complex64)
                iq += (np.random.randn(FFT_POINTS) + 1j * np.random.randn(FFT_POINTS)).astype(np.complex64)
                time.sleep(0.03)
            else:
                iq = self.sdr.read(FFT_POINTS)
                az = self.enc.azimuth()
                if iq is None or len(iq) < 64:
                    time.sleep(0.02)
                    continue

            # 2) Spektrum (dBm) + hedef gücü
            n = len(iq)
            w = win if n == FFT_POINTS else np.hanning(n)
            fc = np.fft.fftshift(np.fft.fft((iq - np.mean(iq)) * w, n=FFT_POINTS))
            fft_dbm = 10.0 * np.log10(np.abs(fc) ** 2 / FFT_POINTS + 1e-12) + self.cal_offset_db
            # Alınan güç = spektrum tepesi (hedef sinyalin gücü; anten yönüyle değişir)
            peak_dbm = float(np.max(fft_dbm))
            noise = float(np.median(fft_dbm))
            snr_db = peak_dbm - noise

            # 3) Genlik-DF: yalnızca sinyal varken (gürültü açı haritasını kirletmesin)
            if snr_db >= 6.0:
                self.df.update(az, peak_dbm)
            bearing, bpk, conf, ncnt = self.df.bearing()

            # 4) Merkeze kerteriz gönder (JSON/UDP) — hız sınırlı
            sent = False
            now = time.time()
            _rate_hz = self.cfg.get("rate_hz") or 10.0     # None/eksikse güvenli varsayılan (çökme yok)
            if self.sending and bearing is not None and (now - self._last_send) >= (1.0 / _rate_hz):
                msg = {
                    "id": self.cfg["id"],
                    "azimuth_deg": round(bearing, 2),
                    "elevation_deg": 0.0,
                    "amp_dbm": round(bpk, 1),
                    "snr_db": round(snr_db, 1),
                    "freq_mhz": round(self.freq_hz / 1e6, 4),
                }
                try:
                    self.sock.sendto(json.dumps(msg).encode("utf-8"), (self.cfg["host"], self.cfg["port"]))
                    sent = True
                    self._last_send = now
                except Exception as e:
                    self.log.emit(f"⚠️ UDP gönderilemedi: {e}")

            self.data_ready.emit({
                "fft_dbm": fft_dbm,
                "peak_dbm": peak_dbm,
                "snr_db": snr_db,
                "azimuth": az,
                "bearing": bearing, "conf": conf, "n": ncnt,
                "sdr_ok": (self.sdr.connected or bool(self.cfg.get("sim"))),
                "enc_ok": self.enc.connected,
                "sent": sent,
            })

        self.sdr.close()
        self.enc.stop()

    def stop(self):
        self._running = False
        self.wait(1500)


# ----------------------------------------------------------------------------- #
#  ARAYÜZ (sadece: RF · Spektrum · Açı · dBm  + kerteriz/durum)
# ----------------------------------------------------------------------------- #
class AuxWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle(f"DF Yardımcı İstasyon — {cfg['id']}  →  {cfg['host']}:{cfg['port']}")
        self.resize(1000, 720)
        self.setStyleSheet("QMainWindow{background:#1e1e1e;} QLabel{color:#e0e0e0;font-size:15px;}"
                           "QPushButton{background:#2e7d32;color:#fff;font-weight:bold;padding:8px;"
                           "border-radius:4px;font-size:14px;} QTextEdit{background:#141414;color:#ccc;"
                           "font-family:monospace;font-size:12px;border:1px solid #333;}"
                           "QDoubleSpinBox{background:#333;color:#fff;padding:6px;font-size:15px;border:1px solid #555;}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- RF kontrolleri ---
        rf = QGridLayout()
        rf.addWidget(QLabel("RF — Frekans (MHz):"), 0, 0)
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(1.0, 6000.0)
        self.spin_freq.setDecimals(4)
        self.spin_freq.setValue(cfg["freq"])
        self.spin_freq.setSingleStep(0.025)
        rf.addWidget(self.spin_freq, 0, 1)
        rf.addWidget(QLabel("Kazanç (dB):"), 0, 2)
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 90.0)
        self.spin_gain.setValue(cfg["gain"])
        rf.addWidget(self.spin_gain, 0, 3)
        self.btn_north = QPushButton("Kuzeyi Ayarla (0°)")
        rf.addWidget(self.btn_north, 0, 4)
        self.btn_reset = QPushButton("Kerteriz Sıfırla")
        self.btn_reset.setStyleSheet("background:#f57c00;")
        rf.addWidget(self.btn_reset, 0, 5)
        self.chk_send = QCheckBox("Merkeze Gönder")
        self.chk_send.setChecked(True)
        self.chk_send.setStyleSheet("color:#e0e0e0;font-size:14px;")
        rf.addWidget(self.chk_send, 0, 6)
        root.addLayout(rf)

        # --- Büyük göstergeler: Açı ve dBm ---
        big = QHBoxLayout()
        self.lbl_angle = QLabel("Açı: --°")
        self.lbl_angle.setStyleSheet("color:#00e676;font-size:40px;font-weight:bold;")
        self.lbl_angle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_dbm = QLabel("dBm: --")
        self.lbl_dbm.setStyleSheet("color:#4fc3f7;font-size:40px;font-weight:bold;")
        self.lbl_dbm.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big.addWidget(self.lbl_angle)
        big.addWidget(self.lbl_dbm)
        root.addLayout(big)

        # --- Kerteriz + durum satırı ---
        self.lbl_bearing = QLabel("KERTERİZ: (anteni döndürün, tepe açı bulunacak)")
        self.lbl_bearing.setStyleSheet("color:#ffb74d;font-size:18px;font-weight:bold;background:#2a2a2a;"
                                       "padding:8px;border-radius:4px;")
        self.lbl_bearing.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.lbl_bearing)

        self.lbl_status = QLabel("SDR: ? | Enkoder: ? | Gönderim: ?")
        self.lbl_status.setStyleSheet("font-size:14px;color:#9e9e9e;")
        root.addWidget(self.lbl_status)

        # --- Spektrum ---
        root.addWidget(QLabel("SPEKTRUM:"))
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#0d0d0d")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Güç (dBm/dBFS)")
        self.plot.setLabel("bottom", "Frekans offseti (kHz)")
        self.curve = self.plot.plot(pen=pg.mkPen("#00e676", width=1))
        root.addWidget(self.plot, stretch=3)

        # --- Log ---
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        root.addWidget(self.log, stretch=1)

        # --- Worker ---
        self.worker = AuxWorker(cfg)
        self.worker.data_ready.connect(self.on_data)
        self.worker.log.connect(self.add_log)
        self.spin_freq.valueChanged.connect(self.worker.set_frequency_mhz)
        self.spin_gain.valueChanged.connect(self.worker.set_gain)
        self.btn_north.clicked.connect(self.worker.set_north)
        self.btn_reset.clicked.connect(self.worker.reset_bearing)
        self.chk_send.toggled.connect(self.worker.set_sending)

        self._x = None
        self.worker.start()
        self.add_log(f"Yardımcı istasyon başladı: {cfg['id']} → {cfg['host']}:{cfg['port']}")

    def on_data(self, d):
        # Spektrum
        if self._x is None or len(self._x) != len(d["fft_dbm"]):
            span_khz = (self.cfg.get("rate") or 2.0) * 1e3   # None/eksikse güvenli varsayılan
            self._x = np.linspace(-span_khz / 2, span_khz / 2, len(d["fft_dbm"]))
        self.curve.setData(self._x, d["fft_dbm"])

        # Büyük göstergeler (None gelirse '—' göster, çökme)
        _az = d.get("azimuth"); _pk = d.get("peak_dbm")
        self.lbl_angle.setText(f"Açı: {_az:.1f}°" if _az is not None else "Açı: —")
        self.lbl_dbm.setText(f"dBm: {_pk:.1f}" if _pk is not None else "dBm: —")

        # Kerteriz
        if d["bearing"] is not None:
            self.lbl_bearing.setText(f"KERTERİZ: {d['bearing']:.1f}°   "
                                     f"(güven %{int(d['conf'] * 100)}, {d['n']} örnek, SNR {d['snr_db']:.0f} dB)")
        # Durum
        s_sdr = "✓" if d["sdr_ok"] else "✗"
        s_enc = "✓" if d["enc_ok"] else "✗"
        s_snd = "→ gönderildi" if d["sent"] else ("açık" if self.chk_send.isChecked() else "kapalı")
        self.lbl_status.setText(f"SDR: {s_sdr}   |   Enkoder: {s_enc}   |   Merkeze gönderim: {s_snd}")

    def add_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def closeEvent(self, e):
        self.worker.stop()
        e.accept()


def main():
    ap = argparse.ArgumentParser(description="DF Yardımcı İstasyon (Pluto / N210)")
    ap.add_argument("--id", default=DEFAULTS["id"], help="Düğüm id (NODE-2/NODE-3) — merkezle aynı olmalı")
    ap.add_argument("--sdr", default=DEFAULTS["sdr"], help='SoapySDR args, ör. "driver=plutosdr" / "driver=uhd,addr=..."')
    ap.add_argument("--host", default=DEFAULTS["host"], help="Merkez bilgisayar IP")
    ap.add_argument("--port", type=int, default=DEFAULTS["port"])
    ap.add_argument("--enc", default=DEFAULTS["enc"], help="ESP32 enkoder seri portu")
    ap.add_argument("--baud", type=int, default=DEFAULTS["baud"])
    ap.add_argument("--freq", type=float, default=DEFAULTS["freq"], help="Başlangıç frekansı (MHz)")
    ap.add_argument("--rate", type=float, default=DEFAULTS["rate"], help="Örnekleme hızı (Msps)")
    ap.add_argument("--gain", type=float, default=DEFAULTS["gain"], help="RX kazancı (dB)")
    ap.add_argument("--sim", action="store_true", help="Donanımsız simülasyon (arayüz testi)")
    a = ap.parse_args()
    cfg = {"id": a.id, "sdr": a.sdr, "host": a.host, "port": a.port, "enc": a.enc, "baud": a.baud,
           "freq": a.freq, "rate": a.rate, "gain": a.gain, "rate_hz": DEFAULTS["rate_hz"], "sim": a.sim}

    app = QApplication(sys.argv)
    win = AuxWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
