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
  PATTERN (opsiyonel ama önerilir): data/antenna_pattern.json'u bu dosyanın yanına (veya data/ altına)
  kopyala -> pattern-eşleştirmeli DF açılır (kerteriz çok daha hassas + σ ile ağırlıklı üçgenleme).
  Dosya yoksa sistem sessizce centroid yöntemine düşer (çalışmaya devam eder).

Çalıştırma:
  # Pluto istasyonu:
  python aux_station.py --id NODE-2 --sdr "driver=plutosdr" --host 192.168.1.10 --enc /dev/ttyUSB0
  # N210 istasyonu:
  python aux_station.py --id NODE-3 --sdr "driver=uhd,addr=192.168.10.2" --host 192.168.1.10 --enc /dev/ttyUSB0
  # Donanımsız arayüz testi (sahte sinyal + sahte açı):
  python aux_station.py --id NODE-2 --sim
"""
import os
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
    "enc": "/dev/ttyACM0",          # ESP32 (AS5600 enkoder) seri portu — ESP32-S3 native USB ttyACM0
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
                             QLabel, QDoubleSpinBox, QPushButton, QTextEdit, QGridLayout, QCheckBox,
                             QLineEdit, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg


# ----------------------------------------------------------------------------- #
#  GENLİK-TABANLI KERTERİZ KESTİRİCİ (anten elle döndürülürken tepe açıyı bulur)
#  (Merkezdeki backend.direction_finding.AmplitudeDFEstimator'ın kompakt, bağımsız kopyası.)
# ----------------------------------------------------------------------------- #
# --- ÖLÇÜLEN VNA PATTERN (opsiyonel) — merkezdeki backend.antenna_pattern'ın bağımsız kompakt kopyası.
#     antenna_pattern.json aux_station.py'nin YANINDA (veya data/ altında) varsa pattern-eşleştirmeli
#     DF açılır (kerteriz + σ çok daha hassas); yoksa sessizce centroid'e düşer. σ merkeze gönderilip
#     ağırlıklı üçgenlemede kullanılır.
# Bu sabitler merkezdeki backend/antenna_pattern.py ile AYNI olmalı (parity testiyle doğrulanır).
_AUX_SIGMA_CENTROID = 8.0
_AUX_SIGMA_CEIL = 45.0
_AUX_SIGMA_FLOOR = 1.5
_AUX_MIN_FB_DB = 10.0
_AUX_MIN_SAMPLES = 8
_AUX_MIN_COVERAGE_DEG = 180.0
_AUX_MIN_AMBIG_Z = 3.0
_AUX_AMBIG_GUARD_DEG = 25.0
_AUX_FREQ_MARGIN_STEPS = 0.5


class _AuxPattern:
    """antenna_pattern.json yükleyip pattern-eşleştirme (template matching) yapar. available()=False
    ise merkezdeki mantıkla birebir aynı şekilde centroid'e düşülür. ASLA sentetik veri üretmez.
    ALGORİTMA merkezdeki backend.antenna_pattern.AntennaPattern ile birebir aynıdır (parity testi
    tests/test_direction_finding.py'de ikisini aynı girdiyle karşılaştırır)."""

    def __init__(self):
        self._ok = False
        here = os.path.dirname(os.path.abspath(__file__))
        for p in (os.path.join(here, "data", "antenna_pattern.json"),
                  os.path.join(here, "antenna_pattern.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                self._freqs = np.asarray(d["freqs_hz"], float)
                self._angles = np.asarray(d["angles_deg"], float)
                self._pat = np.asarray(d["pattern_db"], float)
                if self._pat.shape == (len(self._freqs), len(self._angles)) and len(self._freqs) >= 2:
                    self._ok = True
                    break
            except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
                continue

    def available(self):
        return self._ok

    def in_range(self, freq_hz):
        if not self._ok:
            return False
        step = float(self._freqs[1] - self._freqs[0]) if len(self._freqs) > 1 else 20e6
        m = _AUX_FREQ_MARGIN_STEPS * step
        return (self._freqs[0] - m) <= float(freq_hz) <= (self._freqs[-1] + m)

    def pattern_at(self, freq_hz):
        """Komşu iki dilim arasında GÜÇ alanında lineer interpolasyon; aralık dışı -> None."""
        if not self._ok or not self.in_range(freq_hz):
            return None
        f = float(freq_hz); fr = self._freqs
        if f <= fr[0]:
            i0 = i1 = 0; w = 0.0
        elif f >= fr[-1]:
            i0 = i1 = len(fr) - 1; w = 0.0
        else:
            i1 = int(np.searchsorted(fr, f)); i0 = i1 - 1
            w = (f - fr[i0]) / (fr[i1] - fr[i0] + 1e-12)
        lin = (1.0 - w) * np.power(10.0, self._pat[i0] / 10.0) + w * np.power(10.0, self._pat[i1] / 10.0)
        db = 10.0 * np.log10(lin + 1e-12); db = db - db.max()
        ang = self._angles; ipk = int(np.argmax(db)); pk = float(ang[ipk])
        back = (pk + 180.0) % 360.0
        iback = int(np.argmin(np.abs(((ang - back + 180.0) % 360.0) - 180.0)))
        return ang, db, float(db[ipk] - db[iback]), pk

    @staticmethod
    def coverage_deg(az):
        a = np.sort(np.asarray(az, float) % 360.0)
        if len(a) < 2:
            return 0.0
        return float(max(0.0, 360.0 - max(float(np.diff(a).max()), (a[0] + 360.0) - a[-1])))

    def match(self, az_deg, power_db, freq_hz):
        """(az, güç) örneklerini frekansın kalibre pattern'ine oturtur -> {bearing_deg, sigma_deg,
        quality_ok, coverage_deg, front_back_db, ambiguity_z, ...} veya None. Merkezle aynı algoritma:
        interpolasyon + aralık kapısı + açısal kapsama + ambiguity + parabolik σ."""
        if not self._ok:
            return None
        az = np.asarray(az_deg, float) % 360.0
        pm = np.asarray(power_db, float)
        n = len(az)
        if n < _AUX_MIN_SAMPLES or n != len(pm):
            return None
        got = self.pattern_at(freq_hz)
        if got is None:
            return None
        ang, pref, fb, pk = got
        cov = self.coverage_deg(az)
        pm = pm - np.max(pm)
        ae = np.concatenate([ang - 360.0, ang, ang + 360.0])
        pe = np.concatenate([pref, pref, pref])

        def pref_bore(x):
            return np.interp((np.asarray(x) + pk) % 360.0, ae, pe)

        betas = np.arange(0.0, 360.0, 1.0)
        cost = np.array([float(np.dot(pm - pref_bore(b - az), pm - pref_bore(b - az))) for b in betas])
        kmin = int(np.argmin(cost)); beta = float(betas[kmin]); cmin = float(cost[kmin])
        guard = int(round(_AUX_AMBIG_GUARD_DEG))
        offs = np.abs(((np.arange(len(betas)) - kmin + len(betas) // 2) % len(betas)) - len(betas) // 2)
        second = float(np.min(cost[offs > guard])) if np.any(offs > guard) else float(cost.max())
        ambiguity_z = (second / max(cmin, 1e-9) - 1.0) * np.sqrt(n / 2.0)
        c0, c1, c2 = cost[(kmin - 1) % len(betas)], cost[kmin], cost[(kmin + 1) % len(betas)]
        a = 0.5 * (c0 + c2 - 2.0 * c1)
        if a > 1e-9:
            beta = (beta + 0.5 * (c0 - c2) / (c0 - 2.0 * c1 + c2)) % 360.0
            sigma = float(np.sqrt(max(cmin, 1e-9) / max(n - 1, 1) / a))
        else:
            sigma = _AUX_SIGMA_CEIL
        sigma = float(np.clip(sigma, _AUX_SIGMA_FLOOR, _AUX_SIGMA_CEIL))
        quality_ok = ((fb >= _AUX_MIN_FB_DB) and (cov >= _AUX_MIN_COVERAGE_DEG)
                      and (ambiguity_z >= _AUX_MIN_AMBIG_Z) and (a > 1e-9)
                      and (sigma < _AUX_SIGMA_CEIL))
        return {"bearing_deg": round(beta, 2), "sigma_deg": round(sigma, 2),
                "quality_ok": bool(quality_ok), "n": n, "coverage_deg": round(cov, 1),
                "front_back_db": round(fb, 1), "ambiguity_z": round(ambiguity_z, 2)}


_AUX_PATTERN = None


def _aux_pattern():
    global _AUX_PATTERN
    if _AUX_PATTERN is None:
        _AUX_PATTERN = _AuxPattern()
    return _AUX_PATTERN


class AmplitudeDF:
    """(açı, güç) örneklerinden en yüksek gücün alındığı azimutu (kerteriz) verir.
    Tepe etrafında güç-ağırlıklı dairesel merkez (hassasiyet ANTEN HÜZME GENİŞLİĞİYLE sınırlıdır;
    LPDA'da tipik ±birkaç derece — "alt-derece" değildir). Örnekler zamanla söner
    (yeniden tarama / kaynak değişimi). Ölçülen VNA pattern'i varsa kerteriz pattern-eşleştirmeyle
    RAFİNE edilir + belirsizlik (σ) üretilir (merkeze gönderilip ağırlıklı üçgenlemede kullanılır)."""

    def __init__(self, bin_deg=1.0, window_deg=25.0, decay_sec=15.0, freq_hz=None, use_pattern=True):
        self.bin_deg = bin_deg
        self.window_deg = window_deg
        self.decay_sec = decay_sec
        self._bins = {}   # az_bin -> (amp_dbm, ts)
        self.freq_hz = freq_hz
        self._pattern = _aux_pattern() if use_pattern and _aux_pattern().available() else None
        self._last_sigma = _AUX_SIGMA_CENTROID
        self._last_method = "centroid"

    def reset(self):
        self._bins.clear()

    def set_freq(self, freq_hz):
        self.freq_hz = float(freq_hz) if freq_hz else None

    def last_sigma(self):
        return self._last_sigma

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
        """(azimut|None, tepe_dBm, güven[0..1], örnek_sayısı). σ ayrıca last_sigma() ile alınır."""
        now = time.time() if now is None else now
        self._prune(now)
        self._last_sigma = _AUX_SIGMA_CENTROID
        self._last_method = "centroid"
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
        # PATTERN EŞLEŞTİRME ile RAFİNE (merkezle aynı): frekans + pattern varsa tüm eğriyi oturt.
        # KALİTE KAPISI (gerçek fallback): pattern yalnızca quality_ok ise kullanılır; aksi halde
        # centroid kerterizi korunur (bearing atılmaz-sadece-σ-şişir DEĞİL).
        if self._pattern is not None and self.freq_hz:
            m = self._pattern.match(azs, amps, self.freq_hz)
            if m is not None:
                if m["quality_ok"]:
                    bearing = m["bearing_deg"]
                    self._last_sigma = m["sigma_deg"]
                    self._last_method = "pattern"
                else:
                    self._last_sigma = min(_AUX_SIGMA_CEIL, _AUX_SIGMA_CENTROID * 1.5)
                    self._last_method = "centroid(pattern-düşük-kalite)"
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
        # OTOMATİK PORT — SADECE 'ANGLE' verisi GELEN porta bağlan. ESP32-S3 native USB iki arayüz
        # oluşturur (biri USB-Serial/JTAG, biri CDC); ANGLE çıkışı bunlardan yalnızca BİRİNDEDİR.
        # Eskiden "açılan ilk porta" bağlanıyordu -> yanlış (sessiz) porta düşüp açı gelmiyordu.
        # Her adayı açıp ~0.8s dinliyoruz; ANGLE gören porta bağlanır, gelmiyorsa sonrakini dener.
        import glob as _glob
        cands = [self.port, '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyUSB0'] + \
                sorted(_glob.glob('/dev/ttyACM*') + _glob.glob('/dev/ttyUSB*'))
        seen = set()
        for p in cands:
            if not p or p in seen:
                continue
            seen.add(p)
            try:
                conn = _pyserial.Serial(p, self.baud, timeout=0.3)
            except Exception:
                continue
            got = False
            t0 = time.time()
            while time.time() - t0 < 0.8:          # bu portta ANGLE geliyor mu?
                try:
                    line = conn.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    break
                if line.startswith("ANGLE:"):
                    got = True
                    break
            if got:
                self.conn = conn
                self.port = p
                self.connected = True
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
                return True
            try:
                conn.close()                        # ANGLE yok -> kapat, sonraki portu dene
            except Exception:
                pass
        self.connected = False
        return False

    def _loop(self):
        while self._running and self.connected:
            try:
                # TAMPONU BOŞALT: biriken TÜM satırları oku, yalnızca EN SON ANGLE'ı kullan. Eskiden
                # her turda tek satır okunuyordu; firmware okuyandan hızlı yollarsa seri tampon birikip
                # açı 10-15 sn GERİDEN gelirdi. Boşaltınca raw_angle daima güncel kalır (gecikme ~0).
                latest = None
                while self.conn.in_waiting > 0:
                    line = self.conn.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("ANGLE:"):
                        latest = line
                if latest is not None:
                    try:
                        self.raw_angle = float(latest.split(":")[1])
                    except (ValueError, IndexError):
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
        self.df = AmplitudeDF(freq_hz=self.freq_hz)   # pattern eşleştirme frekansı
        self.enc = EncoderReader(cfg["enc"], cfg["baud"])
        self.sdr = make_sdr(cfg)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_send = 0.0
        self._last_bearing = None        # son BULUNAN kerteriz (sabit dururken göndermeye devam)

    # --- arayüzden gelen kontroller ---
    def set_frequency_mhz(self, mhz):
        self.freq_hz = mhz * 1e6
        self.sdr.set_frequency(self.freq_hz)
        self.df.set_freq(self.freq_hz)  # pattern eşleştirme frekansı güncellensin
        self.df.reset()                 # frekans değişti -> eski açı-güç haritası geçersiz
        self._last_bearing = None       # önbellekli kerterizi de temizle (yeni frekans = yeni hedef)
        self.log.emit(f"Frekans -> {mhz:.4f} MHz (kerteriz sıfırlandı)")

    def set_gain(self, g):
        self.gain_db = g
        self.sdr.set_gain(g)

    def set_north(self):
        self.enc.set_north()
        self.log.emit("Kuzey ayarlandı (mevcut açı = 0°).")

    def reset_bearing(self):
        self.df.reset()
        self._last_bearing = None       # önbellekli kerterizi de temizle
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

            # 2) Spektrum (dBm) + HEDEF gücü
            n = len(iq)
            w = win if n == FFT_POINTS else np.hanning(n)
            fc = np.fft.fftshift(np.fft.fft((iq - np.mean(iq)) * w, n=FFT_POINTS))
            fft_dbm = 10.0 * np.log10(np.abs(fc) ** 2 / FFT_POINTS + 1e-12) + self.cal_offset_db
            # HEDEF gücü: MERKEZ ±80 kHz'te DC-HARİÇ en güçlü tepe. Global tepe DEĞİL (uzak bir
            # interferer'a kilitlenip kerteriz bir bölgede TAKILMASIN — E-S arası sabit kalma sorunu);
            # tam DC de DEĞİL (DC dikeni + mean-removal gücü düşürür -> yanlış ~20 dBm). Hedef, tune
            # edilen merkeze yakındır; onun AÇISAL tepkisi ölçülür -> anten dönünce güç gerçekten değişir
            # -> kerteriz kaynağın yönüne oturur ve orada kalır. Tepe ±2 bin entegre (tek-bin gürültüsü).
            c = FFT_POINTS // 2
            bin_hz = (self.cfg["rate"] * 1e6) / FFT_POINTS
            wbin = int(np.clip(80e3 / bin_hz, 8, FFT_POINTS // 4))
            band = np.array(fft_dbm[c - wbin:c + wbin + 1], dtype=float)
            # DC bastırma KALDIRILDI (P0-04): merkeze denk gelen gerçek HEDEF artık silinmiyor. FFT
            # zaten (iq - mean) ile hesaplandığından LO dikeni bastırılmış durumda; ayrıca "present"
            # (SNR) kapısı zayıf DC dikenini eler. Yine de en sağlamı: aux'u hedeften ~50-100 kHz
            # OFFSET tune etmek -> hedef DC'ye hiç oturmaz. Merkez ±80 kHz'te en güçlü tepe alınır.
            pk = int(np.argmax(band))
            lo_i, hi_i = max(0, pk - 2), min(len(band), pk + 3)
            seg = np.power(10.0, band[lo_i:hi_i] / 10.0)
            peak_dbm = float(10.0 * np.log10(np.mean(seg) + 1e-12))
            noise = float(np.median(fft_dbm))
            snr_db = peak_dbm - noise

            # 3) Genlik-DF: yalnızca sinyal varken (gürültü açı haritasını kirletmesin)
            if snr_db >= 6.0:
                self.df.update(az, peak_dbm)
            bearing, bpk, conf, ncnt = self.df.bearing()

            # SON KERTERİZİ ÖNBELLEKLE: genlik-DF ≥3 taze açı-bin'i ister; anten SABİT dururken bin'ler
            # söner (<3) -> bearing None olur. O anda gönderim durursa ana cihaz düğümü "koptu" (kırmızı)
            # sanır — oysa düğüm CANLI, yalnızca yeni kerteriz yok. Son bulunan kerterizi tutup
            # göndermeye devam et -> bağlantı YEŞİL kalır, kerteriz kalıcı olur (sabit istasyon/hedef).
            # Yeni tarama (anteni çevirince) ya da frekans/reset ile önbellek güncellenir/temizlenir.
            if bearing is not None:
                self._last_bearing = (bearing, bpk, conf, ncnt, self.df.last_sigma())
            tx = self._last_bearing

            # 4) Merkeze gönder (JSON/UDP) — CANLI AÇI HER ZAMAN (kerteriz olmasa da), hız sınırlı.
            #    Böylece ana cihaz aux antenin O ANKİ yönünü ~100 ms'de görür — kerterizin 15 sn'lik
            #    ortalamasını BEKLEMEZ (canlı açı gecikmesi bu yüzden vardı). azimuth_deg (kerteriz)
            #    yalnızca VARSA eklenir (füzyon/üçgenleme için); yoksa paket yine gider (düğüm canlı kalır).
            sent = False
            now = time.time()
            _rate_hz = self.cfg.get("rate_hz") or 10.0     # None/eksikse güvenli varsayılan (çökme yok)
            if self.sending and (now - self._last_send) >= (1.0 / _rate_hz):
                msg = {
                    "id": self.cfg["id"],
                    "live_angle_deg": round(az, 1),        # CANLI enkoder açısı (antenin anlık yönü)
                    "amp_dbm": round(peak_dbm, 1),
                    "snr_db": round(snr_db, 1),
                    "freq_mhz": round(self.freq_hz / 1e6, 4),
                }
                if tx is not None:                          # kerteriz bulunduysa ekle
                    msg["azimuth_deg"] = round(tx[0], 2)
                    msg["elevation_deg"] = 0.0
                    if len(tx) > 4:                          # pattern σ (ağırlıklı üçgenleme için)
                        msg["sigma_deg"] = round(tx[4], 2)
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
class ChatListener(QThread):
    """SAHA SOHBETİ dinleyici (UDP 5006). Merkez hub'dan gelen mesajları arayüze iletir."""
    received = pyqtSignal(dict)

    def __init__(self, port=5006):
        super().__init__()
        self.port = port
        self._running = False
        self.sock = None

    def run(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", self.port))
            self.sock.settimeout(1.0)
            while self._running:
                try:
                    data, _addr = self.sock.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") == "chat" and "text" in msg:
                        self.received.emit({"from": str(msg.get("from", "?")),
                                            "text": str(msg["text"])[:500],
                                            "ts": float(msg.get("ts", time.time()))})
                except socket.timeout:
                    continue
                except Exception:
                    continue
        except OSError:
            pass
        finally:
            if self.sock:
                self.sock.close()

    def stop(self):
        self._running = False
        self.wait(1500)


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
        self.lbl_dbm = QLabel("dBFS: --")   # kalibrasyonsuz BAĞIL güç (gerçek dBm değil); DF için
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

        # --- Log + SAHA SOHBETİ (yan yana; sohbet boş alana yerleşir) ---
        bottom = QHBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        bottom.addWidget(self.log, stretch=1)

        chat_box = QGroupBox("SAHA SOHBETİ (Merkez ↔ Aux)")
        chat_box.setStyleSheet("QGroupBox{border:2px solid #333;border-radius:8px;margin-top:8px;"
                               "font-size:14px;font-weight:bold;color:#00e5ff;}"
                               "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top center;padding:0 5px;}")
        cv = QVBoxLayout(chat_box)
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setStyleSheet("background:#141414;color:#e0e0e0;font-family:monospace;font-size:12px;")
        cv.addWidget(self.chat_log, stretch=1)
        crow = QHBoxLayout()
        self.chat_inp = QLineEdit()
        self.chat_inp.setPlaceholderText("Merkeze mesaj… (Enter)")
        self.chat_inp.setStyleSheet("background:#2a2a2a;color:#fff;padding:6px;border:1px solid #555;")
        self.chat_inp.returnPressed.connect(self._send_chat)
        crow.addWidget(self.chat_inp, stretch=1)
        btn_chat = QPushButton("Gönder")
        btn_chat.setStyleSheet("background:#00838f;color:#fff;font-weight:bold;padding:6px 12px;")
        btn_chat.clicked.connect(self._send_chat)
        crow.addWidget(btn_chat)
        cv.addLayout(crow)
        bottom.addWidget(chat_box, stretch=1)
        root.addLayout(bottom, stretch=1)

        # SAHA SOHBETİ ağı: merkeze (host:5006) yolla, 5006'da dinle (merkez hub dağıtır).
        self.chat_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.chat_listener = ChatListener(port=5006)
        self.chat_listener.received.connect(self._on_chat)
        self.chat_listener.start()

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
        self.lbl_dbm.setText(f"dBFS: {_pk:.1f}" if _pk is not None else "dBFS: —")

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

    def _send_chat(self):
        text = self.chat_inp.text().strip()[:500]
        if not text:
            return
        msg = {"type": "chat", "from": self.cfg["id"], "text": text, "ts": time.time()}
        try:
            self.chat_send_sock.sendto(json.dumps(msg).encode("utf-8"), (self.cfg["host"], 5006))
        except OSError as e:
            self.add_log(f"⚠️ Sohbet gönderilemedi: {e}")
        self._append_chat(self.cfg["id"], text, msg["ts"])   # kendi mesajını göster
        self.chat_inp.clear()

    def _on_chat(self, d):
        self._append_chat(d.get("from", "?"), d.get("text", ""), d.get("ts"))

    def _append_chat(self, sender, text, ts=None):
        tstr = time.strftime("%H:%M:%S", time.localtime(ts or time.time()))
        color = "#00e5ff" if sender == "MERKEZ" else "#00e676"
        safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.chat_log.append(f'<span style="color:#777">[{tstr}]</span> '
                             f'<b style="color:{color}">{sender}:</b> {safe}')
        sb = self.chat_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, e):
        self.worker.stop()
        try:
            self.chat_listener.stop()
        except Exception:
            pass
        e.accept()


def _auto_node_id():
    """Kendi 192.168.1.x IP'sinden düğüm id'sini türet: .20 -> NODE-2, .30 -> NODE-3.
    Böylece her yardımcı bilgisayar --id yazmadan sadece `python aux_station.py` ile çalışır
    (statik IP'ler: ana=.10, NODE-2=.20, NODE-3=.30). BULAMAZSA None döner -> main() AÇIKÇA HATA verir.
    (Eskiden NODE-2'ye düşüyordu; iki aux da beklenmedik IP alırsa İKİSİ DE NODE-2 olup çakışıyordu — P2-03.)"""
    ips = set()
    try:  # merkez IP'ye UDP 'connect' -> paket GÖNDERMEZ, yalnızca yerel arayüzü/IP'yi seçer
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.1.10", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:  # hostname'e bağlı tüm IPv4 adresleri
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    for ip in ips:
        if ip.startswith("192.168.1."):
            last = ip.rsplit(".", 1)[-1]
            if last == "20":
                return "NODE-2"
            if last == "30":
                return "NODE-3"
    return None   # eşleşme yok -> çakışma riski; sessizce NODE-2'ye DÜŞME, main() hata versin


def main():
    ap = argparse.ArgumentParser(description="DF Yardımcı İstasyon (Pluto / N210)")
    # --id verilmezse IP'den otomatik türetilir (.20->NODE-2, .30->NODE-3).
    ap.add_argument("--id", default=None, help="Düğüm id (verilmezse IP'den otomatik: .20=NODE-2, .30=NODE-3)")
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
    node_id = a.id if a.id else _auto_node_id()     # --id verilmediyse IP'den türet
    if not node_id:                                  # IP eşleşmedi + --id yok -> ÇAKIŞMA riski, dur
        sys.exit("HATA: Düğüm id belirlenemedi (IP 192.168.1.20/.30 değil). Çakışmayı önlemek için "
                 "--id NODE-2 (veya NODE-3) ile açıkça belirtin. Sessizce NODE-2'ye düşülmez.")
    cfg = {"id": node_id, "sdr": a.sdr, "host": a.host, "port": a.port, "enc": a.enc, "baud": a.baud,
           "freq": a.freq, "rate": a.rate, "gain": a.gain, "rate_hz": DEFAULTS["rate_hz"], "sim": a.sim}

    app = QApplication(sys.argv)
    win = AuxWindow(cfg)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
