#!/usr/bin/env python3
"""GPS L1 spoofing baseband üretici (5.2.4) — GPS-SDR-SIM + RINEX ile.

RINEX broadcast ephemeris + sahte koordinattan GEÇERLİ nav mesajlı GPS L1 baseband üretir
(data/gpssim_l1.bin). Uygulamada GNSS Aldatma -> "GPS-SDR-SIM (gerçek ephemeris)" modu bunu yayınlar.

KULLANIM:
  python tools/gen_gps_spoof.py --rinex brdc2550.24n --lat 39.8901 --lon 32.7831 --alt 900
  (RINEX: o günün broadcast ephemeris dosyası. Yoksa: tools/install_gps_sdr_sim.sh çıktısındaki
   indirme notuna bak.)

⚠️ Havadan GPS yayınlamak YASAK/tehlikeli — kalkanlı ortam / kablo+zayıflatıcı kullan.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import gps_sdr_sim as g

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpssim_l1.bin")


def main():
    ap = argparse.ArgumentParser(description="GPS L1 spoofing baseband üretici (GPS-SDR-SIM)")
    ap.add_argument("--rinex", required=True, help="RINEX broadcast ephemeris (nav) dosyası")
    ap.add_argument("--lat", type=float, required=True, help="Sahte enlem (derece)")
    ap.add_argument("--lon", type=float, required=True, help="Sahte boylam (derece)")
    ap.add_argument("--alt", type=float, default=100.0, help="Sahte yükseklik (m, varsayılan 100)")
    ap.add_argument("--fs", type=float, default=g.DEFAULT_FS_HZ, help="Örnekleme (Hz, varsayılan 2.6M)")
    ap.add_argument("--dur", type=int, default=30, help="Süre (sn, döngüde tekrarlanır; varsayılan 30)")
    ap.add_argument("--time", default=None, help="Senaryo başlangıcı YYYY/MM/DD,hh:mm:ss (opsiyonel)")
    ap.add_argument("--out", default=_OUT, help="Çıktı .bin yolu")
    a = ap.parse_args()

    if not g.available():
        print("❌ " + g.install_hint())
        return 1
    print(f"GPS L1 baseband üretiliyor -> sahte konum ({a.lat}, {a.lon}, {a.alt} m), fs={a.fs/1e6:.2f} MHz...")
    r = g.generate(a.rinex, a.lat, a.lon, a.alt, a.out, fs_hz=a.fs, duration_s=a.dur, start_datetime=a.time)
    if not r["ok"]:
        print(f"❌ Üretim başarısız: {r['error']}")
        return 1
    sz = os.path.getsize(a.out) / 1e6
    print(f"✅ Üretildi: {a.out} ({sz:.1f} MB, {a.fs/1e6:.2f} Msps, 16-bit I/Q)")
    print("Uygulamada: GNSS Aldatma -> 'GPS-SDR-SIM (gerçek ephemeris)' modu ile 1575.42 MHz'de yayınla.")
    print("⚠️ KALKANLI ortam / kablo+zayıflatıcı — havadan yayınlama!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
