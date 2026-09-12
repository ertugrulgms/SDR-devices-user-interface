#!/usr/bin/env python3
"""Günün GPS broadcast ephemeris (RINEX nav) dosyasını NASA CDDIS'ten indirir (GPS-SDR-SIM için, 5.2.4).

GEREKLİ (bir kereye mahsus): ücretsiz Earthdata hesabı -> https://urs.earthdata.nasa.gov/users/new
Kimlik ~/.netrc dosyasında olmalı:
    machine urs.earthdata.nasa.gov login KULLANICI password SIFRE
    (chmod 600 ~/.netrc)

KULLANIM:
    python tools/fetch_rinex.py                    # BUGÜNÜN dosyası
    python tools/fetch_rinex.py --date 2026-09-18  # belirli gün (yarışma günü)

İNDİRME BAŞARISIZSA: aşağıda yazılan URL'yi tarayıcıdan (Earthdata girişiyle) elle indir.
ÖNEMLİ: Fuar alanında internet güvenilmez olabilir -> bu dosyayı GİTMEDEN ÖNCE indir.
"""
import os
import sys
import gzip
import shutil
import argparse
import datetime
import subprocess

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    ap = argparse.ArgumentParser(description="Günün RINEX broadcast ephemeris'ini indir (CDDIS)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (varsayılan: bugün)")
    ap.add_argument("--out", default=None, help="çıktı .YYn yolu (varsayılan: data/)")
    a = ap.parse_args()

    d = datetime.date.today() if not a.date else datetime.date.fromisoformat(a.date)
    doy = d.timetuple().tm_yday
    yy = d.year % 100
    fname = f"brdc{doy:03d}0.{yy:02d}n"
    url = (f"https://cddis.nasa.gov/archive/gnss/data/daily/"
           f"{d.year}/{doy:03d}/{yy:02d}n/{fname}.gz")
    out = a.out or os.path.join(_DATA, fname)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    gz = out + ".gz"

    print(f"Tarih: {d}  (yılın günü {doy:03d})")
    print(f"İndiriliyor: {url}")
    cmd = ["curl", "-sS", "-L", "-n",
           "-c", "/tmp/cddis_cookies", "-b", "/tmp/cddis_cookies",
           "-o", gz, url]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("❌ 'curl' bulunamadı (sudo apt install curl).")
        return 1

    if not os.path.isfile(gz) or os.path.getsize(gz) < 1000:
        print("❌ İndirme başarısız (Earthdata girişi/~.netrc eksik olabilir).")
        print("   ELLE indir (tarayıcıda Earthdata girişi yaparak):")
        print("   " + url)
        return 1

    with gzip.open(gz, "rb") as fin, open(out, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(gz)
    print(f"✅ RINEX indirildi: {out}")
    print(f"Sonraki adım (SAHTE konumu SEN seçersin):")
    print(f"  python tools/gen_gps_spoof.py --rinex {out} --lat <sahte_enlem> --lon <sahte_boylam> --alt <m>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
