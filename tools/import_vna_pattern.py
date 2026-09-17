#!/usr/bin/env python3
"""VNA .s2p ANTEN PATTERN ölçümlerini DF için tek JSON'a çevirir (spec 5.1.4 — Derece RMS iyileştirme).

Sahada anten ELLE döndürülürken ölçülen güç-açı eğrisi, burada üretilen KALİBRE pattern ile
eşleştirilerek geliş açısı (bearing) + belirsizlik (sigma) tahmin edilir (PatternMatchedDFEstimator).

GİRDİ: her açı için bir .s2p dosyası (dosya adında açı: "45 derece.s2p"). Touchstone 2-port,
       Re/Im, S21 (sütun 4-5) = sabit antenden dönen antene iletim = anten pattern'i.
ÇIKTI: data/antenna_pattern.json
       { meta, freqs_hz[], angles_deg[](5° grid), pattern_db[freq][angle](tepe=0 normalize),
         quality: { front_back_db[], peak_angle_deg[] } }

Frekans ~20 MHz'e seyreltilir (±5 MHz bant-ortalama ile gürültü azaltılır); açı 5° grid'e dairesel
interpole edilir. Her frekans KENDİ tepesine normalize (mutlak gain'e bağımsız — pattern ŞEKLİ yeter).

KULLANIM:
  python tools/import_vna_pattern.py --dir "/media/.../asıl ölçümler"
  (varsayılan çıktı: data/antenna_pattern.json)
"""
import os
import re
import sys
import json
import glob
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUT = os.path.join(_ROOT, "data", "antenna_pattern.json")


_FREQ_UNIT = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def _parse_option_line(line):
    """Touchstone opsiyon satırı '# <freq_unit> <par> <fmt> R <Z0>' -> (freq_mult, fmt).
    Eksik alanlar Touchstone varsayılanlarına düşer (GHz, S, MA, 50). Desteklenmeyen -> hata."""
    toks = line[1:].upper().split()
    freq_mult, fmt = 1e9, "MA"                    # Touchstone varsayılanı
    for t in toks:
        if t in _FREQ_UNIT:
            freq_mult = _FREQ_UNIT[t]
        elif t in ("RI", "MA", "DB"):
            fmt = t
        elif t in ("S", "Y", "Z", "H", "G", "R"):
            if t != "S" and t != "R":
                raise ValueError(f"desteklenmeyen parametre tipi '{t}' (yalnızca S)")
    return freq_mult, fmt


def _to_complex(a, b, fmt):
    """İki sütunu (fmt'e göre) karmaşık S-parametresine çevir: RI=re/im, MA=mag/açı°, DB=dB/açı°."""
    if fmt == "RI":
        return complex(a, b)
    if fmt == "MA":
        return a * np.exp(1j * np.radians(b))
    if fmt == "DB":
        return (10.0 ** (a / 20.0)) * np.exp(1j * np.radians(b))
    raise ValueError(f"bilinmeyen format {fmt}")


def load_s2p_s21(path):
    """Touchstone .s2p'den (freq_hz, S21_complex) döner. Opsiyon satırı (# ...) GERÇEKTEN parse edilir:
    frekans birimi (Hz/kHz/MHz/GHz) ve veri formatı (RI/MA/DB) doğrulanır -> sessiz birim hatası olmaz
    (uzman P1). 2-port sütun düzeni: freq, S11, S21, S12, S22 (her biri iki sütun; S21 = 2. çift)."""
    freq_mult, fmt = 1e9, "MA"                    # opsiyon satırı yoksa Touchstone varsayılanı
    saw_option = False
    freqs, s21 = [], []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        ln = ln.strip()
        if not ln or ln.startswith("!"):
            continue
        if ln.startswith("#"):
            freq_mult, fmt = _parse_option_line(ln)
            saw_option = True
            continue
        p = ln.split()
        if len(p) < 5:
            continue
        try:
            freqs.append(float(p[0]) * freq_mult)
            s21.append(_to_complex(float(p[3]), float(p[4]), fmt))   # S21 = 2. çift (sütun 3-4)
        except ValueError:
            continue
    if not saw_option:
        print(f"  ⚠️ {os.path.basename(path)}: opsiyon satırı (#) yok -> Touchstone varsayılanı (GHz, MA)")
    return np.array(freqs), np.array(s21)


def _angle_from_name(fn):
    """Dosya adından açıyı çıkar. Önce '<sayı> derece/deg' kalıbını dener (frekans gibi başka
    sayılara takılmasın); bulamazsa ilk tam sayıya düşer (uzman P1 dayanıklılık)."""
    base = os.path.basename(fn)
    m = re.search(r"(\d+)\s*(?:derece|deg|°)", base, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+)", base)
    return int(m.group(1)) % 360 if m else None


def circular_interp(target_deg, meas_deg, meas_lin):
    """Ölçülen (düzensiz) açılardaki DOĞRUSAL güçleri hedef açı grid'ine DAİRESEL interpole eder."""
    order = np.argsort(meas_deg)
    md = meas_deg[order]
    ml = meas_lin[order]
    # 360° sarma için başa/sona kopya ekle
    md_ext = np.concatenate([md - 360.0, md, md + 360.0])
    ml_ext = np.concatenate([ml, ml, ml])
    return np.interp(target_deg % 360.0, md_ext, ml_ext)


def main():
    ap = argparse.ArgumentParser(description="VNA .s2p anten pattern -> DF JSON")
    ap.add_argument("--dir", required=True, help="Açı .s2p dosyalarının klasörü")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="Çıktı JSON yolu")
    ap.add_argument("--freq-step-mhz", type=float, default=20.0, help="Frekans grid adımı (MHz)")
    ap.add_argument("--angle-step-deg", type=float, default=5.0, help="Açı grid adımı (derece)")
    ap.add_argument("--band-avg-mhz", type=float, default=5.0, help="Her frekansta ±bant-ortalama (MHz)")
    a = ap.parse_args()

    files = []
    for fn in glob.glob(os.path.join(a.dir, "*.s2p")):
        ang = _angle_from_name(fn)
        if ang is not None:
            files.append((ang, fn))
    files.sort()
    if len(files) < 8:
        sys.exit(f"HATA: {a.dir} içinde yeterli .s2p yok ({len(files)} bulundu).")
    print(f"{len(files)} açı dosyası: {[ang for ang, _ in files]}")

    meas_angles = np.array([ang for ang, _ in files], dtype=float)
    f_axis, _ = load_s2p_s21(files[0][1])
    # [açı, frekans] karmaşık S21
    S = np.array([load_s2p_s21(fn)[1] for _, fn in files])

    # Hedef gridler
    fmin, fmax = float(f_axis[0]), float(f_axis[-1])
    freqs = np.arange(fmin, fmax + 1.0, a.freq_step_mhz * 1e6)
    angles = np.arange(0.0, 360.0, a.angle_step_deg)

    pattern_db = []
    front_back = []
    peak_angle = []
    ba = a.band_avg_mhz * 1e6
    for fc in freqs:
        mask = (f_axis >= fc - ba) & (f_axis <= fc + ba)
        if not np.any(mask):
            mask = np.array([int(np.argmin(np.abs(f_axis - fc)))])
        # GÜÇ alanında ortalama (uzman P1): pattern alınan GÜCÜ (|S21|²) temsil eder; ±bant gürültü
        # azaltması güç ekleyerek yapılmalı -> mean(|S21|²), sonra 10·log10. (20·log10(mean|S21|) ile
        # aynı DEĞİLDİR; güç alanı fiziksel olarak doğru olan.)
        pwr = np.mean(np.abs(S[:, mask]) ** 2, axis=1)      # açı başına ortalama GÜÇ (doğrusal)
        grid_lin = circular_interp(angles, meas_angles, pwr)   # güç interpolasyonu
        db = 10.0 * np.log10(grid_lin + 1e-24)
        db = db - db.max()                                  # tepe = 0 dB (normalize)
        ipk = int(np.argmax(db))
        pk_ang = float(angles[ipk])
        back_ang = (pk_ang + 180.0) % 360.0
        # back_ang'e en yakın grid açısını dairesel farkla bul
        iback = int(np.argmin(np.abs(((angles - back_ang + 180.0) % 360.0) - 180.0)))
        fb = float(db[ipk] - db[iback])
        pattern_db.append([round(float(v), 2) for v in db])
        front_back.append(round(fb, 1))
        peak_angle.append(pk_ang)

    out = {
        "meta": {
            # Mutlak yerel yol SIZDIRILMAZ (uzman P2): yalnızca klasör adı + dosya sayısı bilgi olarak.
            "source": os.path.basename(os.path.normpath(a.dir)),
            "n_files": len(files),
            "avg_domain": "power(|S21|^2)",
            "n_angles": len(angles),
            "angle_step_deg": a.angle_step_deg,
            "n_freq": len(freqs),
            "freq_min_hz": float(freqs[0]),
            "freq_max_hz": float(freqs[-1]),
            "freq_step_hz": a.freq_step_mhz * 1e6,
            "note": "pattern_db: her frekans KENDİ tepesine normalize (0 dB=tepe). Mekanik açı çerçevesi.",
        },
        "freqs_hz": [float(x) for x in freqs],
        "angles_deg": [float(x) for x in angles],
        "pattern_db": pattern_db,
        "quality": {
            "front_back_db": front_back,
            "peak_angle_deg": peak_angle,
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    good = int(np.mean(np.array(front_back) >= 10.0) * 100)
    sz = os.path.getsize(a.out) / 1024
    print(f"✅ Yazıldı: {a.out} ({sz:.0f} KB)")
    print(f"   {len(freqs)} frekans × {len(angles)} açı | ön/arka≥10dB olan frekans: %{good}")
    print(f"   frekans aralığı: {freqs[0]/1e6:.0f}–{freqs[-1]/1e6:.0f} MHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
