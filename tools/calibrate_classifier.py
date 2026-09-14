#!/usr/bin/env python3
"""Sınıflandırıcı EŞİK KALİBRASYON aracı — GERÇEK sinyal kayıtlarıyla.

Büyük testte sinyal jeneratörü/gerçek yayınlar dinlenirken, panelin "Ham IQ Kaydet" düğmesiyle
(veya SDRWorker.dump_raw_iq) alınan .npy dosyaları bu araca verilir. Araç, sınıflandırıcının o
gerçek sinyalden çıkardığı ÖZELLİKLERİ (gamma_max, if_kurt, sigma_aa, kümülantlar, OFDM prominence,
FHSS geçmişi) ve verdiği KARARI yazdırır. Böylece eşikler (signal_classifier.py'deki TH_*)
sentetik değil GERÇEK ölçümlere göre doğrulanıp ayarlanabilir.

Kullanım:
    python tools/calibrate_classifier.py <kayit.npy> --fs 20e6 [--label QPSK]
    python tools/calibrate_classifier.py *.npy --fs 2.4e6      # toplu

Not: .npy dosyası complex64 IQ dizisi olmalıdır (dump_raw_iq bunu üretir).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.signal_classifier import SignalClassifier


def analyze(path: str, fs: float, label: str = None):
    iq = np.load(path).astype(np.complex64)
    clf = SignalClassifier(fs)
    snr = clf._estimate_snr_db(iq)
    bw = clf.estimate_bandwidth_hz(iq)
    baud = clf.estimate_symbol_rate_hz(iq)
    mux, prom, nfft = clf.detect_multiplex(iq)
    # _channelize (work_iq, iso_fs) TUPLE döner; len(tuple)=2 hep <256'ydı -> özellik boş kalıyordu (bug).
    work_iq, iso_fs = clf._channelize(iq)
    feats = clf.extract_features(work_iq, iso_fs) if len(work_iq) >= 256 else {}
    result = clf.classify(iq)

    print(f"\n=== {os.path.basename(path)}  (fs={fs/1e6:.3f} MHz, N={len(iq)}"
          + (f", etiket={label}" if label else "") + ") ===")
    print(f"  SNR≈{snr:.1f} dB | işgal BW≈{bw/1e3:.1f} kHz | baud≈{baud/1e3:.1f} kHz")
    print(f"  OFDM: {mux} (prominence={prom:.1f}, N_fft={nfft})")
    if feats:
        print(f"  gamma_max={feats['gamma_max']:.0f}  if_kurt={feats['if_kurt']:.2f}  "
              f"sigma_aa={feats['sigma_aa']:.3f}  c40={feats['c40']:.2f}  c42={feats['c42']:.2f}")
    print(f"  KARAR -> modülasyon={result['modulation']} (güven={result.get('confidence')})"
          f" | çoklama={result.get('multiplex')} | EKKT={result.get('ekkt')}")
    if label:
        ok = label.lower() in result["modulation"].lower()
        print(f"  {'✓ EŞLEŞTİ' if ok else '✗ EŞLEŞMEDİ'} (beklenen: {label})")


def main():
    ap = argparse.ArgumentParser(description="Sınıflandırıcı eşik kalibrasyon aracı")
    ap.add_argument("files", nargs="+", help=".npy IQ kayıt dosya(lar)ı (glob desteklenir)")
    ap.add_argument("--fs", type=float, required=True, help="Örnekleme hızı (Hz), ör. 20e6")
    ap.add_argument("--label", default=None, help="Beklenen sınıf (doğrulama için, ör. QPSK)")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) or [f])
    for p in paths:
        if not os.path.exists(p):
            print(f"UYARI: bulunamadı: {p}")
            continue
        try:
            analyze(p, args.fs, args.label)
        except Exception as exc:
            print(f"HATA ({p}): {exc}")


if __name__ == "__main__":
    main()
