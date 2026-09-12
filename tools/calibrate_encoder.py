#!/usr/bin/env python3
"""
Enkoder AÇI KALİBRASYONU — mıknatıs eksantriklik/eğiklik hatasını YAZILIMLA düzeltir.

NEDEN: AS5600 mıknatısı çip merkezine tam oturmayınca açı bir turda ~sinüs şeklinde kayar
(Kuzey/Güney doğru, Doğu/Batı ~50° yanlış gibi). Mıknatısı mekanik merkezlemek zorsa, bilinen
açılarda encoder'ın NE OKUDUĞUNU kaydedip canlı açıyı interpolasyonla düzeltiriz.

BONUS: Bu tablo aynı zamanda KUZEY hizalamasını da içerir (gerçek pusula açılarını girdiğin için)
-> CMD:ZERO'ya GEREK KALMAZ. Mıknatıs montajı sabit kaldıkça yeniden başlatmada da geçerli olur.

KULLANIM:
  1) Ana uygulamayı KAPAT (port meşgul olmasın).
  2) python tools/calibrate_encoder.py            # varsayılan: her 30° (12 nokta)
     python tools/calibrate_encoder.py 15         # her 15° (24 nokta, daha hassas)
  3) Her adımda anteni PUSULAYLA istenen açıya getir, Enter'a bas.
  4) Biter; data/encoder_cal.json yazılır. Uygulamayı aç -> düzeltme otomatik aktif.

NOT: Mıknatıs/mil montajını sonradan oynatırsan tekrar kalibre et. CMD:ZERO KULLANMA
(kullanırsan tablo kayar); bu araç Kuzey'i zaten hallediyor.
"""
import os
import sys
import json
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from backend.hardware_controller import HardwareController, _ENC_CAL_PATH


def find_and_connect():
    """ANGLE verisi GERÇEKTEN gelen porta bağlan (ESP32-S3 iki CDC arayüzü -> doğrulamalı)."""
    cands = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0'] + \
            sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    seen = set()
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        hc = HardwareController(port=p)
        if hc.connect(verify_angle=True, verify_timeout=1.5):
            print(f"✅ Enkoder bağlı: {p}")
            return hc
    return None


def read_avg(hc, n=25, dt=0.04):
    """Ham encoder açısının DAİRESEL ortalaması (hafif titremeyi yumuşatır). current_angle HAM değerdir
    (get_angle() eski kalibrasyonu uygulardı -> yeni tablo için ham gerekir)."""
    rad = []
    for _ in range(n):
        rad.append(np.radians(hc.current_angle))
        time.sleep(dt)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    return float(np.degrees(np.arctan2(s, c))) % 360.0


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    if 360 % step != 0 or not (5 <= step <= 90):
        print("Adım 360'ı tam bölmeli ve 5-90 arası olmalı (ör. 15, 30, 45).")
        return
    trues = list(range(0, 360, step))

    print("=" * 62)
    print(f"  ENKODER AÇI KALİBRASYONU — {len(trues)} nokta (her {step}°)")
    print("=" * 62)
    hc = find_and_connect()
    if hc is None:
        print("❌ Enkoder bulunamadı (ANGLE verisi gelen port yok). Kabloyu/portu kontrol et.")
        return

    print("\nHer adımda anteni PUSULAYLA belirtilen açıya getir, sonra Enter'a bas.")
    print("(0° = Kuzey, 90° = Doğu, 180° = Güney, 270° = Batı)\n")
    measured = []
    try:
        for t in trues:
            input(f"  → Anteni {t:3d}°'ye getir ve Enter...")
            m = read_avg(hc)
            measured.append(round(m, 2))
            print(f"      gerçek {t:3d}° -> encoder {m:6.1f}° okudu")
    except KeyboardInterrupt:
        print("\nİptal edildi (kayıt yapılmadı).")
        hc.disconnect()
        return

    data = {
        "measured_deg": measured,
        "true_deg": trues,
        "step_deg": step,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(_ENC_CAL_PATH), exist_ok=True)
    with open(_ENC_CAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Kaydedildi: {_ENC_CAL_PATH}")

    # Düzeltmeyi yükleyip kalibrasyon noktalarında artık-hatayı göster (düğümlerde ~0 olmalı)
    hc._load_calibration()
    print("\nDoğrulama (düzeltilmiş açı, kalibrasyon noktalarında):")
    for t, m in zip(trues, measured):
        corr = hc._apply_calibration(m)
        err = ((corr - t + 180) % 360) - 180
        print(f"   gerçek {t:3d}° | encoder {m:6.1f}° | düzeltme {corr:6.1f}° | hata {err:+.1f}°")
    print("\nUygulamayı başlatınca düzeltme otomatik uygulanır. İyi çalışmalar!")
    hc.disconnect()


if __name__ == "__main__":
    main()
