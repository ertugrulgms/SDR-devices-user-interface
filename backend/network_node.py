#!/usr/bin/env python3
"""Uzak DF DÜĞÜM GÖNDERİCİSİ (test/saha) — ana cihaza JSON kerteriz (LOB) yayınlar.

Sahada her uzak düğüm (PlutoSDR + bilgisayar) bunun BENZERİNİ çalıştırır: yönlü anten elle
döndürülürken sensör en yüksek genliğin geldiği azimutu ölçer ve ana cihaza UDP/JSON gönderir.
Bu script test/gösterim amaçlıdır: gerçek bir hedef konumu verilirse, düğümün konumundan hedefe
GERÇEK azimut/elevation'ı hesaplayıp (küçük gürültüyle) yayınlar — böylece ana cihazın füzyon
+ PPI + rapor zinciri donanımsız uçtan uca test edilebilir.

YENİ ŞEMA (ana cihazın NodeBearingStore'unun beklediği):
    {"id","azimuth_deg","elevation_deg","amp_dbm","snr_db","freq_mhz"}

Kullanım:
    # Ana cihazda uygulamayı aç (UDP 5005 dinler). Sonra her düğüm için:
    python backend/network_node.py --id NODE-2 --pos 500,0,0 --target 300,200,120 --freq 433.0
    python backend/network_node.py --id NODE-3 --pos 250,433,0 --target 300,200,120 --freq 433.0
    # Sabit açı yaymak için (gerçek sahada sensörün ölçtüğü açı elle verilebilir):
    python backend/network_node.py --id NODE-2 --azimuth 315 --elevation 23

Not: id ve pos, ana cihazdaki data/df_nodes.json ile TUTARLI olmalıdır (aksi halde konumsuz
kalır ve yok sayılır).
"""
import argparse
import json
import math
import socket
import time


def bearing_to(pos, target):
    """pos'tan target'a azimut (Kuzey'den saat yönü, derece) + elevation (derece)."""
    dx = target[0] - pos[0]      # Doğu
    dy = target[1] - pos[1]      # Kuzey
    dz = target[2] - pos[2]      # Yukarı
    az = math.degrees(math.atan2(dx, dy)) % 360.0
    ground = math.hypot(dx, dy)
    el = math.degrees(math.atan2(dz, ground)) if ground > 1e-9 else 0.0
    return az, el


def _parse_vec(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description="DF düğüm kerteriz göndericisi (yeni şema)")
    ap.add_argument("--id", required=True, help="Düğüm id (df_nodes.json ile aynı, ör. NODE-2)")
    ap.add_argument("--host", default="127.0.0.1", help="Ana cihaz IP (varsayılan localhost)")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--freq", type=float, default=433.0, help="Frekans (MHz)")
    ap.add_argument("--rate", type=float, default=10.0, help="Saniyedeki paket sayısı")
    # Ya (pos + target) ile gerçek azimut hesapla, ya da sabit azimut/elevation ver:
    ap.add_argument("--pos", type=_parse_vec, default=None, help="Düğüm konumu x,y,z (m)")
    ap.add_argument("--target", type=_parse_vec, default=None, help="Hedef konumu x,y,z (m)")
    ap.add_argument("--azimuth", type=float, default=None, help="Sabit azimut (derece)")
    ap.add_argument("--elevation", type=float, default=0.0, help="Sabit elevation (derece)")
    ap.add_argument("--noise", type=float, default=1.0, help="Azimut ölçüm gürültüsü (derece RMS)")
    args = ap.parse_args()

    if args.azimuth is None and (args.pos is None or args.target is None):
        ap.error("Ya --azimuth [--elevation] ver, ya da --pos ve --target birlikte ver.")

    import random
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[{args.id}] -> {args.host}:{args.port} JSON kerteriz yayını başladı (Ctrl+C ile dur).")
    try:
        while True:
            if args.azimuth is not None:
                az, el = args.azimuth, args.elevation
            else:
                az, el = bearing_to(args.pos, args.target)
            az_n = (az + random.gauss(0.0, args.noise)) % 360.0     # ölçüm gürültüsü
            msg = {
                "id": args.id,
                "azimuth_deg": round(az_n, 2),
                "elevation_deg": round(el, 2),
                "amp_dbm": round(-55.0 + random.uniform(-2, 2), 1),
                "snr_db": round(18.0 + random.uniform(-3, 3), 1),
                "freq_mhz": args.freq,
            }
            sock.sendto(json.dumps(msg).encode("utf-8"), (args.host, args.port))
            time.sleep(1.0 / max(0.1, args.rate))
    except KeyboardInterrupt:
        print(f"\n[{args.id}] durduruldu.")


if __name__ == "__main__":
    main()
