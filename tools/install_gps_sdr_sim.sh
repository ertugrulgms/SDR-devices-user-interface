#!/usr/bin/env bash
# =============================================================================
#  GPS-SDR-SIM KURULUMU — gerçek ephemeris'li GPS L1 spoofing baseband üretici (5.2.4)
# =============================================================================
#  GPS-SDR-SIM (osqzss): RINEX broadcast ephemeris + sahte koordinattan GEÇERLİ nav mesajlı
#  (subframe'li) tam GPS L1 C/A baseband üretir -> gerçek alıcı sahte konuma kilitlenir.
#  Tek dosya C programı; sudo GEREKMEZ (kullanıcı diznine derlenir).
#
#  KULLANIM:  bash tools/install_gps_sdr_sim.sh
#  Bitince:   RINEX indir (aşağıda) + tools/gen_gps_spoof.py ile baseband üret + uygulamada yayınla.
# =============================================================================
set -euo pipefail

BUILD_DIR="${HOME}/gps_sdr_sim_build"
BIN_DIR="${HOME}/.local/bin"
mkdir -p "$BIN_DIR"

echo ">>> GPS-SDR-SIM indiriliyor + derleniyor..."
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
if [ ! -d gps-sdr-sim ]; then
    git clone --depth 1 https://github.com/osqzss/gps-sdr-sim.git
fi
cd gps-sdr-sim && git pull --ff-only || true
# USER_MOTION_SIZE: dinamik senaryo max nokta (statik spoof için önemsiz ama derleme bayrağı)
gcc gpssim.c -lm -O3 -o gps-sdr-sim -DUSER_MOTION_SIZE=4000
cp -f gps-sdr-sim "$BIN_DIR/gps-sdr-sim"

echo ""
if command -v gps-sdr-sim >/dev/null 2>&1 || [ -x "$BIN_DIR/gps-sdr-sim" ]; then
    echo "✅ BAŞARILI: gps-sdr-sim -> $BIN_DIR/gps-sdr-sim"
    "$BIN_DIR/gps-sdr-sim" 2>&1 | head -3 || true
else
    echo "⚠️  gps-sdr-sim bulunamadı; derleme çıktısını kontrol et."
fi

cat <<'EOF'

=== SONRAKİ ADIM: RINEX (ephemeris) dosyası ===
GPS-SDR-SIM güncel bir 'broadcast ephemeris' (RINEX nav) dosyası ister. İki yol:
  1) İnternetten (o günün dosyası — yarışma günü indir):
       NASA CDDIS / IGS: brdcDDD0.YYn  (DDD = yılın günü, YY = yıl)
       Örnek arama: "GPS broadcast ephemeris rinex daily brdc"
  2) Kendi GPS alıcından (u-blox vb.) RINEX kaydı.

Sonra baseband üret:
  python tools/gen_gps_spoof.py --rinex brdc.YYn --lat 39.89 --lon 32.78 --alt 900
Bu, data/gpssim_l1.bin üretir. Uygulamada GNSS Aldatma -> "GPS-SDR-SIM (gerçek ephemeris)" modu
onu 1575.42 MHz'de yayınlar.

⚠️ Havadan GPS yayınlamak yasak/tehlikeli — kalkanlı ortam / kablo+zayıflatıcı kullan.
EOF
