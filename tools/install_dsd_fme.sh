#!/usr/bin/env bash
# =============================================================================
#  DSD-FME KURULUMU — Sayısal amatör telsiz SES ÇÖZME (DMR/YSF/P25/NXDN) (spec 5.1.3)
# =============================================================================
#  Sayısal telsizin sesi AMBE/IMBE vokoderle kodlanır (patentli). Açık çözümü mbelib + DSD-FME'dir.
#  Bu script Ubuntu 24.04'te bağımlılıkları kurar, mbelib ve DSD-FME'yi KAYNAKTAN derler.
#
#  KULLANIM:   bash tools/install_dsd_fme.sh
#  (sudo şifresi sorulacak — apt ve 'make install' için gerekli.)
#
#  Bitince uygulamada "Sayısal Çöz"e bas -> şifresiz DMR/YSF/P25/NXDN sesi hoparlörden çıkar.
# =============================================================================
set -euo pipefail

echo "==================================================================="
echo "  DSD-FME KURULUMU (mbelib + dsd-fme) — Ubuntu 24.04"
echo "==================================================================="

# --- 1) Bağımlılıklar ---------------------------------------------------------
echo ""; echo ">>> 1/4  Bağımlılıklar (apt)..."
sudo apt-get update
# Çekirdek bağımlılıklar (24.04'te mevcut). Ses çıkışı için PulseAudio + portaudio.
sudo apt-get install -y \
    git cmake build-essential pkg-config \
    libsndfile1-dev libncurses-dev libpulse-dev \
    libcodec2-dev portaudio19-dev libusb-1.0-0-dev libitpp-dev || \
sudo apt-get install -y \
    git cmake build-essential pkg-config \
    libsndfile1-dev libncurses-dev libpulse-dev \
    libcodec2-dev portaudio19-dev libusb-1.0-0-dev
    # (libitpp yoksa onsuz devam — DSD-FME opsiyonel olarak kullanır)

BUILD_DIR="${HOME}/dsd_build"
mkdir -p "$BUILD_DIR"

# CMake 4.x, eski projelerin (mbelib gibi) cmake_minimum_required(<3.5) çağrısını reddeder.
# Bu bayrak eski politikayla yine de yapılandırmaya izin verir (hata mesajının önerdiği çözüm).
CMFLAG="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# --- 2) mbelib (AMBE/IMBE vokoder kütüphanesi) --------------------------------
echo ""; echo ">>> 2/4  mbelib derleniyor..."
cd "$BUILD_DIR"
if [ ! -d mbelib ]; then git clone https://github.com/szechyjs/mbelib.git; fi
cd mbelib && git pull --ff-only || true
rm -rf build && mkdir -p build && cd build           # eski başarısız cache'i temizle
cmake $CMFLAG .. && make -j"$(nproc)" && sudo make install
sudo ldconfig

# --- 3) DSD-FME ---------------------------------------------------------------
echo ""; echo ">>> 3/4  DSD-FME derleniyor..."
cd "$BUILD_DIR"
if [ ! -d dsd-fme ]; then git clone https://github.com/lwvmobile/dsd-fme.git; fi
cd dsd-fme && git pull --ff-only || true
rm -rf build && mkdir -p build && cd build           # eski başarısız cache'i temizle
cmake $CMFLAG .. && make -j"$(nproc)" && sudo make install
sudo ldconfig

# --- 4) Doğrulama -------------------------------------------------------------
echo ""; echo ">>> 4/4  Doğrulama..."
if command -v dsd-fme >/dev/null 2>&1; then
    echo "✅ BAŞARILI: dsd-fme kuruldu -> $(command -v dsd-fme)"
    dsd-fme -h 2>&1 | head -8 || true
    echo ""
    echo "Artık uygulamada 'Sayısal Çöz' butonu DSD-FME'yi otomatik bulur ve kullanır."
    echo "Şifresiz DMR/YSF/P25/NXDN yayınında ses hoparlörden çıkar."
else
    echo "⚠️  dsd-fme PATH'te görünmüyor. /usr/local/bin'i kontrol et:"
    ls -l /usr/local/bin/dsd-fme 2>/dev/null || echo "   (bulunamadı — derleme çıktısını yukarıda kontrol et)"
fi
