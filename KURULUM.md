# SDR EH Kontrol Paneli — Kurulum Rehberi

Bu proje, **USRP B200mini-i** ve benzeri SDR donanımları ile çalışmak üzere tasarlanmıştır.
Donanım bağlı değilse uygulama otomatik olarak **simülasyon moduna** düşer, yani arayüz
donanımsız da açılıp test edilebilir. Aşağıdaki adımlar Linux (Ubuntu 22.04+) içindir.

---

## Adım 1: Depoyu Klonlama
```bash
git clone https://github.com/ertugrulgms/SDR-devices-user-interface.git
cd SDR-devices-user-interface
```

## Adım 2: Sistem Bağımlılıkları (Donanım / C++ motoru için)
`sdr_core` C++ modülü, UHD ve SoapySDR kütüphanelerine bağlıdır. Sadece simülasyon
modunda çalışacaksanız bu adımı atlayabilirsiniz (uygulama yine açılır).

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake python3-dev python3-pip python3-venv \
    libuhd-dev uhd-host \
    libsoapysdr-dev soapysdr-tools soapysdr-module-uhd \
    pybind11-dev
```

UHD FPGA/firmware imajlarını indirin (USRP ilk kullanımda gerekir):
```bash
sudo uhd_images_downloader
```

Donanımın göründüğünü doğrulayın:
```bash
SoapySDRUtil --find          # "driver=uhd" cihazını listelemeli
uhd_find_devices             # B200mini-i görünmeli
```

## Adım 3: Python Ortamı ve Bağımlılıklar
> **Python 3.10 veya üstü gerekir** (kod PEP 604 `X | None` tip söz dizimini kullanır).
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` gerçek zamanlı ses için `sounddevice`'i de kurar (spec 5.1.3).
Sayısal amatör telsiz **sesini çözmek** için (opsiyonel) harici DSD-FME gerekir; kurulu
değilse 4FSK/C4FM tespiti yine çalışır (bkz. requirements.txt notu).

## Adım 4: C++ SDR Motorunu (`sdr_core`) Derleme
Bu adım `backend/sdr_core.so` dosyasını üretir. Derlenmezse uygulama simülasyon
modunda çalışır (spektrum boş/düz görünür, gerçek RF alınmaz).

```bash
cd backend
mkdir -p build && cd build
cmake ..
make -j$(nproc)
# Üretilen modülü backend/ altına kopyala (Python buradan import eder)
cp sdr_core.so ../
cd ../..
```

> Not: `.so` dosyası platforma özgüdür ve depoya dahil **edilmez**; her makinede
> yukarıdaki adımla yeniden derlenir.

## Adım 5: Uygulamayı Çalıştırma
```bash
# venv aktifken, projenin kök dizininde:
python main.py
```

- **"Sinyal Alımını Başlat"** ile RX motoru başlar.
- Donanım bağlıysa gerçek spektrum akar; değilse simülasyon modu devreye girer.

## Adım 6 (Opsiyonel): Ağ Yön Bulma Simülasyonu
Otonom 3-cihaz yön bulma verisini (UDP) test etmek için ayrı bir terminalde:
```bash
python backend/network_node.py
```

---

## Sorun Giderme
- **`ModuleNotFoundError: sdr_core`** → C++ modülü derlenmemiş; Adım 4'ü uygulayın.
  (Bu bir hata değildir; uygulama simülasyon moduna düşer.)
- **`SoapySDRUtil --find` cihazı bulamıyor** → UHD kurulumunu ve USB izinlerini
  (udev kuralları / `sudo`) kontrol edin, `uhd_images_downloader` çalıştırın.
- **Ekran siyah / grafik hatası** → `main.py` OpenGL ES niteliğini zorlar; sürücü
  sorununda `export QT_OPENGL=software` deneyin.
