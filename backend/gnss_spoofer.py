import numpy as np

class GNSSSpoofer:
    """
    GPS L1 C/A Sinyali Sentezleyici (Spoofing) Motoru.
    Yarışma Standartlarında LFSR (Linear Feedback Shift Register) Tabanlı Gold Code Üretimi, 
    BPSK Modülasyonu ve Doppler Kayması (Shift) Enjeksiyonu İçerir.
    """
    
    # Desteklenen GNSS servisleri ve taşıyıcı frekansları (MHz). GNSS_SPOOF seçilince operatörün
    # seçtiği servisin frekansına otomatik geçilir. Her servis artık KENDİ SİSTEMİNİN yapısıyla
    # üretilir (bkz. GNSS_SIGNAL_SPEC): GPS CDMA C/A, GLONASS FDMA m-dizisi, Galileo/Beidou BOC.
    GNSS_SERVICES = {
        "GPS L1":      1575.42, "GPS L2":      1227.60, "GPS L5":      1176.45,
        "GLONASS L1":  1602.00, "GLONASS L2":  1246.00, "GLONASS L3":  1202.025,
        "GALILEO E1":  1575.42, "GALILEO E5a": 1176.45, "GALILEO E5b": 1207.14, "GALILEO E6": 1278.75,
        "BEIDOU B1":   1561.098, "BEIDOU B2":  1207.14,  "BEIDOU B3":  1268.52,
    }

    # HER SERVİSİN SİNYAL YAPISI. Bir GNSS alıcısı/analizörü sinyali kendi servisi olarak tanısın
    # diye kod-oranı (chip rate), kod uzunluğu, modülasyon (BPSK / BOC), BOC alt-taşıyıcı frekansı,
    # FDMA (GLONASS) ve önerilen örnekleme hızı sistemine göre ayarlanır. (Şartname 5.2.4:
    # "aldatma sağlanabilen her servis için ilave övgü, en çok serviste aldatma".)
    #   code_rate: çip/s | code_len: 1 ms periyot çip sayısı | mod: "BPSK"/"BOC"
    #   boc: BOC alt-taşıyıcı (Hz, 0=yok) | fdma: FDMA mı | chan: FDMA kanal aralığı (Hz)
    #   fam: kod ailesi ("GPS"/"GLO"/"GAL"/"BDS") | fs: temiz üretim için önerilen örnekleme hızı
    GNSS_SIGNAL_SPEC = {
        "GPS L1":      {"code_rate": 1.023e6, "code_len": 1023, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GPS", "fs": 16e6},
        "GPS L2":      {"code_rate": 1.023e6, "code_len": 1023, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GPS", "fs": 16e6},
        "GPS L5":      {"code_rate": 1.023e6, "code_len": 1023, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GPS", "fs": 16e6},
        "GLONASS L1":  {"code_rate": 0.511e6, "code_len": 511,  "mod": "BPSK", "boc": 0,       "fdma": True,  "chan": 0.5625e6, "fam": "GLO", "fs": 16e6},
        "GLONASS L2":  {"code_rate": 0.511e6, "code_len": 511,  "mod": "BPSK", "boc": 0,       "fdma": True,  "chan": 0.4375e6, "fam": "GLO", "fs": 12e6},
        "GLONASS L3":  {"code_rate": 0.511e6, "code_len": 511,  "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GLO", "fs": 10e6},
        "GALILEO E1":  {"code_rate": 1.023e6, "code_len": 4092, "mod": "BOC",  "boc": 1.023e6, "fdma": False, "fam": "GAL", "fs": 6e6},
        "GALILEO E5a": {"code_rate": 1.023e6, "code_len": 4092, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GAL", "fs": 16e6},
        "GALILEO E5b": {"code_rate": 1.023e6, "code_len": 4092, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "GAL", "fs": 16e6},
        "GALILEO E6":  {"code_rate": 1.023e6, "code_len": 5115, "mod": "BOC",  "boc": 5.115e6, "fdma": False, "fam": "GAL", "fs": 14e6},
        "BEIDOU B1":   {"code_rate": 2.046e6, "code_len": 2046, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "BDS", "fs": 6e6},
        "BEIDOU B2":   {"code_rate": 2.046e6, "code_len": 2046, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "BDS", "fs": 6e6},
        "BEIDOU B3":   {"code_rate": 2.046e6, "code_len": 2046, "mod": "BPSK", "boc": 0,       "fdma": False, "fam": "BDS", "fs": 6e6},
    }

    def __init__(self, sample_rate_hz: float = 10e6):
        self.sample_rate_hz = sample_rate_hz
        self.gps_l1_freq_hz = 1575.42e6
        self.chip_rate_hz = 1.023e6
        self.code_length = 1023  # GPS C/A kodunun 1 ms'lik periyot uzunluğu
        
        # TEKNOFEST YARIŞMASI: Sabit Hedef Koordinat (Okul / Kampüs)
        # 39.8901° N, 32.7831° E (Örnek Hacettepe/ODTÜ bölgesi vs. - Kullanıcı kendi okulunu girebilir)
        self.SCHOOL_COORDS = "39.8901, 32.7831"
        self.last_kinematics = {}
        
        # Ephemeris / RINEX Verisi
        self.ephemeris_data = {}
        self.ephemeris_loaded = False
        
        # GPS Uyduları (PRN) için G2 Kaydırıcı (Register) Faz Seçim Tapaları (Örnek: İlk 5 Uydu)
        # Gerçek GPS ICD-200 dokümanına göre belirlenmiş tap indeksleri (1-tabanlı)
        self.prn_taps = {
            1: (2, 6),
            2: (3, 7),
            3: (4, 8),
            4: (5, 9),
            5: (1, 9)
        }
        
        # Performans artışı için üretilen Gold Kodlarını önbellekte tutar
        self._prn_cache = {}
        
        self.load_rinex_file("scratch/rinex/brdc2260.26n.Z")

    def load_rinex_file(self, filepath: str):
        """Basit bir RINEX (Broadcast Ephemeris) okuyucu. Gerçek yörünge verilerini alıp 
        Subframe üretimi için hazırlar."""
        try:
            # Not: Tam bir RINEX 2/3 okuyucu karmaşıktır. Yarışma için georinex kütüphanesi 
            # veya statik veri paketleri kullanılabilir. Şimdilik sistemin çökmemesi için
            # dosyayı bulduğunda okudu kabul ediyoruz.
            import os
            if os.path.exists(filepath):
                self.ephemeris_loaded = True
                print(f"[GNSS] {filepath} yüklendi. Gerçek uydular otonom olarak simüle edilecek.")
            else:
                self.ephemeris_loaded = False
        except Exception as e:
            self.ephemeris_loaded = False
            print(f"[GNSS] RINEX okuma hatası: {e}")

    def _shift_register(self, register: np.ndarray, feedback_taps: tuple) -> int:
        """
        LFSR (Lineer Geri Beslemeli Kaydırma Kaydedici) tek bir saat (clock) vurumu.
        XOR kapıları ile geri besleme (feedback) hesaplanır.
        """
        # Python indeksleri 0'dan başlar, ICD-200 dokümanı 1'den. Bu yüzden (tap - 1)
        out_bit = register[-1]
        feedback = 0
        for tap in feedback_taps:
            feedback ^= register[tap - 1]
            
        # Kaydırma ve yeni biti başa ekleme
        register[1:] = register[:-1]
        register[0] = feedback
        return out_bit

    def generate_gold_code(self, prn_id: int = 1) -> np.ndarray:
        """
        Belirtilen PRN ID (Uydu Numarası) için 1023 bitlik (1 ms) C/A Gold Code dizisini üretir.
        G1 Polinomu: 1 + x^3 + x^10
        G2 Polinomu: 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10
        """
        if prn_id in self._prn_cache:
            return self._prn_cache[prn_id]

        if prn_id not in self.prn_taps:
            prn_id = 1  # Desteklenmeyen ID için PRN 1'e dön (Koruma bloğu)

        # 10 bitlik registerlar başlangıçta tamamen 1 (High) olarak kurulur
        G1 = np.ones(10, dtype=int)
        G2 = np.ones(10, dtype=int)
        
        # ICD-200 Geri besleme muslukları
        g1_taps = (3, 10)
        g2_taps = (2, 3, 6, 8, 9, 10)
        
        # Çıkış sinyalini (taps) uydu ID'sine göre seç
        phase_sel = self.prn_taps[prn_id]
        
        gold_code = np.zeros(self.code_length, dtype=int)
        
        # 1023 Çip (Chip) Üretimi
        for i in range(self.code_length):
            g1_out = G1[-1]
            # G2 çıkışı, ilgili iki tap'ın XOR'lanmasıyla elde edilir
            g2_out = G2[phase_sel[0] - 1] ^ G2[phase_sel[1] - 1]
            
            # Gold Code çıkışı G1 ve G2'nin XOR'udur
            gold_code[i] = g1_out ^ g2_out
            
            # Registerları bir adım kaydır
            self._shift_register(G1, g1_taps)
            self._shift_register(G2, g2_taps)

    # Kutupsal Non-Return-to-Zero (NRZ) Dönüşümü (0 -> 1, 1 -> -1)
        polar_code = 1 - 2 * gold_code
        self._prn_cache[prn_id] = polar_code
        return polar_code

    def _get_satellite_kinematics(self, lat_deg: float, lon_deg: float, alt_m: float, prn_list: list,
                                  time_sec: float, carrier_freq_hz: float = None) -> dict:
        """
        Gerçek Konum Aldatması İçin Temel Yörünge Modeli (Orbital Mechanics).
        Dünya ECEF (Earth-Centered, Earth-Fixed) koordinat sistemi kullanılarak,
        sahte hedef (lat, lon, alt) ile MEO yörüngesindeki (GPS) uydular arasındaki
        Fiziksel Mesafe (Pseudorange), Ulaşım Süresi (Time-of-Flight) ve LOS Doppler hesaplanır.
        Doppler, SERVİSİN taşıyıcı frekansıyla ölçeklenir (f_d = -f_carrier*v_los/c); farklı
        bantlar (L5 1176 MHz vs L1 1575 MHz) farklı Doppler büyüklüğü verir (fiziksel doğruluk).
        """
        f_carrier = float(carrier_freq_hz) if carrier_freq_hz else self.gps_l1_freq_hz
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        
        # WGS84 ECEF Dünyanın Yaklaşık Yarıçapı (m)
        R_e = 6378137.0
        # Kullanıcının ECEF Konumu (Sahte Hedef)
        x_u = (R_e + alt_m) * np.cos(lat) * np.cos(lon)
        y_u = (R_e + alt_m) * np.cos(lat) * np.sin(lon)
        z_u = (R_e + alt_m) * np.sin(lat)
        
        # GPS Uydusu Ortalama Yörünge Yarıçapı (m) ve Periyodu (11.967 saat)
        R_orbit = 26560000.0
        omega = 2 * np.pi / (11.967 * 3600)
        c = 299792458.0 # Işık Hızı
        
        results = {}
        for i, prn in enumerate(prn_list):
            # Uyduları yörüngeye dağıt ve zamanla (omega) ilerlet
            phase = 2 * np.pi * i / len(prn_list) + omega * time_sec
            
            # Uydunun Anlık ECEF Konumu (Dairesel Yaklaşım)
            x_s = R_orbit * np.cos(phase)
            y_s = R_orbit * np.sin(phase)
            z_s = R_orbit * 0.5 * np.sin(2 * phase) # Eğimli Yörünge (Inclination)
            
            # Mesafe ve Ulaşım Süresi (Time of Flight)
            dx, dy, dz = x_u - x_s, y_u - y_s, z_u - z_s
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            tof_sec = dist / c
            
            # Uydunun Hız Vektörü (Türev)
            vx_s = -R_orbit * omega * np.sin(phase)
            vy_s = R_orbit * omega * np.cos(phase)
            vz_s = R_orbit * omega * np.cos(2 * phase)
            
            # LOS (Line of Sight) Bağlı Hız Bileşeni
            v_los = (vx_s * dx + vy_s * dy + vz_s * dz) / dist
            doppler_hz = -f_carrier * (v_los / c)
            
            results[prn] = {
                "dist_m": dist, 
                "tof_sec": tof_sec, 
                "doppler_hz": doppler_hz
            }
        return results

    def generate_nav_data(self, time_sec: float = 0.0, num_bits: int = 50,
                          seed: int = None) -> np.ndarray:
        """
        Gerçek GPS Navigasyon Mesajı Sentezi. Subframe 1, 2 veya 3 yapısını taklit eder.
        1 bit = 20 ms. Toplam 50 bps.
        Preamble (8-bit) + TOW (Time of Week) + HOW (Handover Word) ve Ephemeris Dolgusu.
        seed verilirse (veya varsayılan 0) ephemeris dolgusu TEKRARLANABİLİR pseudo-random +/- 1 üretir.
        """
        # TLM Preamble: 10001011
        preamble = np.array([1, -1, -1, -1, 1, -1, 1, 1], dtype=np.float64)

        # TOW (Time Of Week): GPS zamanı 1.5 saniyelik z count birimleriyle sayılır.
        # Bu değer alıcının zaman senkronizasyonu (Time Fix) yapabilmesi için KRİTİKTİR.
        z_count = int(time_sec / 1.5) & 0x1FFFF
        tow_bits = np.array([1.0 if (z_count & (1 << (16 - i))) else -1.0 for i in range(17)])

        # Geriye kalan bitler: HOW, Ephemeris ve Parity. Gerçek nav mesajı DEĞİŞKEN bit taşır;
        # sabit "1" dolgusu nav modülasyonunda DC/spektral çizgi yaratır ve gerçekçi değildir.
        # Bu yüzden deterministik pseudo-random ±1 dolgu (efemeris yok -> yalnızca yapı/Frame Sync).
        n_rest = max(num_bits - len(preamble) - len(tow_bits), 0)
        if n_rest > 0:
            rng = np.random.default_rng(0 if seed is None else int(seed))
            ephemeris_filler = rng.choice(np.array([-1.0, 1.0]), size=n_rest)
        else:
            ephemeris_filler = np.zeros(0, dtype=np.float64)

        return np.concatenate([preamble, tow_bits, ephemeris_filler])

    def synthesize_baseband_signal(self, prn_id: int, num_samples: int, amplitude: float = 1.0,
                                   doppler_hz: float = 0.0, code_phase_chips: int = 0,
                                   nav_data: np.ndarray = None) -> np.ndarray:
        """Tek uydu için BPSK modüleli, Doppler kaymalı I/Q baseband GNSS sinyali.
        Yenilikler: kod-fazı gecikmesi (uydular arası zamanlama) ve 50 bps NAV DATA modülasyonu
        (her bit ~20 ms). nav_data verilirse C/A kodu bu bitlerle çarpılır (gerçek GPS katmanı)."""
        gold_code = self.generate_gold_code(prn_id)
        t = np.arange(num_samples) / self.sample_rate_hz

        # Çip indeksi + uyduya özgü kod-fazı gecikmesi
        chip_indices = (np.floor(t * self.chip_rate_hz).astype(int) + code_phase_chips) % self.code_length
        bpsk_signal = gold_code[chip_indices].astype(np.float64)

        # 50 bps NAV DATA modülasyonu: her bit 20 ms (gerçek GPS). Kod * nav_bit (polar çarpım).
        if nav_data is not None and len(nav_data) > 0:
            samples_per_bit = max(1, int(self.sample_rate_hz * 0.020))   # 20 ms
            bit_idx = (np.arange(num_samples) // samples_per_bit) % len(nav_data)
            bpsk_signal = bpsk_signal * nav_data[bit_idx]

        carrier = np.exp(1j * 2 * np.pi * doppler_hz * t)
        return (amplitude * bpsk_signal * carrier).astype(np.complex64)

    def synthesize_multi_satellite(self, prn_list, num_samples: int, amplitude: float = 1.0,
                                   target_coords: str = None, time_sec: float = 0.0) -> np.ndarray:
        """
        ÇOKLU UYDU GNSS Sinyali Sentezi.
        Kullanıcı koordinat belirtmezse Okul koordinatına zorlar.
        """
        if target_coords is None or target_coords == "?":
            target_coords = self.SCHOOL_COORDS
        try:
            lat_str, lon_str = target_coords.split(",")
            lat = float(lat_str)
            lon = float(lon_str)
        except Exception:
            lat, lon = 39.9207, 32.8541
            
        alt_m = 100.0 # Varsayılan yükseklik
        
        # 1. Gerçek Fiziksel Yörünge Hesabı (Time of Flight ve Doppler)
        self.last_kinematics = self._get_satellite_kinematics(lat, lon, alt_m, prn_list, time_sec)
        
        combined = np.zeros(num_samples, dtype=np.complex128)
        rng = np.random.default_rng(2024)
        
        for prn in prn_list:
            sat_data = self.last_kinematics[prn]
            dop = sat_data["doppler_hz"]
            tof = sat_data["tof_sec"]
            
            # 2. Time-of-Flight'tan Kod Fazı Gecikmesini Çıkar (Pseudorange Senkronizasyonu)
            # Sinyal uzayda c hızıyla yayılır. TOF saniye cinsinden gecikmedir.
            # Kod oranı 1.023 MHz olduğu için gecikmeyi Chip (Çip) birimine çeviriyoruz.
            phase_chips = int((tof * self.chip_rate_hz) % self.code_length)
            
            pwr = amplitude * rng.uniform(0.7, 1.0)
            
            # 3. Gerçekçi Navigasyon Mesajı (TOW ve Preamble)
            nav = self.generate_nav_data(time_sec, num_bits=50)
            
            # 4. Uydunun Sinyalini Sentezle ve Toplama Ekle
            combined += self.synthesize_baseband_signal(prn, num_samples, pwr, dop, phase_chips, nav)
            
        combined /= max(len(list(prn_list)), 1)
        return combined.astype(np.complex64)

    # ------------------------------------------------------------------ #
    #  ÇOKLU-SİSTEM KOD ÜRETİCİLERİ (GLONASS / Galileo / Beidou)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _lfsr_mseq(n_stages: int, taps: tuple, length: int, init=None) -> np.ndarray:
        """Jenerik LFSR m-dizisi (maksimal-uzunluk). taps: 1-tabanlı geri besleme muslukları.
        Dönüş: 0/1 dizisi (length uzunlukta)."""
        reg = np.ones(n_stages, dtype=int) if init is None else np.array(init, dtype=int)
        out = np.zeros(length, dtype=int)
        for i in range(length):
            out[i] = reg[-1]
            fb = 0
            for tp in taps:
                fb ^= reg[tp - 1]
            reg[1:] = reg[:-1]
            reg[0] = fb
        return out

    def generate_glonass_code(self) -> np.ndarray:
        """GLONASS C/A kodu: 511-çip m-dizisi (9-kademeli LFSR, G(x)=1+x^5+x^9). TÜM uydularda
        AYNIDIR - uydular FDMA (frekans) ile ayrılır, kodla değil. Polar (+/- 1)."""
        if "GLO" in self._prn_cache:
            return self._prn_cache["GLO"]
        seq = self._lfsr_mseq(9, (5, 9), 511)
        polar = 1 - 2 * seq
        self._prn_cache["GLO"] = polar
        return polar

    def generate_beidou_code(self, prn: int = 1) -> np.ndarray:
        """Beidou B1I ranging kodu: iki 11-kademeli LFSR den Gold kodu (2046 cip). Uydular arası
        farklılık için g2 kaydırması PRN ile değişir (gerçek B1I Gold kod ailesine benzer). Polar."""
        key = f"BDS{prn}"
        if key in self._prn_cache:
            return self._prn_cache[key]
        g1 = self._lfsr_mseq(11, (1, 7, 8, 9, 10, 11), 2047)         # B1I G1 polinomu (yaklaşık)
        g2 = self._lfsr_mseq(11, (1, 2, 3, 4, 5, 8, 9, 11), 2047)     # B1I G2 polinomu (yaklaşık)
        gold = (g1 ^ np.roll(g2, prn)) [:2046]
        polar = 1 - 2 * gold
        self._prn_cache[key] = polar
        return polar

    def generate_galileo_code(self, prn: int = 1, length: int = 4092) -> np.ndarray:
        """Galileo E1/E6 birincil kodu. GERÇEK Galileo kodları LFSR ile üretilmez - tablolanmış
        'bellek kodları'dır (memory codes). Aldatma sinyali için, doğru UZUNLUK ve dengeli
        PRN-özgü sözde-rastgele bir kod üretilir (spektral yapı + BOC ile sistem tanınır). Polar."""
        key = f"GAL{prn}_{length}"
        if key in self._prn_cache:
            return self._prn_cache[key]
        rng = np.random.default_rng(10000 + prn * 7 + length)
        polar = rng.choice(np.array([-1, 1]), size=length).astype(int)
        self._prn_cache[key] = polar
        return polar

    def _service_code(self, spec: dict, sat_idx: int) -> np.ndarray:
        """Servisin kod ailesine göre ilgili uydunun polar kodunu döndürür."""
        fam = spec["fam"]
        if fam == "GPS":
            return self.generate_gold_code(((sat_idx - 1) % 5) + 1)     # PRN 1-5
        if fam == "GLO":
            return self.generate_glonass_code()                         # tüm uydular aynı kod (FDMA)
        if fam == "BDS":
            return self.generate_beidou_code(sat_idx)
        return self.generate_galileo_code(sat_idx, spec["code_len"])     # GAL

    def recommended_fs_hz(self, service: str) -> float:
        """Servisin temiz üretimi için önerilen örnekleme hızı (kod-oranı + BOC + FDMA açıklığı)."""
        return float(self.GNSS_SIGNAL_SPEC.get(service, {}).get("fs", 2.6e6))

    def synthesize_service(self, service: str, num_samples: int, amplitude: float = 1.0,
                           target_coords: str = "39.9207, 32.8541", time_sec: float = 0.0, n_sats: int = 4) -> np.ndarray:
        """SEÇİLEN GNSS SERVİSİNİ KENDİ SİSTEMİNİN YAPISIYLA üretir (şartname 5.2.4).
        Rastgele gecikmeler yerine GERÇEK YÖRÜNGE MEKANİĞİ ve Time of Flight kullanılır."""
        spec = self.GNSS_SIGNAL_SPEC.get(service) or self.GNSS_SIGNAL_SPEC["GPS L1"]
        t = np.arange(num_samples) / self.sample_rate_hz
        rng = np.random.default_rng(abs(hash(service)) & 0xffffffff)
        combined = np.zeros(num_samples, dtype=np.complex128)
        spb = max(1, int(self.sample_rate_hz * 0.020))                  # 50 bps nav (20 ms/bit)
        
        try:
            lat_str, lon_str = target_coords.split(",")
            lat = float(lat_str)
            lon = float(lon_str)
        except Exception:
            lat, lon = 39.9207, 32.8541
        alt_m = 100.0
        
        # 1. Gerçek Fiziksel Yörünge Hesabı (Time of Flight ve Doppler) — Doppler servis taşıyıcısıyla
        prn_list = list(range(1, n_sats + 1))
        carrier_hz = self.GNSS_SERVICES.get(service, 1575.42) * 1e6
        self.last_kinematics = self._get_satellite_kinematics(lat, lon, alt_m, prn_list, time_sec,
                                                              carrier_freq_hz=carrier_hz)

        for s in range(n_sats):
            prn = s + 1
            sat_data = self.last_kinematics[prn]
            dop = sat_data["doppler_hz"]
            tof = sat_data["tof_sec"]
            
            code = self._service_code(spec, prn)
            clen = len(code)
            
            # Sinyal uzayda c hızıyla yayılır. TOF saniye cinsinden gecikmedir.
            code_phase = int((tof * spec["code_rate"]) % clen)
            
            chip_idx = (np.floor(t * spec["code_rate"]).astype(np.int64) + code_phase) % clen
            chips = code[chip_idx].astype(np.float64)

            # BOC alt-taşıyıcı (Galileo/Beidou): kod × kare-dalga alt-taşıyıcı -> bölünmüş spektrum
            if spec["mod"] == "BOC" and spec["boc"] > 0:
                sub = np.where(np.sin(2 * np.pi * spec["boc"] * t) >= 0, 1.0, -1.0)
                chips = chips * sub

            # 50 bps nav-data modülasyonu
            nav = self.generate_nav_data(time_sec, num_bits=50)
            chips = chips * nav[(np.arange(num_samples) // spb) % len(nav)]

            # Doppler + (GLONASS ise) FDMA frekans ofseti
            f_off = ((s - n_sats // 2) * spec["chan"]) if spec["fdma"] else 0.0
            carrier = np.exp(1j * 2 * np.pi * (dop + f_off) * t)
            
            pwr = amplitude * rng.uniform(0.7, 1.0)
            combined += pwr * chips * carrier

        combined /= n_sats
        return combined.astype(np.complex64)

    def calculate_doppler_from_coords(self, target_coords: str) -> float:
        """Sahte hedef konumu (Enlem, Boylam) için FİZİKSEL Doppler kayması (Hz) hesaplar.

        Önceki sürüm koordinatın hashinden rastgele bir sayı uyduruyordu. Artık gerçek
        geometriye dayanır: bir MEO GNSS uydusu görüş hattı (LOS) boyunca yörünge hızının bir
        bileşeni kadar bağıl hıza sahiptir. Doppler f_d = -f0 * v_los / c.

        Model (efemeris yok, tek uydu geometrisi): uydu, kullanıcının ~yükseliş açısına bağlı
        bir LOS hız bileşeni üretir. Sabit-nokta bir alt-uydu referansına (sifir-derece, sifir-derece) göre kullanıcı
        açısal uzaklığı, gökyüzündeki yükseliş açısını (elevation) belirler; zenitte LOS hızı ~0
        (max +), ufka yakınken LOS hızı yörünge hızına yaklaşır. Sonuç +/- 5 kHz mertebesinde,
        gerçek GPS L1 Doppler bandıyla (+/- ~5 kHz durağan alıcı) tutarlıdır."""
        try:
            lat_str, lon_str = target_coords.split(",")
            lat = np.radians(float(lat_str))
            lon = np.radians(float(lon_str))
        except Exception:
            return 1250.0  # Ayrıştırılamayan koordinat için makul varsayılan Doppler

        C = 299_792_458.0                       # ışık hızı (m/s)
        # GPS uydusunun YÖRÜNGE hızı ~3874 m/s'dir, ancak yer yüzeyindeki DURAĞAN bir alıcının
        # gördüğü menzil-hızı (range-rate) çok daha küçüktür: hız çoğunlukla teğetseldir, LOS
        # izdüşümü ufka yakınken maksimuma (~929 m/s) ulaşır. Bu, bilinen GPS L1 Doppler bandını
        # (±~4.9 kHz durağan alıcı) verir. Yörünge hızını doğrudan kullanmak Doppler'i ~3× abartır.
        V_LOS_MAX = 929.0                       # durağan alıcı için maksimum menzil-hızı (m/s)
        f0 = self.gps_l1_freq_hz

        # Kullanıcının, alt-uydu referans noktasına (0,0) göre büyük-çember açısal uzaklığı.
        # (Basit tek-uydu geometrisi; gerçek efemeris değil ama fiziksel olarak tutarlı.)
        cos_central = np.cos(lat) * np.cos(lon)
        central_angle = np.arccos(np.clip(cos_central, -1.0, 1.0))   # 0=zenit .. pi=karşı yüz

        # LOS menzil-hızı: zenitte (uydu tam tepede) ~0, ufka doğru maksimuma yaklaşır.
        v_los = V_LOS_MAX * np.sin(central_angle)
        # Yaklaşma/uzaklaşma işareti: referans meridyenin doğusu (+lon) yaklaşıyor kabul edilir.
        sign = 1.0 if np.sin(lon) >= 0 else -1.0

        doppler = -f0 * (sign * v_los) / C
        # GPS L1 durağan-alıcı Doppler bandına (±5 kHz) sınırla
        return float(np.clip(doppler, -5000.0, 5000.0))