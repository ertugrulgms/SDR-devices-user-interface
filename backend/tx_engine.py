"""TX dalga-şekli üreticisi (TxWaveformBuilder).

SDRWorker'ın "Tanrı Nesnesi" olmaktan çıkarılması için TX baseband tampon üretimi buraya
taşındı. SDRWorker artık yalnızca donanımı sürer; hangi modda hangi baseband'in üretileceği,
hedef sinyal tipine göre parametrelerin nasıl ayarlanacağı ve DAC güvenliği burada yönetilir.

Tüm çıktılar JammingGenerator.enforce_dac_safe ile [-TX_DIGITAL_PEAK, TX_DIGITAL_PEAK] tepe
büyüklüğe sınırlanır -> donanımda DAC clipping İMKANSIZ (kritik bulgu #1)."""
import numpy as np

from backend.jamming_generator import JammingGenerator, TX_DIGITAL_PEAK
from backend.audio_deception import AnalogDeceptionModulator, DEFAULT_NBFM_DEVIATION_HZ

# Analog aldatmada gerçek ses mesajı (WAV) seçildiğinde kullanılan dalga-şekli etiketi
WAV_DECEPTION_WAVE = "Ses Dosyası (WAV Mesaj)"

# Loop'ta tekrar yayınlanacak baseband bloğu (~40k örnek). Sabit isimlendirildi (bulgu #13).
TX_BLOCK_SAMPLES = 2048 * 20

# --- HEDEF SİNYAL TİPİ PROFİLLERİ (bulgu #11) ---
# "Hedef Sinyal Tipi" artık dekoratif değil: seçilen hedefe göre karıştırma bant genişliği,
# çoklu-ton sayısı ve ton aralığı gerçekten değişir. bw_frac = örnekleme bandının kaplanan oranı.
#   - Dar-bant hedefler (analog/sayısal telsiz) -> gücü dar banda topla (yüksek güç yoğunluğu).
#   - Geniş-bant hedefler (veri linki, Wi-Fi) -> tüm bandı kapla.
DEFAULT_TARGET_PROFILE = {"bw_frac": 0.9, "n_tones": 5, "spacing_hz": None}
TARGET_PROFILES = {
    "Analog Telsiz":            {"bw_frac": 0.03, "n_tones": 3,  "spacing_hz": 12.5e3},
    "Sayısal Telsiz (DMR/P25)": {"bw_frac": 0.06, "n_tones": 4,  "spacing_hz": 12.5e3},
    "Geniş Bant Veri":         {"bw_frac": 0.90, "n_tones": 8,  "spacing_hz": None},
    "Wi-Fi 2.4 GHz":           {"bw_frac": 0.90, "n_tones": 12, "spacing_hz": None},
    "Wi-Fi 5.8 GHz":           {"bw_frac": 0.90, "n_tones": 12, "spacing_hz": None},
}

# Wi-Fi bandı merkez frekansları (MHz) — Wi-Fi engelleme artık gerçek baraj yayını yapar (bulgu #9).
WIFI_BAND_CENTERS = {
    "2.4 GHz (802.11 b/g/n)": 2437.0,   # kanal 6
    "5.8 GHz (802.11 a/n/ac)": 5800.0,
}


def target_profile(target_signal: str) -> dict:
    return TARGET_PROFILES.get(target_signal, DEFAULT_TARGET_PROFILE)


class TxWaveformBuilder:
    def __init__(self, jam_gen: JammingGenerator, gnss_gen, sample_rate_hz: float,
                 block_samples: int = TX_BLOCK_SAMPLES):
        self.jam_gen = jam_gen
        self.gnss_gen = gnss_gen
        self.sample_rate = sample_rate_hz
        self.block_samples = block_samples

    def build(self, params: dict):
        """Seçili moda göre DAC-güvenli baseband tamponu üretir.
        Dönüş: (buf: np.complex64[N], meta: dict) — meta, worker'ın loglayacağı ek bilgi
        (ör. GNSS Doppler/PRN listesi) taşır; worker'ı ince tutar."""
        mode = params.get("mode", "BARRAGE")
        amp = self.jam_gen.jsr_to_amplitude(params.get("jsr_db", 15.0), signal_amplitude=1.0)
        prof = target_profile(params.get("target_signal", ""))
        n = self.block_samples
        meta = {}

        # KAPALI-ÇEVRİM ODAK: RX'te ölçülen hedef bandı geçerliyse, baraj modlarında gücü tüm
        # banda değil O BANDA topla (bandlimit + offset kaydırma) -> güç yoğunluğu artar.
        focus_bw = float(params.get("focus_bw_hz", 0.0) or 0.0)
        focus_off = float(params.get("focus_offset_hz", 0.0) or 0.0)
        focus_active = 0.0 < focus_bw < 0.85 * self.sample_rate

        if mode in ("SPOT", "SINGLE_TONE"):
            # TEKLİ (SPOT) karıştırma: gücü TEK hedef kanalına yoğunlaştırılmış DAR-BANT GÜRÜLTÜ.
            # (Saf CW ton gürültü tabanını yükseltmez -> Shannon-etkin değil; bkz. generate_spot_noise.)
            # Kapalı-çevrim: RX'te hedef BW ölçüldüyse ona tam daralt + hedef offsetine kaydır (en
            # yüksek güç yoğunluğu). Yoksa profil bandının dar bir kesrine daralt (spot < baraj).
            if focus_active:
                spot_bw = min(0.25, max(0.004, (focus_bw / self.sample_rate) * 1.2))
                off = focus_off
                meta["focus_bw_hz"] = focus_bw
                meta["focus_offset_hz"] = focus_off
            else:
                spot_bw = min(max(prof["bw_frac"] * 0.5, 0.004), 0.05)
                off = 0.0
            buf = self.jam_gen.generate_spot_noise(n, amplitude=amp, bw_frac=spot_bw, offset_hz=off)

        elif mode == "MULTI_TONE":
            # ÇOKLU karıştırma: GÜRÜLTÜ tabanlı çok-bant tarağı (Shannon-etkin — saf CW ton DEĞİL).
            # Hedef profilindeki bant sayısı kadar dar-bant gürültü alt-bandı, bandın %80'ine yayılır.
            buf = self.jam_gen.generate_multi_band_noise(
                n, amplitude=amp, n_bands=max(2, int(prof["n_tones"])), band_bw_frac=0.03)

        elif mode == "SWEEP":
            # Süpürmeli (chirp), faz-sürekli; frekans-çevik hedefler için. Süpürme bandı hedefe göre.
            buf = self.jam_gen.generate_chirp_sweep(
                n, amplitude=amp, sweep_fraction=max(0.1, prof["bw_frac"]), n_sweeps=16)

        elif mode == "ANALOG_SPOOF":
            wave_type = params.get("wave_type", "Sinüs Dalga (Tone)")
            decept_audio = params.get("decept_audio", None)
            if wave_type == WAV_DECEPTION_WAVE and decept_audio is not None and len(decept_audio) > 1:
                # GERÇEK SES ALDATMA (5.2.3): operatörün WAV mesajını hedef modülasyonuna (NBFM/AM)
                # taşı -> hedef analog telsiz gerçek-dışı ama ANLAŞILIR yayını çalar ('yanlış duyar').
                dm = AnalogDeceptionModulator(self.sample_rate)
                buf = dm.modulate(
                    np.asarray(decept_audio, dtype=np.float64),
                    int(params.get("decept_audio_rate", 48000)),
                    mode=params.get("decept_mod", "NBFM"),
                    deviation_hz=float(params.get("decept_deviation_hz", DEFAULT_NBFM_DEVIATION_HZ)),
                    amplitude=amp,
                    # CTCSS alt-ses tonu (hedef ton-squelch'ini açar; ton yoksa hoparlör açılmaz)
                    ctcss_hz=float(params.get("decept_ctcss_hz", 0.0) or 0.0),
                    ctcss_dev_hz=float(params.get("decept_ctcss_dev_hz", 500.0)),
                    preemphasis=bool(params.get("decept_preemphasis", True)),
                    # DCS: yakalanan alt-ses kodu geri-oynatılır (CTCSS yoksa; kod-squelch açar)
                    subaudio=params.get("decept_subaudio", None),
                    subaudio_rate=float(params.get("decept_subaudio_rate", 0.0) or 0.0))
                meta["deception_audio_samples"] = int(len(buf))
                meta["decept_mod"] = params.get("decept_mod", "NBFM")
                meta["decept_ctcss_hz"] = float(params.get("decept_ctcss_hz", 0.0) or 0.0)
            else:
                buf = self.jam_gen.generate_analog_spoofing(
                    wave_type, params.get("offset_ms", 12.0), n, amplitude=amp)

        elif mode == "GNSS_SPOOF":
            # Otonom ve Dinamik Modları Yönet:
            spoof_mode = params.get("spoof_mode", "MANUAL")
            base_coords = params.get("coords", "39.9207, 32.8541")
            
            # Zaman damgası: Sistem çalışma süresini SDRWorker 'dan alırız
            # Eğer SDRWorker bunu henüz göndermiyorsa geçici 0.0 kullanırız.
            time_sec = params.get("time_sec", 0.0) 
            
            # Koordinat Çözümleyici (Spoof Mode Logic)
            if spoof_mode == "AUTONOMOUS_RANDOM":
                # Otonom mod: Rastgele uzak bir okyanus noktası (sabit)
                # Otonom mod her defasında aynı yere ışınlamak için hash tabanlı seed kullanır
                import hashlib
                h = int(hashlib.md5("AUTONOMOUS".encode()).hexdigest(), 16)
                rng = np.random.default_rng(h & 0xFFFFFFFF)
                lat = rng.uniform(-60, 60)
                lon = rng.uniform(-180, 180)
                target_coords = f"{lat:.4f}, {lon:.4f}"
            elif spoof_mode == "DYNAMIC_DRIFT":
                # Dinamik Drift: Hedefi kuzeye doğru saniyede 100 metre kaydırır.
                try:
                    b_lat, b_lon = map(float, base_coords.split(","))
                except Exception:
                    b_lat, b_lon = 39.9207, 32.8541
                speed_mps = 100.0
                R_e = 6378137.0
                lat_offset_deg = np.degrees((speed_mps * time_sec) / R_e)
                target_coords = f"{b_lat + lat_offset_deg:.6f}, {b_lon:.6f}"
            else:
                target_coords = base_coords

            service = params.get("gnss_code", "GPS L1")
            
            # Yeni Mimari: calculate_doppler_from_coords yerine, synthesize_service artık
            # gerçek kinematiği (Time of Flight ve Doppler'i) içeride hesaplıyor.
            buf = self.gnss_gen.synthesize_service(service, n, amplitude=amp, target_coords=target_coords, time_sec=time_sec)

            # Temsili Doppler: her uydunun Doppler'i ayrı hesaplanır; raporlama/log için ortalama
            # |Doppler| (servis taşıyıcısıyla ölçekli) meta'ya konur.
            dop_hz = 0.0
            lk = getattr(self.gnss_gen, "last_kinematics", None)
            if lk:
                dop_hz = float(np.mean([abs(v.get("doppler_hz", 0.0)) for v in lk.values()]))
            meta = {"gnss_service": service, "target_coords": target_coords,
                    "spoof_mode": spoof_mode, "gnss_doppler_hz": round(dop_hz, 1)}
            
        elif mode == "LOOK_THROUGH":
            # Arabakışlı karıştırma: aç/kapa (T/R) artık worker'da GERÇEK zaman-paylaşımıyla
            # yapılır (set_tx_active on/off). Buffer bu yüzden SÜREKLİ baraj gürültüsüdür;
            # dinleme penceresinde donanım TX'i fiziksel olarak kapanır (bulgu #3).
            buf = self._barrage(n, amp, prof["bw_frac"], focus_active, focus_bw, focus_off, meta)

        elif mode == "WIFI_JAMMING":
            # Wi-Fi engelleme artık GERÇEK: seçilen Wi-Fi bandını geniş-bant baraj ile doldurur
            # (frekans geçişi worker'da yapılır). "Deauth" gibi sahte seçenek kaldırıldı (bulgu #9).
            buf = self._barrage(n, amp, 0.9, focus_active, focus_bw, focus_off, meta)

        else:  # BARRAGE ve diğerleri -> hedef profiline göre (dar/geniş) baraj gürültüsü
            buf = self._barrage(n, amp, prof["bw_frac"], focus_active, focus_bw, focus_off, meta)

        # KRİTİK: donanıma gitmeden önce DAC-güvenli tepe büyüklüğe sınırla (clipping imkansız)
        buf = self.jam_gen.enforce_dac_safe(buf, peak=TX_DIGITAL_PEAK)
        return np.ascontiguousarray(buf, dtype=np.complex64), meta

    def _barrage(self, n, amp, prof_bw_frac, focus_active, focus_bw, focus_off, meta):
        """Baraj gürültüsü üretir; KAPALI-ÇEVRİM ODAK aktifse gücü ölçülen hedef bandına toplar
        (bandlimit + offset kaydırma) -> aynı güçle daha yüksek güç yoğunluğu (etkili jamming)."""
        if focus_active:
            # Hedef bandını biraz genişleterek (1.5x) kenar kaymalarını tolere et
            bw_frac = min(0.85, max(0.01, (focus_bw / self.sample_rate) * 1.5))
            buf = self.jam_gen.generate_barrage_noise(n, amplitude=amp, bw_frac=bw_frac)
            if abs(focus_off) > 1.0:
                t = np.arange(n) / self.sample_rate
                buf = (buf * np.exp(1j * 2 * np.pi * focus_off * t)).astype(np.complex64)  # hedefe kaydır
            meta["focus_bw_hz"] = focus_bw
            meta["focus_offset_hz"] = focus_off
            return buf
        return self.jam_gen.generate_barrage_noise(n, amplitude=amp, bw_frac=prof_bw_frac)
