import os
import json
import numpy as np

# --- SABİTLER ---
C_SPEED = 299792458.0  # Işık hızı (m/s)

# Mutlak güç kalibrasyon ofsetinin kalıcı saklandığı dosya (dBFS -> dBm tek-nokta ofseti).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_CAL_PATH = os.path.join(_DATA_DIR, "power_calibration.json")


class DSPProcessor:
    """
    Elektronik Destek (ED) Sinyal İşleme Motoru.
    3 Kanallı I/Q verisinden FFT Spektrumu, Faz Farkı, Açı (AOA)
    ve Nirengi (Triangulation) ile Konum (X, Y) hesaplar.
    """

    def __init__(self, fft_size: int = 512, antenna_spacing_m: float = 0.0625):
        """
        :param fft_size: FFT Nokta Sayısı (ör. 512)
        :param antenna_spacing_m: Antenler arası mesafe (m) - Örn: 2.4 GHz için d = lambda/2 ~ 6.25 cm
        """
        self.fft_size = fft_size
        self.antenna_spacing_m = antenna_spacing_m
        # MUTLAK GÜÇ KALİBRASYONU (dBFS -> dBm tek-nokta ofseti). 0 = kalibre edilmemiş (ham dBFS).
        # Bilinen bir kaynakla (sinyal jeneratörü) calibrate_power ile hesaplanıp kalıcı saklanır.
        self.cal_offset_db = self._load_cal_offset()

    # ------------------------------------------------------------------ #
    #  MUTLAK GÜÇ KALİBRASYONU (dBFS -> dBm)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_cal_offset() -> float:
        try:
            with open(_CAL_PATH, encoding="utf-8") as f:
                return float(json.load(f).get("cal_offset_db", 0.0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0.0

    def set_cal_offset(self, offset_db: float):
        """Kalibrasyon ofsetini doğrudan ayarla ve kalıcı sakla."""
        self.cal_offset_db = float(offset_db)
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_CAL_PATH, "w", encoding="utf-8") as f:
                json.dump({"cal_offset_db": self.cal_offset_db}, f)
        except OSError:
            pass
        return self.cal_offset_db

    def calibrate_power(self, iq_samples: np.ndarray, known_dbm: float) -> float:
        """TEK-NOKTA KALİBRASYON: bilinen güçte (known_dbm, ör. sinyal jeneratörü -50 dBm)
        bir sinyal verilirken çağrılır. Ham (ofsetsiz) dBFS ölçülüp ofset = known_dbm - dBFS
        hesaplanır ve kalıcı saklanır. Bundan sonra tüm güç değerleri gerçek dBm'e yakınsar."""
        raw_power = np.mean(np.abs(iq_samples) ** 2)
        raw_dbfs = 10.0 * np.log10(raw_power + 1e-12)   # ofsetsiz ham dBFS
        return self.set_cal_offset(float(known_dbm) - raw_dbfs)

    def compute_fft_dbm(self, iq_samples: np.ndarray):
        """
        Karmaşık (Complex) I/Q verisinden dBFS cinsinden FFT güç spektrumu hesaplar.
        Dönüş: (EMA-yumuşatılmış görüntü dizisi, ham spektrum dizisi) — ikisi de dBFS.
        (Metod adı tarihsel nedenle "_dbm"; değer artık kalibre edilmemiş dBFS'tir.)

        WELCH (SEGMENT ORTALAMA) yöntemi: Tek uzun FFT çok gürültülüdür; her bin
        kareden kareye ±5-10 dB zıplar, spektrum çizgisi KALIN ve titrek görünür.
        Bunun yerine veri örtüşen kısa segmentlere bölünür, her segmentin güç
        spektrumu ayrı hesaplanıp ORTALANIR. Bu, gürültü varyansını ~sqrt(segment)
        kadar düşürür → ince, pürüzsüz, kararlı bir çizgi (profesyonel spektrum
        analizörü davranışı). Gerçek sinyaller ortalamada hayatta kalır, sadece
        rastgele gürültü bastırılır.
        """
        # Yazılımsal DC Offset Temizleme: B200mini gibi Zero-IF cihazların merkez
        # (0 Hz) LO sızıntısı dikenini yok eder.
        x = iq_samples - np.mean(iq_samples)
        n = len(x)

        seg = 512
        if n >= seg * 2:
            step = seg // 4              # %75 örtüşme -> daha çok segment, daha pürüzsüz
            window = np.hamming(seg)
            win_norm = np.sum(window ** 2)
            acc = np.zeros(seg)
            count = 0
            for start in range(0, n - seg + 1, step):
                block = x[start:start + seg] * window
                acc += np.abs(np.fft.fftshift(np.fft.fft(block))) ** 2
                count += 1
            power = acc / (count * win_norm)
            # UI x-ekseni self.fft_size nokta bekliyor; segment çözünürlüğünü ona genişlet
            if self.fft_size != seg:
                power = np.interp(np.linspace(0, 1, self.fft_size),
                                  np.linspace(0, 1, seg), power)
        else:
            window = np.hamming(n)
            fft_shifted = np.fft.fftshift(np.fft.fft(x * window, n=self.fft_size))
            power = np.abs(fft_shifted) ** 2 / self.fft_size

        # dBFS: referans = ADC tam skalası (|örnek|=1 -> 0 dBFS). Önceki sürümde buraya
        # eklenen "+30" sabiti değeri sözde "dBm"e çeviriyordu ama fiziksel temeli yoktu
        # (0 dBFS = 1 W varsayımı; gerçekte cihazın RF kazancı, anten kazancı ve ADC referansı
        # bilinmeden dBFS'ten dBm'e geçilemez). Kaldırıldı: değer artık dürüst, kalibre
        # edilmemiş BAĞIL dBFS'tir. Mutlak dBm için bilinen bir kaynakla tek-nokta kalibrasyon
        # offseti gerekir. (Not: SNR / işgal BW / düzlük fark-tabanlı olduğu için bu değişimden
        # etkilenmez; yalnızca mutlak seviye ölçeği düzelir.)
        # Mutlak dBm için kalibrasyon ofseti eklenir (cal_offset_db=0 iken ham dBFS kalır).
        # SNR/işgal-BW/düzlük fark-tabanlı olduğu için ofsetten ETKİLENMEZ; yalnızca mutlak ölçek.
        power_dbfs = 10 * np.log10(power + 1e-12) + self.cal_offset_db  # 1e-12 log(0) hatasını önler

        # HAFİF zaman yumuşatma (EMA alpha=0.6): Welch zaten frekans-ekseninde pürüzsüzlük
        # sağlar; ağır EMA (0.75/0.25) grafiği "donduruyor" ve aralıklı sinyalleri söndürüyordu.
        # Hafif EMA, minik titremeyi alır ama grafiğin CANLI akmasını ve sinyal tepelerinin
        # görünmesini bozmaz. (Tuple API korunuyor: hem yumuşatılmış hem ham dönüş.)
        if not hasattr(self, '_fft_smoothed') or self._fft_smoothed.shape != power_dbfs.shape:
            self._fft_smoothed = power_dbfs
        else:
            self._fft_smoothed = 0.4 * self._fft_smoothed + 0.6 * power_dbfs

        return self._fft_smoothed, power_dbfs

    # NOT: Eski compute_aoa_from_phase (sahte 2-kanal faz-DF) ve estimate_target_position (2B km
    # nirengi) KALDIRILDI. Yön bulma artık backend/direction_finding.py'de gerçek genlik-tabanlı DF
    # + 3B LOB üçgenleme ile yapılıyor; bu metotlar ölü koddu ve kaldırıldı.

    def compute_channel_power_dbm(self, iq_samples: np.ndarray) -> float:
        """
        Bir alım (RX) kanalının ortalama sinyal gücünü dBFS cinsinden hesaplar.
        Genlik Tabanlı Yön Bulma (5.1.4) ve Genlik Kıyaslama paneli için kullanılan
        gerçek zamanlı, ölçülen (rastgele üretilmemiş) güç değeridir.

        Not: Değer kalibre edilmemiş BAĞIL dBFS'tir (referans: ADC tam skalası). Önceki
        sürümdeki "+30" sahte-dBm ofseti kaldırıldı; mutlak dBm için bilinen bir kaynakla
        tek-nokta kalibrasyon gerekir. (Metod adı tarihsel nedenle "_dbm".)
        Genlik KARŞILAŞTIRMASI (DF) fark-tabanlı olduğu için bu değişimden etkilenmez.
        """
        power = np.mean(np.abs(iq_samples) ** 2)
        power_dbfs = 10 * np.log10(power + 1e-12) + self.cal_offset_db  # kalibrasyon ofseti (varsa)
        return round(float(power_dbfs), 1)

    def detect_signals(self, fft_dbm: np.ndarray, noise_floor, center_mhz: float,
                       bandwidth_mhz: float, thresh_db: float = 10.0, min_width_bins: int = 3):
        """SİNYAL TESPİTİ (5.1.1) — DAR + GENİŞ BANT birlikte. Verilen GÜRÜLTÜ TABANINI (tarihsel-min,
        self-masking'e bağışık) thresh_db aşan BİTİŞİK ENERJİ ADALARINI bulur. Her ada bir sinyaldir:
        sivri iğne (dar telsiz) küçük ada, düz masa (LTE/Wi-Fi) büyük ada. Uzman eleştirisi #1+#2'nin
        çözümü: medyan yerine tarihsel-min taban + tepe-arama yerine enerji-adası (bant genişliği verir).
        Dönüş: [(freq_mhz, power_dbfs, snr_db, bw_mhz)] (güce göre azalan)."""
        fft = np.asarray(fft_dbm, dtype=float)
        nf = np.asarray(noise_floor, dtype=float)
        n = len(fft)
        if n < 8 or nf.shape != fft.shape:
            return []
        thr = nf + float(thresh_db)
        above = fft > thr
        nf_med = float(np.median(nf))
        out = []
        i = 0
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                if (j - i) >= int(min_width_bins):
                    seg = fft[i:j]
                    idx = np.arange(i, j)
                    w = np.power(10.0, (seg - nf[i:j]) / 10.0)     # enerji ağırlığı (doğrusal)
                    centroid = float(np.sum(idx * w) / (np.sum(w) + 1e-12))
                    freq = center_mhz + (centroid - n / 2.0) / n * bandwidth_mhz
                    bw = (j - i) / n * bandwidth_mhz
                    peak = float(np.max(seg))
                    out.append((round(float(freq), 4), round(peak, 1),
                                round(float(peak - nf_med), 1), round(float(bw), 4)))
                i = j
            else:
                i += 1
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def detect_peaks(self, fft_dbm: np.ndarray, center_mhz: float, bandwidth_mhz: float,
                     thresh_db: float = 10.0, min_sep_bins: int = 10):
        """SİNYAL TESPİTİ (şartname 5.1.1): FFT güç spektrumunda (dBFS) gürültü tabanını (medyan)
        thresh_db AŞAN YEREL TEPELERİ bulur ve mutlak frekanslarını hesaplar. GERÇEK ölçülen
        spektrumdan çalışır — hiçbir tepe uydurulmaz; sinyal yoksa boş liste döner.
        Dönüş: [(freq_mhz, power_dbfs, snr_db)] (güce göre azalan)."""
        fft = np.asarray(fft_dbm, dtype=float)
        n = len(fft)
        if n < 8:
            return []
        noise = float(np.median(fft))
        thr = noise + float(thresh_db)
        peaks = []
        b = 1
        while b < n - 1:
            # yerel maksimum + eşik üstü
            if fft[b] >= thr and fft[b] >= fft[b - 1] and fft[b] > fft[b + 1]:
                peaks.append(b)
                b += max(1, int(min_sep_bins))     # yakın bin'leri atla (aynı sinyal)
            else:
                b += 1
        out = []
        for pb in peaks:
            freq = center_mhz + (pb - n / 2.0) / n * bandwidth_mhz
            out.append((round(float(freq), 4), round(float(fft[pb]), 1), round(float(fft[pb] - noise), 1)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def analyze_spectrum(self, fft_dbm: np.ndarray, bandwidth_hz: float, noise_floor=None,
                         present_db: float = 6.5) -> dict:
        """FFT güç spektrumundan GERÇEK, ölçülen sinyal parametrelerini çıkarır (şartname 5.1.2).
        DÜŞÜK-SNR SAĞLAMLIĞI (uzman eleştirileri): gürültü tabanı verilirse (tarihsel-min) kullanılır;
        varlık eşiği 3 dB'e indirildi (şelalede görülen zayıf iz artık 'yok' sayılmaz); işgal bant
        genişliği KÜMÜLATİF ENERJİ yerine EŞİK-ÜSTÜ bin YAYILIMINDAN ölçülür (gürültü şişmesi yok).

        Dönüş: occupied_bw_hz, flatness, snr_db, signal_class, weak (marjinal SNR bayrağı)."""
        n = len(fft_dbm)
        peak_db = float(np.max(fft_dbm))
        # Gürültü tabanı: tarihsel-min verildiyse onu, yoksa DÜŞÜK PERSANTİL (%15) kullan (self-masking).
        if noise_floor is not None and len(noise_floor) == n:
            nf = np.asarray(noise_floor, dtype=float)
            noise_db = float(np.median(nf))
        else:
            noise_db = float(np.percentile(fft_dbm, 15))
            nf = np.full(n, noise_db)
        snr_db = peak_db - noise_db

        # VARLIK EŞİĞİ = present_db (uzman #1). Tek-kare spektrumda gürültünün doğal tepe-taban
        # farkı ~5-6 dB olduğu için tek karede bunun altını GÜVENİLİR ayırt edemeyiz (yanlış-pozitif).
        # Worker ZAMAN-ORTALAMALI spektrum besleyip present_db'yi düşürür (gürültü varyansı azalır ->
        # zayıf ama KALICI sinyal ortaya çıkar; şelalenin gözle yaptığı temporal integrasyon).
        if snr_db < present_db:
            return {"occupied_bw_hz": 0.0, "flatness": None, "snr_db": round(snr_db, 1),
                    "signal_class": "Sinyal Yok / Gürültü Tabanı", "weak": False}

        # İŞGAL BANT GENİŞLİĞİ: iki taraflı eşik ile ölçülür.
        #  - Düşük SNR: gürültü_tabanı + BW_THRESH_DB (gürültüyü saymaz; kümülatif enerji şişmezdi).
        #  - Yüksek SNR: tepe − OCC_PEAK_REF_DB. GÜÇLÜ sinyalde (tepe gürültüden ~40 dB yüksek) tek
        #    'taban+6' eşiği tepe−34 dB'de kalıp FM derin etekleri/pencere yan-loblarını sayarak BW'yi
        #    ~20× ŞİŞİRİYORDU (NBFM 13 kHz -> 300 kHz). Tepe−20 dB referansı gerçek işgal bandını verir.
        BW_THRESH_DB = 6.0
        OCC_PEAK_REF_DB = 20.0
        bin_hz = bandwidth_hz / n
        occ_thresh = np.maximum(nf + BW_THRESH_DB, peak_db - OCC_PEAK_REF_DB)
        above = fft_dbm > occ_thresh
        if np.any(above):
            idx = np.flatnonzero(above)
            occ_hz = float((idx[-1] - idx[0] + 1) * bin_hz)
            band = fft_dbm[above]
        else:
            occ_hz = 0.0
            band = fft_dbm

        # Spektral düzlük (yalnızca sinyal bin'leri üzerinde): dar/tonal düşük, gürültü-benzeri yüksek.
        p = np.power(10.0, (band - peak_db) / 10.0)
        gm = float(np.exp(np.mean(np.log(p + 1e-20))))
        am = float(np.mean(p))
        flatness = gm / am if am > 0 else 1.0

        weak = snr_db < (present_db + 5.0)
        if weak:
            # Marjinal SNR: Analog/Sayısal kararı güvenilmez -> dürüstçe 'zayıf sinyal, şelaleyi izleyin'.
            signal_class = "Zayıf Sinyal (marjinal SNR — şelaleyi izleyin)"
        else:
            # Heuristik eşik: dar/tonal (CW/FM/FSK vb) düşük düzlük; gürültü-benzeri
            # geniş bant (OFDM/DSSS/gürültü) yüksek düzlük verir. Bu oran YALNIZCA bant
            # profilini verir; Analog/Sayısal ayrımı modülasyon sınıflandırıcıda yapılır.
            signal_class = "Dar Bant Profil" if flatness < 0.25 else "Geniş Bant Profil"

        return {"occupied_bw_hz": occ_hz, "flatness": round(flatness, 3),
                "snr_db": round(snr_db, 1), "signal_class": signal_class, "weak": weak}


class NoiseFloorTracker:
    """GÜRÜLTÜ TABANI = TARİHSEL MİNİMUM (min-hold + yavaş yükseliş). Sinyal tüm pencereyi kaplasa
    bile önceki düşük seviyeyi 'hatırlar' -> SELF-MASKING (kendini kör etme) önlenir (uzman #1).

    Her bin için: taban = min(önceki_taban + leak, güncel). Bir sinyal belirince taban eski DÜŞÜK
    değerde kalır (min-hold) ve yalnızca sinyal kalıcıysa çok yavaş (leak) yükselir. Sinyal çekilince
    hemen aşağı iner. Böylece dar telsiz de, ekranı kaplayan dev Wi-Fi de eşiğin ÜSTÜNDE görünür."""

    def __init__(self, leak_db: float = 0.3, spatial_pct: float = 20.0):
        self.leak_db = float(leak_db)
        self.spatial_pct = float(spatial_pct)   # anlık uzamsal taban (dar-bant ilk karede yakalansın)
        self.nf = None

    def update(self, fft_dbm: np.ndarray) -> np.ndarray:
        f = np.asarray(fft_dbm, dtype=float)
        spatial = float(np.percentile(f, self.spatial_pct))   # bandın düşük persantili = anlık taban
        if self.nf is None or self.nf.shape != f.shape:
            self.nf = np.minimum(f, spatial)
        else:
            self.nf = np.minimum(self.nf + self.leak_db, f)
        # Taban, uzamsal düşük persantili ASLA aşamaz: dar-bant sinyal (band çoğu boş -> persantil=
        # gürültü) ilk karede yakalanır; geniş-bant self-masking ise min-hold ile zamanla çözülür.
        self.nf = np.minimum(self.nf, spatial)
        return self.nf

    def floor(self):
        return self.nf

    def reset(self):
        self.nf = None


class DFAccuracyTracker:
    """
    Yön Bulma (DF) doğruluğunu, bilinen bir referans (kalibrasyon) kerterizine göre
    izleyen yardımcı sınıf. Şartname Madde 5.1.4'te tanımlanan "Derece RMS" doğruluk
    metriğini üretir: RMS, ölçülen yön değerleri ile gerçek/bilinen yön arasındaki
    hataların karelerinin ortalamasının karekökü olarak tanımlanır.

    Kullanım: Sahaya bilinen bir azimuttaki bir kalibrasyon vericisi (referans sinyal
    kaynağı) yerleştirilip `set_reference()` ile beklenen açı girilir; sistem her yeni
    ölçümü `add_measurement()` ile bu referansla karşılaştırıp RMS hatasını günceller.
    Bu değer, Kritik Tasarım Raporu'nda DF doğruluğunun kanıtlanması için kullanılabilir.
    """

    def __init__(self, max_samples: int = 200):
        self.reference_deg: float | None = None
        self._errors: list[float] = []
        self.max_samples = max_samples

    def set_reference(self, reference_deg: float) -> None:
        """Kalibrasyon referans açısını (derece) ayarlar ve geçmiş hata örneklerini sıfırlar."""
        self.reference_deg = reference_deg % 360.0
        self._errors.clear()

    def clear_reference(self) -> None:
        """Kalibrasyon modunu kapatır."""
        self.reference_deg = None
        self._errors.clear()

    def add_measurement(self, measured_deg: float) -> None:
        """Yeni bir ölçülen açı ekler (yalnızca bir referans tanımlıysa etkilidir)."""
        if self.reference_deg is None:
            return
        # -180..+180 aralığına sarılmış açısal hata (360° dönüşü doğru ele alır)
        diff = (measured_deg - self.reference_deg + 180.0) % 360.0 - 180.0
        self._errors.append(diff)
        if len(self._errors) > self.max_samples:
            self._errors.pop(0)

    def rms_error_deg(self) -> float | None:
        """Şu ana kadarki ölçümlerin RMS hatasını (derece) döndürür; veri yoksa None."""
        if not self._errors:
            return None
        return round(float(np.sqrt(np.mean(np.square(self._errors)))), 2)

    def sample_count(self) -> int:
        return len(self._errors)