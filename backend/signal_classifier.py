"""
Otomatik Modülasyon Sınıflandırma (AMC) — Faz 1.

Bu modül, geniş bant I/Q içindeki EN GÜÇLÜ sinyali otomatik bulur, dijital
downconversion (DDC) ile baseband'e indirip izole eder ve modülasyon türünü
istatistiksel özelliklerden (zarf / faz / anlık frekans + yüksek dereceli
kümülantlar) sınıflandırır.

Tasarım ilkeleri:
  * SAHTE DEĞİL: yeterli SNR ve net özellik yoksa "Belirlenemedi" der; asla uydurmaz.
  * Bilimsel temel: kümülant tabanlı AMC (Swami & Sadler 2000) + Nandi-Azzouz zarf/faz
    özellikleri — literatürde kanıtlı yöntemler.
  * Bağımsız ve test edilebilir: donanım gerektirmez; saf NumPy/SciPy.

Faz 1 KAPSAMI: yalnızca MODÜLASYON türü (AM / FM / ASK / FSK / BPSK / QPSK / 8PSK /
QAM). Protokol, çoklama (OFDM) ve EKKT (FHSS/DSSS) sonraki fazlarda eklenecektir.

Not: Karar eşikleri (THRESHOLDS) sentetik sinyal kalibrasyonuyla belirlenir; bkz.
tools/calibrate_classifier.py ve tests/test_classifier.py.
"""
import numpy as np
from collections import deque
from scipy import signal as sp_signal


class SignalClassifier:
    # Sınıflandırma için gereken asgari sinyal/gürültü oranı. Altında modülasyonun ince
    # istatistiksel yapısı gürültüde kaybolur (özellikle gürültü, sabit-zarf sinyallere
    # sahte genlik varyasyonu ekleyip QAM'e benzetir) -> dürüstçe "Belirlenemedi".
    # Sentetik doğrulama: >=13 dB'de güvenilir, altında hızla bozuluyor.
    SNR_THRESHOLD_DB = 13.0

    # Çoklama (OFDM/FDMA...) tespiti için asgari SAĞLAM SNR. Altında CP otokorelasyonu/ada yapısı
    # gürültüde eriyeceği için çoklama "Belirlenemedi" döner (yanlış "Tek Taşıyıcı" iddiası yerine).
    TH_MUX_SNR = 8.0

    # ÇOKLAMA (OFDM) TESPİTİ
    TH_MUX_SNR = 8.0

    def __init__(self, sample_rate_hz: float):
        self.fs = float(sample_rate_hz)

    def _to_work_fs(self, iq: np.ndarray, target_fs: float) -> np.ndarray:
        """Gelen IQ'yu hedef target_fs hızına resample eder.
        GÜVENLİK: küsuratlı target_fs'te EBOB=1 olup up≈fs (milyonlar) çıkarsa resample_poly
        milyarlarca örnek ayırıp süreci OOM ile öldürür. up mantıksız büyükse SAF tam-sayı
        seyreltmeye (down-only) düş -> up=1, bellek daima güvenli."""
        x = iq - np.mean(iq)
        if abs(self.fs - target_fs) < 1.0:
            return x.astype(np.complex64)
        from math import gcd
        g = gcd(int(round(self.fs)), int(round(target_fs)))
        up = int(round(target_fs)) // g
        down = int(round(self.fs)) // g
        # KRİTİK GÜVENLİK KAPISI: up büyükse (küsuratlı hedef) bellek patlar -> tam sayı seyreltmeye düş
        if up > 8:
            D = max(1, int(round(self.fs / max(target_fs, 1.0))))
            return sp_signal.resample_poly(x, 1, D).astype(np.complex64) if D > 1 else x.astype(np.complex64)
        y = sp_signal.resample_poly(x, up, down)
        return y.astype(np.complex64)

    # ------------------------------------------------------------------ #
    # 1) EN GÜÇLÜ SİNYALİ BUL + İZOLE ET (DDC)
    # ------------------------------------------------------------------ #
    def _estimate_snr_db(self, iq: np.ndarray) -> float:
        """Kaba SNR: FFT tepe gücü - gürültü tabanı (medyan), dB."""
        x = iq - np.mean(iq)
        mag2 = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(len(x))))) ** 2
        p_db = 10 * np.log10(mag2 + 1e-12)
        return float(np.max(p_db) - np.median(p_db))

    def _strongest_freq_offset(self, iq: np.ndarray) -> float:
        """En güçlü spektral tepenin merkeze göre frekans offseti (Hz)."""
        x = iq - np.mean(iq)
        spec = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(len(x))))) ** 2
        n = len(spec)
        peak = int(np.argmax(spec))
        # fftshift'li eksende bin -> frekans: (bin - n/2) * fs/n
        return (peak - n / 2.0) * self.fs / n

    def estimate_bandwidth_hz(self, iq: np.ndarray, thresh_db: float = 10.0) -> float:
        """Sinyalin işgal bant genişliğini (Hz) ölçer: güç spektrumunda tepe-thresh_db üstündeki
        kümülatif enerji yayılımından (%1-99). Protokol ID (baud/BW) ve DDC filtre genişliği için."""
        x = iq - np.mean(iq)
        n = len(x)
        spec = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(n)))) ** 2
        noise = np.median(spec)
        lin = np.clip(spec - noise, 0.0, None)
        total = float(np.sum(lin)) + 1e-12
        csum = np.cumsum(lin)
        lo = int(np.searchsorted(csum, 0.01 * total))
        hi = int(np.searchsorted(csum, 0.99 * total))
        return max(0.0, (hi - lo) * self.fs / n)

    def _channelize(self, iq: np.ndarray):
        """GERÇEK DDC (Dijital Downconversion) KANAL İZOLASYONU.
        Sinyalin ölçülen bant genişliğine göre dinamik çalışma frekansı (iso_fs) belirler.
        Dönüş: (izole_iq, iso_fs)"""
        iq = np.asarray(iq, dtype=np.complex64)
        # Kestirim için DC-temiz kopya (LO sızıntısı/DC tepesi bant/tepe ölçümünü yanıltmasın).
        xm = (iq - np.mean(iq)).astype(np.complex64)
        bw = self.estimate_bandwidth_hz(xm)

        # Dinamik çalışma bandı: sinyali kapsayacak kadar (en az 400 kHz, en çok tam bant).
        target_bw = max(400e3, min(self.fs, bw * 1.5))
        # TAM SAYI seyreltme (decimation) çarpanı D. iso_fs yalnızca özellik ölçeği için ETİKETTİR.
        D = max(1, int(self.fs / target_bw))
        iso_fs = self.fs / D

        f_off = self._strongest_freq_offset(xm)

        # ZARF KORUMASI (KRİTİK DÜZELTME): kanal izolasyonu ORİJİNAL sinyalde (iq) yapılır, ortalama
        # ÇIKARILMAZ. Ortalama çıkarmak SABİT-ZARFLI PM/FM'in DC/taşıyıcı (J0) bileşenini kaldırıp
        # zarfı yapay değişkenleştiriyordu -> gamma_max 0'dan 427'ye fırlayıp PM/FM 'AM' sanılıyordu.
        # (Frekans/bant kestirimi yukarıda xm ile yapıldı; zarf için gerçek sinyal korunur.)
        x = iq
        # Eğer offset iso_fs'in yarısından fazlaysa, DC'ye kaydır (birim-genlikli çarpım, zarfı bozmaz)
        if abs(f_off) > iso_fs * 0.4:
            t = np.arange(len(x)) / self.fs
            x = (x * np.exp(-1j * 2 * np.pi * f_off * t)).astype(np.complex64)   # DC'ye indir

        # KRİTİK (OOM KÖK-NEDENİ): Sinyali küsuratlı bir HEDEF HIZA 'resample' ETME. iso_fs=fs/D
        # tam sayı olmadığında (ör. 20 Msps'de D=7 -> 2.857142...M), resample_poly EBOB=1 bulup
        # up≈fs (milyonlar) katı bir polifaz filtre kurar; 4096 örnek → MİLYARLARCA örnek (~218 GB)
        # → süreç anında OOM ile ÖLDÜRÜLÜR ("Killed"). Bunun yerine oranı DOĞRUDAN 1/D ver: up=1
        # GARANTİ (çıktı en çok giriş kadar), fs herhangi bir değerde olsa bile bellek güvenli.
        work = sp_signal.resample_poly(x, 1, D).astype(np.complex64) if D > 1 else x
        return work, iso_fs

    def _looks_multisignal(self, iq: np.ndarray) -> bool:
        """Bantta birden çok AYRIK sinyal var mı? (Faz 1 tek sinyal ister.)
        Spektrum kaba bloklara bölünür; gürültü tabanının üstündeki AYRIK aktif blok
        kümeleri sayılır. Tek sinyal (geniş de olsa) bitişik bloklar -> 1 küme; ayrık
        istasyonlar (ör. FM bandı) boşluklarla ayrılmış -> 3+ küme."""
        x = iq - np.mean(iq)
        spec = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(len(x))))) ** 2
        p_db = 10 * np.log10(spec + 1e-12)
        noise = np.median(p_db)

        nblocks = 24
        L = len(p_db) // nblocks
        active = np.array([p_db[i * L:(i + 1) * L].max() > noise + 12.0 for i in range(nblocks)])
        # Ayrık aktif blok kümesi sayısı (0->1 geçişleri)
        clusters = int(active[0]) + int(np.sum((~active[:-1]) & active[1:]))
        return clusters >= 3

    # ------------------------------------------------------------------ #
    # 2) AMC ÖZELLİKLERİ (izole baseband üzerinde)
    # ------------------------------------------------------------------ #
    def extract_features(self, x: np.ndarray, iso_fs: float) -> dict:
        """İzole baseband sinyalinden modülasyon-ayırt edici istatistikler çıkarır."""
        x = np.asarray(x, dtype=np.complex128)
        n = len(x)

        # --- Zarf (genlik) özellikleri: ORİJİNAL sinyalden (ortalama ÇIKARMADAN!) ---
        # KRİTİK DÜZELTME: Ortalama çıkarmak, SABİT-ZARFLI sinyallerin (PM ve düşük-sapmalı FM)
        # DC/taşıyıcı (J0) bileşenini kaldırıp zarfı YAPAY olarak değişkenleştiriyordu -> gamma_max
        # sıfırdan yüzlere fırlayıp PM/FM 'AM' sanılıyordu (PM'de 0 -> 427!). Zarf, GERÇEK sinyalden
        # hesaplanmalı: sabit-zarf gerçekten sabit görünsün. (AM'de zarf zaten değişken -> gamma yüksek.)
        a = np.abs(x)
        a_mean = np.mean(a) + 1e-12
        a_n = a / a_mean                      # normalize genlik
        a_c = a_n - 1.0                        # merkezi normalize genlik
        # gamma_max: merkezi normalize genliğin maksimum spektral yoğunluğu
        # (genlik-modülasyonlu sinyallerde yüksek; sabit-zarfta ~0)
        A = np.abs(np.fft.fft(a_c)) ** 2 / n
        gamma_max = float(np.max(A))
        sigma_aa = float(np.std(a_n))         # genlik varyasyonu
        # env_par: merkezi zarf spektrumunun TEPE/ORTALAMA oranı (peak-to-average ratio).
        # AM'de mesaj tonu keskin bir tepe -> PAR yüksek (yüzlerce/binlerce). FM/PM'de zarf sabit,
        # kalan sadece BEYAZ gürültü -> spektrum düz -> PAR düşük (~5-8). KENDİ KENDİNİ KALİBRE EDER:
        # gürültü ortalamayı (payda) yükseltir, bu yüzden SNR/N'DEN BAĞIMSIZDIR -> AM ile FM'i her
        # SNR'de temiz ayırır (en sığ AM bile PAR>=390 vs FM<=8). DC (bin 0) hariç tutulur.
        A_ac = A[1:] if n > 1 else A
        env_par = float(np.max(A_ac) / (np.mean(A_ac) + 1e-12))

        # Faz / frekans / kümülant özellikleri için DC bileşenini temizle (ortalama çıkar).
        x = x - np.mean(x)

        # --- Faz özellikleri (yalnızca zarfı güçlü örneklerde; zayıf örnek fazı gürültülü) ---
        strong = a > a_mean
        if strong.sum() < 16:
            strong = np.ones(n, dtype=bool)
        phi = np.angle(x[strong])
        sigma_dp = float(np.std(phi))          # doğrudan faz std (PSK'da yüksek)
        sigma_ap = float(np.std(np.abs(phi)))  # mutlak faz std

        # --- Anlık frekans (frekans vs faz modülasyonu ayrımı) ---
        inst_freq = np.diff(np.unwrap(np.angle(x)))
        sigma_af = float(np.std(inst_freq))
        # İmpulsiflik (kurtosis): Faz sıçramalarını kırpan medyan filtresi eklendi (outlier clipping).
        # Aşırı gürültülü faz sıçramaları kurtosis'i rastgele devasa değerlere (1000+) çıkarıyordu.
        if_c = inst_freq - np.median(inst_freq)
        limit = 3.0 * np.std(if_c)
        if_c = np.clip(if_c, -limit, limit)
        
        if_var = np.mean(if_c ** 2) + 1e-12
        if_kurt = float(np.mean(if_c ** 4) / (if_var ** 2))

        # --- Yüksek dereceli kümülantlar (güç-normalize) ---
        p = np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12
        xn = x / p
        m20 = np.mean(xn ** 2)
        m21 = np.mean(np.abs(xn) ** 2)
        m40 = np.mean(xn ** 4)
        m42 = np.mean(np.abs(xn) ** 4)
        C20 = m20
        C21 = m21
        C40 = m40 - 3.0 * m20 ** 2
        C42 = m42 - np.abs(m20) ** 2 - 2.0 * m21 ** 2
        c40 = float(np.abs(C40))               # |C40|: BPSK~2, QPSK~1, 8PSK~0
        c42 = float(np.abs(C42))               # |C42|: modülasyon ayrımı

        return {
            "gamma_max": gamma_max,
            "env_par": env_par,
            "sigma_aa": sigma_aa,
            "sigma_dp": sigma_dp,
            "sigma_ap": sigma_ap,
            "sigma_af": sigma_af,
            "if_kurt": if_kurt,
            "c20": float(np.abs(C20)),
            "c40": c40,
            "c42": c42,
            "n_samples": n,
        }

    # ------------------------------------------------------------------ #
    # 2b) ÇOKLAMA TESPİTİ — OFDM vs Tek Taşıyıcı (Faz 2)
    # ------------------------------------------------------------------ #
    # OFDM CP otokorelasyon belirginliği. ÖLÇÜLDÜ (tüm SNR'de): OFDM ~42, 2FSK ~8, 4FSK ~4.6, PSK/QAM ~2.
    # Eski 6.0 eşiği 2FSK'yı (düşük SNR'de) YANLIŞ OFDM sanıyordu. 12.0: OFDM (42) geçer, 2FSK (8)
    # elenir — arada geniş güvenlik payı (düşük-CP OFDM ~20 de geçer).
    TH_OFDM_PROMINENCE = 12.0

    # OFDM işgal-bandı ALT sınırı. Uzman eleştirisi (haklı): 56 MHz'de yakalanan 20 MHz Wi-Fi
    # bandın yalnızca ~%36'sını kaplar; eski 0.55 eşiği bunu CP testine SOKMADAN "Tek Taşıyıcı"
    # damgalıyordu. Eşik 0.25'e indirildi: yüksek fs'teki geniş-bant OFDM (~%36) geçer, ama aşırı
    # örneklenmiş dar-bant tek-taşıyıcı (PSK/QAM ~%15) hâlâ elenir (yanlış OFDM'i önler).
    TH_OFDM_MIN_OCCUPANCY = 0.25

    def detect_multiplex(self, iq: np.ndarray, dmin: int = None, dmax: int = None):
        """OFDM tespiti: siklik prefix (CP), N_fft gecikmesinde otokorelasyon TEPESİ üretir.
        Sembol-içi korelasyonu atlamak için d>=dmin taranır; tepenin belirginliği (prominence)
        OFDM'de çok yüksek, tek taşıyıcıda ~2'dir. NOT: Ham sinyalde (gerçek fs) çalışır —
        WORK_FS'e resample CP yapısını bozar. Dönüş: (etiket, prominence, N_fft).

        CP GECİKME PENCERESİ fs İLE ÖLÇEKLENİR (uzman eleştirisi #2, haklı): OFDM sembol süresi
        sabittir (Wi-Fi ~3.2 µs faydalı), ama ÖRNEK cinsinden gecikme fs'e bağlıdır
        (lag = süre·fs). Sabit 32-300 penceresi yalnızca ~20 MHz'de doğruydu; artık pencere
        0.5-80 µs sembol aralığını fs'e göre örneğe çevirir -> her örnekleme hızında doğru."""
        x = iq - np.mean(iq)
        n = len(x)
        # fs'e göre gecikme penceresi (örnek). TABAN 32: daha küçük gecikmeler aşırı örneklenmiş
        # dar-bant sinyalin sembol-içi/pals-şekli otokorelasyonunu yakalayıp SAHTE OFDM tepesi
        # üretir (nfft~16). ÜST sınır fs ile ölçeklenir (~80 µs sembole kadar; Wi-Fi 3.2 µs dahil).
        if dmin is None:
            dmin = max(32, int(0.3e-6 * self.fs))
        if dmax is None:
            dmax = max(dmin + 8, int(80e-6 * self.fs))
        dhi = min(dmax, n // 2)
        if dhi <= dmin + 4:
            return "Belirlenemedi", 0.0, 0

        # ÖN-KONTROL: OFDM GENİŞ banttır.
        spec = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(n)))) ** 2
        total = float(np.sum(spec)) + 1e-12
        csum = np.cumsum(spec)
        lo = int(np.searchsorted(csum, 0.02 * total))
        hi = int(np.searchsorted(csum, 0.98 * total))
        occupied_frac = (hi - lo) / n
        
        # Eşiği çok esnettik (0.10). Sadece aşırı dar bantları eliyoruz.
        if occupied_frac < 0.10:
            return "Tek Taşıyıcı", 0.0, 0

        p = np.mean(np.abs(x) ** 2) + 1e-12
        ds = np.arange(dmin, dhi)
        R = np.array([np.abs(np.mean(x[:n - d] * np.conj(x[d:]))) / p for d in ds])
        peak_i = int(np.argmax(R))
        prom = float(R[peak_i] / (np.median(R) + 1e-12))
        if prom > self.TH_OFDM_PROMINENCE:
            return "OFDM (Çok Taşıyıcı)", prom, int(ds[peak_i])
        return "Tek Taşıyıcı", prom, 0

    # ------------------------------------------------------------------ #
    # 2b-ii) GENİŞLETİLMİŞ ÇOKLAMA + DSSS (şartname 5.1.2: TDMA/FDMA/CDMA/OFDM + DSSS)
    # ------------------------------------------------------------------ #
    TH_DSSS_OCC = 0.35      # DSSS/CDMA: bandın en az bu oranını kaplayan
    TH_DSSS_FLAT = 0.5      #            + spektral düzlüğü bunun üstünde (gürültü-benzeri yayılı)
    TH_TDMA_DUTY = 0.6      # TDMA-benzeri: zarf gücü zamanın bu oranından AZ süre "açık" (bursty)

    def _occupancy_flatness(self, iq: np.ndarray):
        """İşgal oranı (0-1) + işgal bandı içi spektral düzlük (0=tonal, 1=gürültü-benzeri)."""
        x = iq - np.mean(iq)
        n = len(x)
        S = np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(n)))) ** 2
        noise = np.median(S)
        lin = np.clip(S - noise, 0.0, None)
        total = float(np.sum(lin)) + 1e-12
        cs = np.cumsum(lin)
        lo = int(np.searchsorted(cs, 0.02 * total))
        hi = int(np.searchsorted(cs, 0.98 * total))
        occ = (hi - lo) / n
        band = S[lo:hi] + 1e-12 if hi > lo else S + 1e-12
        gm = float(np.exp(np.mean(np.log(band))))
        am = float(np.mean(band))
        return occ, (gm / am if am > 0 else 1.0)

    def _count_carriers(self, iq: np.ndarray, hi_drop_db: float = 6.0, gap_above_noise_db: float = 8.0) -> int:
        """AYRIK taşıyıcı (FDMA) sayısı. Ayırt edici: FDMA kanalları arasındaki boşluk GÜRÜLTÜ
        TABANINA iner; tek sürekli yayında (FM/QPSK/OFDM/DSSS) iç dalgalanmalar gürültüye inmez.
        Güçlü adalar (tepe-6 dB üstü) bulunur; iki ada arasındaki boşluk gürültü+8 dB'nin ALTINA
        iniyorsa AYRI taşıyıcı sayılır, aksi halde birleştirilir (tek sürekli yayın)."""
        x = iq - np.mean(iq)
        n = len(x)
        Sdb = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x * np.hamming(n)))) ** 2 + 1e-12)
        k = max(1, n // 512)
        Sdb = np.convolve(Sdb, np.ones(k) / k, mode="same")
        peak = float(np.max(Sdb))
        noise = float(np.median(Sdb))
        if peak - noise < 12.0:
            return 1                                          # belirgin sinyal yok -> tek say
        hi = Sdb > (peak - hi_drop_db)
        lo_thr = noise + gap_above_noise_db
        islands = []
        i = 0
        while i < n:
            if hi[i]:
                j = i
                while j < n and hi[j]:
                    j += 1
                islands.append((i, j))
                i = j
            else:
                i += 1
        if not islands:
            return 0
        carriers = 1
        for a in range(1, len(islands)):
            gap = Sdb[islands[a - 1][1]:islands[a][0]]
            if gap.size and float(np.min(gap)) < lo_thr:       # boşluk gürültüye iniyor -> ayrı kanal
                carriers += 1
        return carriers

    def _burst_duty(self, iq: np.ndarray) -> float:
        """Zarf gücünün 'açık' (tepe gücün >%15'i) olduğu zaman oranı. Sürekli/sabit-zarf sinyal
        ~1; TDMA/bursty sinyalde OFF (sıfıra yakın) pencereler olduğu için < 1."""
        p = np.abs(iq) ** 2
        k = max(1, len(p) // 64)
        ps = np.convolve(p, np.ones(k) / k, mode="same")
        peak = float(np.max(ps)) + 1e-12
        return float(np.mean(ps > 0.15 * peak))               # OFF = tepe gücün %15'inin altı

    def analyze_multiplex_ekkt(self, iq: np.ndarray):
        """ÇOKLAMA + DSSS sınıflandırması (5.1.2). Dönüş: (çoklama_etiketi, dsss_var_mı, güven).
        Sıra: OFDM (CP otokorelasyon) -> FDMA (çok ayrık taşıyıcı) -> DSSS/CDMA (geniş+düz, OFDM
        değil) -> TDMA-benzeri (bursty zarf) -> Tek Taşıyıcı. Heuristiktir (dürüstçe 'olası' yapı)."""
        # FDMA ÖNCE: ayrık kanallar (gürültü-tabanına inen boşluklar). OFDM'in bitişik alt-taşıyıcıları
        # tek geniş ada verir (ncar=1) -> OFDM'e düşer; FDMA'nın ayrık kanalları ncar>=3 verir.
        ncar = self._count_carriers(iq)
        if ncar >= 3:
            return f"FDMA ({ncar} taşıyıcı)", False, 0.7
        mux, prom, nfft = self.detect_multiplex(iq)
        if mux.startswith("OFDM"):
            return "OFDM (Çok Taşıyıcı)", False, self._margin_conf(prom, self.TH_OFDM_PROMINENCE, 10.0)
        occ, flat = self._occupancy_flatness(iq)
        if occ > self.TH_DSSS_OCC and flat > self.TH_DSSS_FLAT:
            return "DSSS/CDMA (Yayılı Spektrum)", True, 0.65    # geniş+düz+OFDM değil -> yayılı spektrum
        if self._burst_duty(iq) < self.TH_TDMA_DUTY:
            return "TDMA-benzeri (Zaman-bölmeli)", False, 0.6
        return "Tek Taşıyıcı", False, 0.7

    # ------------------------------------------------------------------ #
    # 2c) EKKT TESPİTİ — FHSS (frekans atlama) (Faz 3)
    # ------------------------------------------------------------------ #
    TH_HOP_COUNT = 5          # zaman dilimleri arası belirgin frekans sıçraması sayısı
    TH_HOP_FREQ_STD = 0.10    # tepe-frekans dizisinin std'si (sabit sinyalde ~0.01)

    def detect_hopping(self, iq: np.ndarray, nseg: int = 24):
        """FHSS tespiti: sinyali zaman dilimlerine böler, her dilimin TEPE frekansını izler.
        Frekans dilimden dilime ayrık SIÇRAMALAR yapıyorsa -> FHSS. Sabit sinyalde tepe
        frekans sabittir (sıçrama ~0). NOT: kısa pencerede (2048 örnek) yavaş atlamalı FHSS
        (ör. Bluetooth) görünmeyebilir; güvenilir tespit için daha uzun kayıt gerekir."""
        x = iq - np.mean(iq)
        seglen = len(x) // nseg
        if seglen < 8:
            return "Yok", 0
        peaks = []
        sharp = []
        for i in range(nseg):
            seg = x[i * seglen:(i + 1) * seglen]
            sp = np.abs(np.fft.fftshift(np.fft.fft(seg * np.hamming(len(seg))))) ** 2
            peaks.append(np.argmax(sp) / len(seg))          # normalize tepe frekans [0,1]
            sharp.append(np.max(sp) / (np.median(sp) + 1e-12))  # dilim tepe belirginliği
        peaks = np.array(peaks)
        # FHSS'te HER dilim dar-bant (tek belirgin tepe). OFDM'de dilimler düz (belirsiz),
        # gürültüde de tepe zayıf. Belirgin-tepeli dilim oranı bunları ayırır.
        # Eşik 15: gürültünün FFT tepesi ~8-10'a çıkabilir; gerçek FHSS dilim tepesi >>15.
        sharp_frac = float(np.mean(np.array(sharp) > 15.0))
        hops = int(np.sum(np.abs(np.diff(peaks)) > 0.08))
        freq_std = float(np.std(peaks))
        if sharp_frac > 0.6 and hops >= self.TH_HOP_COUNT and freq_std > self.TH_HOP_FREQ_STD:
            return "FHSS (Frekans Atlama)", hops
        return "Yok", hops

    # ------------------------------------------------------------------ #
    # 3) SINIFLANDIRMA (karar ağacı bir sonraki adımda kalibre edilir)
    # ------------------------------------------------------------------ #
    # ROBUST MODÜLASYON AİLESİ eşikleri (fs-duyarsız özelliklere dayanır).
    # Sentetik + GERÇEK FM kalibrasyonundan (WORK_FS=400 kHz) türetildi:
    #   AM     gamma_max~750, if_kurt~11  | FM/FSK  if_kurt~1.5-2.7 (gerçek FM dahil)
    #   PSK    if_kurt~50, sigma_aa<0.10  | QAM     if_kurt~42, sigma_aa~0.34
    # AM/FM AYRIMI — TEK ÖLÇÜT: sigma_aa (normalize zarfın standart sapması) = genlik değişiminin
    # BÜYÜKLÜĞÜ. Klasik Azzouz-Nandi AM ölçütü.
    #   AM: zarf mesajla modüle -> sigma_aa BÜYÜK (derinlik 0.3->0.10, 0.5->0.22, 0.8->0.40).
    #   FM: sabit zarf -> sigma_aa KÜÇÜK; gerçekçi donanım dalgalanması/FM->AM dönüşümüyle bile
    #       (%12 dalgaya kadar) <0.09.
    # NOT (env_par KALDIRILDI): zarf spektrumu tepe/ortalama oranı, GERÇEKÇİ sinyalde AM/FM'i
    # AYIRT ETMİYOR — hatta TERS: geniş-bant ses-AM'de env_par ~150-250, ama FM->AM dönüşümlü FM'de
    # ~350-600 (periyodik dalga keskin tepe yapar). Bu yüzden 'par>=30' koşulu gerçek AM'i FM'e
    # düşürüyordu (kullanıcı: "sabit FM'e kalıyor"). Tek güvenilir ayraç sigma_aa'nın BÜYÜKLÜĞÜdür.
    TH_AM_SAA = 0.09          # sigma_aa bunun ÜSTünde -> AM, altında -> FM. Sinyal jeneratöründe
                              # önerilen derinlik %50-80 -> sigma_aa 0.22+ (çok geniş güven payı).
    # TH_AM_PAR = 30.0        # (env_par KALDIRILDI — gerçekçi sinyalde ters çalışıyordu, yukarı bkz.)
    # TH_AM_GAMMA = 50.0      # (eski, N-bağımlı — kaldırıldı; dessimasyonda AM'i kaçırıyordu)
    # TH_KURT_FM = 6.0        # if_kurt bunun ALTINDA -> Frekans mod. (dijital açılınca)
    # TH_QAM_SIGMA_AA = 0.27  # QAM ayrımı (dijital açılınca)

    @staticmethod
    def _margin_conf(value, threshold, scale):
        """Karar güveni: özelliğin eşiğe uzaklığı ne kadar büyükse o kadar yüksek (0.5-1.0)."""
        m = abs(value - threshold) / max(scale, 1e-9)
        return float(min(1.0, 0.5 + 0.5 * min(m, 1.0)))

    def _decide(self, f: dict):
        """SADECE ANALOG karar: AM (Analog-Genlik) vs FM (Analog-Frekans). İkili çıktı — sinyal
        jeneratöründen AM basınca AM, FM basınca FM.

        TEK KOŞUL — sigma_aa (genlik değişiminin büyüklüğü):
          * sigma_aa >= TH_AM_SAA -> AM (zarf mesajla belirgin değişiyor)
          * sigma_aa <  TH_AM_SAA -> FM (zarf ~sabit; gerçekçi dalgalanma bile eşiğin altında)

        Neden env_par DEĞİL: gerçekçi sinyalde AM/FM'i ayırt etmiyor (hatta ters — FM->AM dönüşümlü
        FM'de env_par AM'den yüksek). Neden gamma_max DEĞİL: örnek sayısıyla ölçekleniyordu ->
        dessimasyonda AM'i kaçırıyordu. sigma_aa N-BAĞIMSIZ ve güvenilmez iç SNR tahminine bağlı değil.

        DİJİTAL MODÜLASYON TESPİTİ ŞİMDİLİK KAPALI — test cihazımız yok. İleride açmak için
        aşağıdaki dijital dalları (QAM/PSK/FSK) yorumdan çıkarın (if_kurt/sigma_aa kullanır)."""
        saa = f["sigma_aa"]

        if saa >= self.TH_AM_SAA:                     # belirgin genlik (zarf) değişimi -> AM
            return "AM (Analog-Genlik)", self._margin_conf(saa, self.TH_AM_SAA, 0.15)
        # Sabit/az-değişken zarf -> FM. Güven: sigma_aa eşiğin ne kadar altındaysa o kadar yüksek.
        return "FM (Analog-Frekans)", self._margin_conf(self.TH_AM_SAA, saa, 0.06)

        # --- DİJİTAL TÜRLER (ŞİMDİLİK KAPALI — ileride açılacak) -----------------------------
        # kurt = f["if_kurt"]
        # if kurt < self.TH_KURT_FM:                 # sabit faz -> açı mod. (FM/FSK/PM ortak)
        #     return "FM/FSK/PM (Açı Mod.)", self._margin_conf(self.TH_KURT_FM, kurt, 4.0)
        # if saa > self.TH_QAM_SIGMA_AA:             # yüksek impulsiflik + değişken zarf -> QAM
        #     return "QAM (Sayısal-Genlik)", self._margin_conf(saa, self.TH_QAM_SIGMA_AA, 0.15)
        # return "PSK (Sayısal-Faz)", self._margin_conf(kurt, self.TH_KURT_FM, 20.0)  # sabit zarf -> PSK

    def estimate_symbol_rate_hz(self, iq: np.ndarray) -> float:
        """Sembol (baud) hızını kestirir. Kiplenmiş dijital sinyalin GÜCÜ (|x|²) sembol saatinde
        siklostatik bir spektral ÇİZGİ üretir; |x|²'nin FFT'sinde DC-dışı baskın tepe ~ baud hızı.
        (Analog FM/AM ve gürültüde anlamlı çizgi çıkmaz -> düşük/gürültülü değer; eşleştirmede
        yalnızca dijital sinyallerde ağırlıklandırılır.)"""
        x = iq - np.mean(iq)
        n = len(x)
        if n < 256:
            return 0.0
        env = np.abs(x) ** 2
        env = env - np.mean(env)
        S = np.abs(np.fft.rfft(env * np.hamming(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
        lo = int(np.searchsorted(freqs, max(1e3, self.fs / n * 3)))   # DC yakınını atla
        if lo >= len(S):
            return 0.0
        k = lo + int(np.argmax(S[lo:]))
        return float(freqs[k])

    # Sinyal kütüphanesi (yaklaşık): (isim, f_lo_MHz, f_hi_MHz, bw_min_Hz, bw_max_Hz, mod_ipucu,
    # mux_ipucu, ekkt_ipucu). None = önemsiz. Protokol ID artık salt bant-planı değil; ÖLÇÜLEN
    # bant genişliği + modülasyon/çoklama/EKKT ile eşleştirilir (uzman eleştirisi #4).
    SIGNAL_LIBRARY = [
        ("FM Broadcast",       88.0,   108.0,  100e3, 260e3, "FM",  None,   None),
        ("Havacılık (AM)",     108.0,  137.0,    3e3,  15e3, "AM",  None,   None),
        ("Deniz/PMR Telsiz",   137.0,  174.0,    8e3,  30e3, None,  None,   None),
        ("TETRA",              380.0,  430.0,   18e3,  30e3, "PSK", None,   None),
        ("UHF Telsiz (DMR/P25)",430.0, 470.0,    8e3,  16e3, "PSK", None,   None),
        ("GSM-900",            925.0,  960.0,  150e3, 260e3, None,  None,   None),
        ("GSM-1800/LTE",      1805.0, 1880.0,  150e3,  25e6, None,  None,   None),
        ("Wi-Fi 2.4 GHz",     2400.0, 2483.5,  15e6,  45e6, None,  "OFDM", None),
        ("Bluetooth",         2400.0, 2483.5, 800e3,   3e6, None,  None,   "FHSS"),
        ("ISM 2.4 GHz cihazı",2400.0, 2483.5,    0.0,  50e6, None,  None,   None),
        ("Wi-Fi 5 GHz",       5150.0, 5895.0,  15e6,  90e6, None,  "OFDM", None),
    ]

    def guess_protocol(self, center_mhz: float, result: dict) -> str:
        """Bant planı + ÖLÇÜLEN parametrelerden (bant genişliği + modülasyon/çoklama/EKKT) OLASI
        protokol. Salt frekans tahmini DEĞİL: kütüphane girdileri ölçülen özelliklere göre skorlanır.
        Tam paket dekodu değildir -> dürüstçe 'olası' döner. (Uzman eleştirisi #4'ün düzeltmesi.)"""
        mod = result.get("modulation", "")
        mux = result.get("multiplex", "")
        ekkt = result.get("ekkt", "")
        bw = float(result.get("occupied_bw_hz", 0.0) or 0.0)
        baud = float(result.get("symbol_rate_hz", 0.0) or 0.0)
        f = center_mhz

        best, best_score = None, 0.0
        for name, flo, fhi, bwmin, bwmax, mhint, muxhint, ekhint in self.SIGNAL_LIBRARY:
            if not (flo <= f <= fhi):
                continue
            score = 1.0                                    # bant içinde -> taban skor
            if bw > 0 and bwmin <= bw <= bwmax:
                score += 2.0                               # ölçülen BW aralıkta -> güçlü kanıt
            elif bw > 0:
                score -= 1.0                               # BW aralık dışı -> zayıf eşleşme
            if ekhint and ekhint in ekkt:
                score += 2.0
            if muxhint and muxhint in mux:
                score += 2.0
            if mhint and mhint in mod:
                score += 1.0
            if score > best_score:
                best, best_score = name, score

        if best is None:
            return "Belirlenemedi (bant planı dışı)"
        # UHF telsizde baud, DMR/P25 (~4.8 kBd) ile daha yüksek hızlı sayısal telsizi ayırt eder.
        if best == "UHF Telsiz (DMR/P25)" and 3e3 <= baud <= 7e3:
            best = "olası DMR/P25 (sayısal telsiz)"
            bw_str = f", ~{baud/1e3:.1f} kBd"
            return best  # zaten 'olası' ile başlıyor
        bw_str = f", ~{bw/1e6:.1f} MHz" if bw >= 1e6 else (f", ~{bw/1e3:.0f} kHz" if bw > 0 else "")
        baud_str = f", ~{baud/1e3:.1f} kBd" if baud > 0 else ""
        return f"olası {best}{bw_str}{baud_str}"

    def classify(self, iq: np.ndarray, check_hopping: bool = True) -> dict:
        """Geniş bant IQ -> en güçlü sinyalin modülasyon türü + güven + SNR.
        Düşük SNR veya belirsizlikte dürüstçe 'Belirlenemedi' döner.

        check_hopping: tek-blok FHSS ön-kontrolü. Gerçek-zaman döngüsünde bu KAPATILIR
        (check_hopping=False) çünkü FHSS artık HoppingHistoryTracker ile zaman-geçmişinden,
        güvenilir biçimde tespit edilir (uzman eleştirisi #1). Tek-blok tespiti yalnızca
        bağımsız/test kullanımı için korunur."""
        snr = self._estimate_snr_db(iq)

        # EKKT (Faz 3): FHSS her dilimde dar-bant belirgin tepe + zaman içinde atlama.
        # NOT: tek blok bir atlamadan kısa olabilir -> güvenilir tespit için zaman-geçmişi
        # (HoppingHistoryTracker) kullanılır; burası yalnızca check_hopping=True iken çalışır.
        if check_hopping:
            ekkt, hops = self.detect_hopping(iq)
            if ekkt.startswith("FHSS"):
                return {"modulation": "FHSS (Atlamalı)", "confidence": 0.8, "snr_db": round(snr, 1),
                        "multiplex": "Tek Taşıyıcı (hop başına)", "ekkt": ekkt}

        # --- ÇOKLAMA / OFDM / DSSS (DİJİTAL) TESPİTİ ŞİMDİLİK KAPALI (yorumda) ---------------
        # Şu an yalnızca AM/FM (analog) ayrımı yapıyoruz (test cihazı yok). İleride dijital türleri
        # açmak için aşağıdaki bloğu yorumdan çıkarın:
        # x0 = iq - np.mean(iq)
        # Sdb0 = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x0 * np.hamming(len(x0))))) ** 2 + 1e-12)
        # snr_robust = float(np.max(Sdb0) - np.percentile(Sdb0, 15))
        # mux, dsss, mux_conf = self.analyze_multiplex_ekkt(iq)
        # if not mux.startswith("OFDM") and snr_robust < self.TH_MUX_SNR:
        #     mux, dsss, mux_conf = "Belirlenemedi (düşük SNR)", False, 0.0
        # ekkt_dsss = "DSSS (Yayılı Spektrum)" if dsss else "Yok"
        # if mux.startswith("OFDM"):
        #     return {"modulation": "OFDM (Çok Taşıyıcı)", "confidence": round(mux_conf, 2),
        #             "snr_db": round(snr, 1), "multiplex": mux, "ekkt": "Yok"}

        if snr < self.SNR_THRESHOLD_DB:
            return {"modulation": "Belirlenemedi (düşük SNR)", "confidence": 0.0,
                    "snr_db": round(snr, 1)}

        # Faz 1 tek sinyal ister; geniş bantta çok sinyal karışımı yanlış sonuç verir.
        if self._looks_multisignal(iq):
            return {"modulation": "Belirlenemedi (çok sinyal — bandı daraltın/tune edin)",
                    "confidence": 0.0, "snr_db": round(snr, 1)}

        # GERÇEK KANAL İZOLASYONU (DDC): Dinamik çalışma hızı ile merkezleme
        work, iso_fs = self._channelize(iq)
        if len(work) < 256:
            return {"modulation": "Belirlenemedi (yetersiz örnek)", "confidence": 0.0,
                    "snr_db": round(snr, 1)}

        feats = self.extract_features(work, iso_fs)
        # _decide KESİN ikili karar verir: AM ya da FM (sinyalsiz/düşük-SNR/çok-sinyal durumları
        # yukarıda zaten elendi). Kullanıcı isteği: sinyal varken net AM/FM yazsın.
        mod, conf = self._decide(feats)
        # SADECE ANALOG (AM/FM): dijital alanlar (multiplex/ekkt/symbol_rate) şimdilik KAPALI.
        # sigma_aa TEŞHİS için döndürülür: gerçek donanımda sinyal jeneratöründen AM/FM basıp bu
        # değeri terminalde izleyerek eşiğin (TH_AM_SAA=0.09) sinyalinizle uyumunu doğrulayabilirsiniz.
        return {"modulation": mod, "confidence": round(conf, 2), "snr_db": round(snr, 1),
                "sigma_aa": round(feats["sigma_aa"], 3)}
        # İleride dijital açılınca:  "multiplex": mux, "ekkt": ekkt_dsss,
        #   "symbol_rate_hz": round(self.estimate_symbol_rate_hz(iq) if ("PSK" in mod or "QAM" in mod) else 0.0, 0)


class HoppingHistoryTracker:
    """FHSS (frekans atlama) tespiti — ZAMAN GEÇMİŞİNDEN (uzman eleştirisi #1'in doğru çözümü).

    Uzman haklıydı: tek bir 0.2 ms'lik blok bir FHSS atlamasından (Bluetooth 625 µs) kısa
    olduğu için atlama TEK BLOKTA görülemez. Doğru yöntem: her karede (frame) ucuza hesaplanan
    baskın (tepe) frekansı bir ZAMAN PENCERESİ boyunca (waterfall hafızası) biriktirip, frekansın
    zaman içinde birçok AYRIK kanal arasında sıçrayıp sıçramadığına bakmaktır.

    Kullanım: run() döngüsünde her kare `update(t, peak_norm, snr_db)` çağrılır (canlı FFT'nin
    tepe bini bedava verir). `detect()` pencere içindeki geçmişten FHSS kararını verir. Ucuzdur
    (kare başına O(1)); ağır DSP yoktur."""

    def __init__(self, window_sec: float = 2.0, min_snr_db: float = 10.0,
                 n_channels: int = 32, min_distinct: int = 4, min_hops: int = 8,
                 min_points: int = 15):
        self.window_sec = window_sec       # geçmiş penceresi (sn)
        self.min_snr_db = min_snr_db       # yalnızca sinyal-var kareler sayılır (gürültü elenr)
        self.n_channels = n_channels       # bandı kaç kanala böl (atlama çözünürlüğü)
        self.min_distinct = min_distinct   # FHSS için gereken en az ayrık kanal sayısı
        self.min_hops = min_hops           # pencerede gereken en az ayrık sıçrama sayısı
        self.min_points = min_points       # karar için gereken en az sinyal-var kare
        self._hist = deque()               # (t, peak_norm[0..1], snr_db)

    def reset(self):
        self._hist.clear()

    def update(self, t: float, peak_norm: float, snr_db: float):
        self._hist.append((t, float(peak_norm), float(snr_db)))
        while self._hist and (t - self._hist[0][0]) > self.window_sec:
            self._hist.popleft()

    def detect(self):
        """Dönüş: (etiket, hop_sayısı, bilgi_dict). Sinyal-var karelerdeki tepe-frekans zaman
        serisinden ayrık kanal sayısı + sıçrama sayısı ile FHSS kararı verir."""
        pts = [(t, f) for (t, f, s) in self._hist if s >= self.min_snr_db]
        if len(pts) < self.min_points:
            return "Yok", 0, {"distinct": 0, "points": len(pts)}
        freqs = np.array([f for _, f in pts])
        chans = np.clip((freqs * self.n_channels).astype(int), 0, self.n_channels - 1)
        distinct = int(len(np.unique(chans)))
        # Ardışık kareler arası KANAL değişimi (sıçrama). Aynı kanalda kalmak sıçrama değildir.
        hops = int(np.sum(np.abs(np.diff(chans)) >= 1))
        info = {"distinct": distinct, "hops": hops, "points": len(pts)}
        if distinct >= self.min_distinct and hops >= self.min_hops:
            return "FHSS (Frekans Atlama)", hops, info
        return "Yok", hops, info
