import numpy as np

# --- DAC GÜVENLİĞİ (ORTAK TX SABİTLERİ) ---
# UHD/Soapy complex-float TX akışı baseband'i [-1, 1] aralığında bekler. Bu tavanın üstündeki
# her örnek DONANIMDA sert DAC clipping'e girer -> kontrolsüz spektral genişleme, komşu kanala
# taşma, ölçüm bozulması. Tüm TX modları üretim sonunda bu tepe büyüklüğe faz-korumalı olarak
# geri çekilir (enforce_dac_safe). Böylece JSR/genlik ne olursa olsun DAC ASLA clip olmaz.
TX_DIGITAL_PEAK = 0.9

# JSR (Jammer-to-Signal Ratio) hedefi artık DİJİTAL SÜRÜŞ seviyesini belirler ve bu değerde
# (dB) dijital genlik tam-skala tavanına (TX_DIGITAL_PEAK) ulaşır. Bunun üstündeki JSR dijital
# olarak DOYAR (clip olmaz, tavanda kalır); gerçek radyasyon gücü RF TX kazancından (dB) gelir.
JSR_FULLSCALE_DB = 20.0


class JammingGenerator:
    def __init__(self, sample_rate_hz=10e6):
        self.sample_rate = sample_rate_hz

        # --- PRE-ALLOCATION (ÖN BELLEKLEME) MİMARİSİ ---
        # RAM'de bir kereye mahsus 1 milyon örnekli devasa bir gürültü havuzu oluşturulur.
        # Bu sayede döngü içinde sürekli np.random çağrısı yapıp CPU boğulmaz.
        self.PREALLOC_SIZE = 1000000
        _i = np.random.normal(0, 1, self.PREALLOC_SIZE)
        _q = np.random.normal(0, 1, self.PREALLOC_SIZE)
        self._noise_pool = (_i + 1j * _q).astype(np.complex64)
        self._pool_idx = 0

    # ------------------------------------------------------------------ #
    #  DAC GÜVENLİK LİMİTLEYİCİSİ                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def enforce_dac_safe(buf: np.ndarray, peak: float = TX_DIGITAL_PEAK) -> np.ndarray:
        """Tampon tepe büyüklüğü `peak`i aşıyorsa TÜM tamponu doğrusal olarak geri çeker.
        Doğrusal ölçekleme (per-örnek kırpma DEĞİL) seçildi: ton/çoklu-ton/chirp/FM gibi
        deterministik dalgaların ŞEKLİ birebir korunur (clipping kaynaklı intermod/harmonik
        oluşmaz); gürültü için PAPR kadar güç geri çekilir (jammer'lar zaten bu şekilde çalışır).
        Sonuç: max|buf| <= peak GARANTİ -> donanımda DAC clipping İMKANSIZ."""
        buf = np.ascontiguousarray(buf, dtype=np.complex64)
        if buf.size == 0:
            return buf
        m = float(np.max(np.abs(buf)))
        if m > peak:
            buf = (buf * (peak / m)).astype(np.complex64)
        return buf

    def jsr_to_amplitude(self, jsr_db: float, signal_amplitude: float = 1.0) -> float:
        """JSR (dB) -> DAC-GÜVENLİ dijital sürüş genliği. JSR yükseldikçe genlik tam-skala
        tavanına (TX_DIGITAL_PEAK) yaklaşır ve JSR_FULLSCALE_DB'de tavana OTURUR; üstünde
        doyar (tavanı ASLA aşmaz). Böylece bu skaler tek başına bile DAC clipping üretemez;
        gerçek RF çıkış gücü ise ayrıca RF TX kazancından (dB, arayüzden) gelir."""
        rel = 10 ** ((jsr_db - JSR_FULLSCALE_DB) / 20.0)
        return signal_amplitude * TX_DIGITAL_PEAK * min(1.0, rel)

    # ------------------------------------------------------------------ #
    #  BARAJ (GENİŞ BANT GÜRÜLTÜ)                                          #
    # ------------------------------------------------------------------ #
    def generate_barrage_noise(self, num_samples: int, amplitude: float = 1.0,
                               bw_frac: float = 1.0) -> np.ndarray:
        """Önceden üretilmiş havuzdan hızlıca (O(1)) gürültü dilimi okur.
        Havuz sonu aşılırsa dilim SARILARAK birleştirilir (kuyruk ATILMAZ; önceki sürümde
        sarmada havuzun kalan kuyruğu atlanıp süreksizlik oluşuyordu). num_samples havuzdan
        büyükse döngüsel tekrar edilir. bw_frac < 1 ise gürültü, bandın merkezî bw_frac'ine
        FFT ile sınırlandırılır (hedefe göre dar/geniş baraj)."""
        P = self.PREALLOC_SIZE
        idx = self._pool_idx
        pool = self._noise_pool

        if num_samples <= P:
            end = idx + num_samples
            if end <= P:
                noise = pool[idx:end]
                self._pool_idx = end % P
            else:
                # Havuz sonunu aşıyor -> baştan sararak birleştir (kuyruk korunur)
                noise = np.concatenate((pool[idx:], pool[:end - P]))
                self._pool_idx = end - P
        else:
            # İstenen miktar havuzdan büyük -> döngüsel tekrar
            noise = np.take(pool, (np.arange(num_samples) + idx) % P)
            self._pool_idx = (idx + num_samples) % P

        out = (noise * amplitude).astype(np.complex64)
        if bw_frac < 0.999:
            out = self._bandlimit(out, bw_frac)
        return out

    def _bandlimit(self, x: np.ndarray, bw_frac: float) -> np.ndarray:
        """Sinyali, örnekleme bandının merkezî bw_frac oranındaki kısmına FFT maskesiyle
        sınırlar (dar-bant baraj) VE gücü korunan banda TOPLAR (güç yoğunluğu/PSD artar).

        KRİTİK (önceki hata): yalnızca bant-dışı binleri sıfırlamak, bant-dışı ENERJİYİ ATAR ->
        toplam güç düşer, korunan banttaki YÜKSEKLİK (PSD) ARTMAZ. Sonuç: bant daralsa bile jam
        gücü hedef kanalda yükselmiyordu ('yükseklik değişmiyor, sadece frekans daralıyor').
        DÜZELTME: filtrelemeden sonra RMS'i (toplam güç) eski değerine geri ölçekle -> orijinal güç
        DAR BANDA toplanır, PSD ~1/bw_frac kat artar -> odaklanmış jam hedef kanalda ÇOK daha güçlü.
        (Nihai tepe downstream enforce_dac_safe ile 0.9'a sınırlanır -> DAC güvenli kalır.)"""
        bw_frac = max(0.005, min(1.0, bw_frac))
        n = len(x)
        rms_in = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        X = np.fft.fftshift(np.fft.fft(x))
        keep = int(n * bw_frac)
        lo = (n - keep) // 2
        mask = np.zeros(n, dtype=bool)
        mask[lo:lo + keep] = True
        X[~mask] = 0.0
        y = np.fft.ifft(np.fft.ifftshift(X))
        # GÜÇ KORUMA: atılan bant-dışı enerjiyi telafi et -> gücü korunan (dar) banda topla
        rms_out = float(np.sqrt(np.mean(np.abs(y) ** 2)))
        if rms_out > 1e-12 and rms_in > 1e-12:
            y = y * (rms_in / rms_out)
        return y.astype(np.complex64)

    # ------------------------------------------------------------------ #
    #  TEKLİ / SPOT (DAR-BANT GÜRÜLTÜ — TEK HEDEF KANALI)                  #
    # ------------------------------------------------------------------ #
    def generate_spot_noise(self, num_samples: int, amplitude: float = 1.0,
                            bw_frac: float = 0.02, offset_hz: float = 0.0) -> np.ndarray:
        """TEKLİ (SPOT) karıştırma: gücü TEK hedef kanalına yoğunlaştırılmış DAR-BANT GÜRÜLTÜ.

        Neden gürültü, neden CW ton DEĞİL: şartname 5.2.1 karıştırmayı 'alıcı girişinde gürültü
        seviyesinin yükseltilmesi' (Shannon kanal kapasitesine taarruz) olarak tanımlar. Saf CW ton
        gürültü tabanını YÜKSELTMEZ — tek spektral çizgidir; sayısal/yayılı-spektrum alıcı onu
        çentikle atar, FM sese karşı da yalnızca marjinal (yakalama) etki yapar. SPOT gürültü ise
        hedef kanal İÇİNDE gürültü tabanını azami güç yoğunluğuyla yükseltir -> tek hedefe en etkili
        karıştırma. Gürültü, hedef kanal bandına (bw_frac) sınırlanıp hedef offsetine (offset_hz)
        kaydırılır."""
        buf = self.generate_barrage_noise(num_samples, amplitude=amplitude, bw_frac=bw_frac)
        if abs(offset_hz) > 1.0:
            t = np.arange(num_samples) / self.sample_rate
            buf = (buf * np.exp(1j * 2 * np.pi * offset_hz * t)).astype(np.complex64)
        return buf

    # ------------------------------------------------------------------ #
    #  TON / ÇOKLU TON / SÜPÜRME                                           #
    # ------------------------------------------------------------------ #
    def generate_single_tone(self, freq_offset_hz: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        # Tone statik olduğu için anlık hesaplanması CPU'yu çok yormaz
        t = np.arange(num_samples) / self.sample_rate
        tone = amplitude * np.exp(1j * 2 * np.pi * freq_offset_hz * t)
        return tone.astype(np.complex64)

    def generate_multi_tone(self, num_samples: int, amplitude: float = 1.0,
                            n_tones: int = 5, spacing_hz: float = None) -> np.ndarray:
        """Çoklu-ton (multi-tone) karıştırma: merkez etrafında eşit aralıklarla n taşıyıcı.
        Tek tona göre hedef bandın daha geniş bir bölümünü aynı anda kaplar. Tonlar merkeze
        göre simetrik yerleşir; toplam örneklenir ve tepe genliği kontrolü için normalize edilir."""
        if n_tones < 1:
            n_tones = 1
        if spacing_hz is None:
            # Bandın büyük kısmını kapla (Nyquist'in altında kalacak şekilde)
            spacing_hz = (self.sample_rate * 0.8) / max(n_tones, 1)
        t = np.arange(num_samples) / self.sample_rate
        offsets = (np.arange(n_tones) - (n_tones - 1) / 2.0) * spacing_hz
        sig = np.zeros(num_samples, dtype=np.complex128)
        for f in offsets:
            sig += np.exp(1j * 2 * np.pi * f * t)
        sig /= n_tones                                   # tepe genliğini sınırla
        return (amplitude * sig).astype(np.complex64)

    def generate_chirp_sweep(self, num_samples: int, amplitude: float = 1.0,
                             sweep_fraction: float = 0.9, n_sweeps: int = 8) -> np.ndarray:
        """SÜPÜRMELİ (Chirp / LFM) jammer: frekans bant içinde -B/2'den +B/2'ye HIZLA tarar ve
        bunu tampon boyunca n_sweeps kez tekrarlar. Frekans-çevik hedeflere (WiFi otomatik kanal,
        Bluetooth AFH) karşı barrage'dan etkilidir (gücü dar banda toplar, her kanaldan geçer).

        FAZ SÜREKLİLİĞİ: anlık frekans testere-dişi (periyodik) olsa da FAZ, anlık frekansın
        KÜMÜLATİF integralidir. Frekans süpürme başında -B/2'ye 'sıçrasa' bile faz sürekli kalır
        (integralin kendisi süreklidir) -> blok sınırlarında tıklama/spektral saçılma OLMAZ.
        (Önceki sürüm her bloğu bağımsız üretip fazı sıfırlıyor, sınırlarda süreksizlik yaratıyordu.)"""
        n_sweeps = max(1, int(n_sweeps))
        B = self.sample_rate * sweep_fraction              # süpürülen bant genişliği
        T = num_samples / self.sample_rate / n_sweeps      # tek süpürme süresi
        if T <= 0:
            T = num_samples / self.sample_rate
        t = np.arange(num_samples) / self.sample_rate
        tau = np.mod(t, T)                                 # her süpürme içinde 0..T
        inst_freq = -B / 2.0 + B * (tau / T)               # doğrusal süpürme (-B/2 -> +B/2)
        # SÜREKLİ faz = anlık frekansın kümülatif integrali
        phase = 2 * np.pi * np.cumsum(inst_freq) / self.sample_rate
        return (amplitude * np.exp(1j * phase)).astype(np.complex64)

    # ------------------------------------------------------------------ #
    #  ANALOG ALDATMA                                                      #
    # ------------------------------------------------------------------ #
    def generate_analog_spoofing(self, wave_type: str, offset_ms: float, num_samples: int, amplitude: float = 1.0) -> np.ndarray:
        """Analog telsiz ALDATMA (5.2.3): analog amatör telsize sahte/gerçek-dışı yayın üretir.
        RF karıştırma 'hiç duymaz', aldatma 'yanlış duyar' hedefler. Dalga şekline göre üretim.
        Zarf tipi doğru olsun diye: FM türleri SABİT ZARF (frekans modülasyonu), AM türü ise
        DEĞİŞKEN ZARF (genlik modülasyonu) üretir."""
        t = np.arange(num_samples) / self.sample_rate
        t_offset = t + (offset_ms / 1000.0)

        if wave_type == "Sinüs Dalga (Tone)":
            # Tek tonlu (CW) aldatma sinyali
            return (amplitude * np.exp(1j * 2 * np.pi * 1e4 * t_offset)).astype(np.complex64)

        elif wave_type == "Ses/Audio Sahte Ses":
            # GERÇEK SES-BENZERİ FM: insan sesi bandında (300-3400 Hz) çok bileşenli, zamanla
            # değişen bir "mesaj" üretilip FM ile taşıyıcıya bindirilir. Analog FM telsizde
            # gerçek-dışı ama SES gibi algılanan bir yayın oluşturur (yanlış duyurma). Sabit zarf.
            rng = np.random.default_rng(1234)
            msg = (np.sin(2 * np.pi * 350 * t_offset)
                   + 0.6 * np.sin(2 * np.pi * 900 * t_offset)
                   + 0.4 * np.sin(2 * np.pi * 1800 * t_offset)
                   + 0.25 * np.sin(2 * np.pi * (600 + 300 * np.sin(2 * np.pi * 4 * t)) * t_offset)  # formant kayması
                   + 0.15 * rng.standard_normal(num_samples))                                        # soluk/gürültü
            msg = msg / (np.max(np.abs(msg)) + 1e-9)
            freq_dev = 3.0e3                                    # dar-bant FM sapması (ses telsizi)
            phase = 2 * np.pi * freq_dev * np.cumsum(msg) / self.sample_rate
            return (amplitude * np.exp(1j * phase)).astype(np.complex64)

        elif wave_type == "Gürültü Modüleli FM":
            # GERÇEK FM: gürültü mesajı FREKANSA integre edilir -> SABİT ZARF, geniş-bant gürültü
            # yayını. (Önceki "AM/FM" etiketi aslında sadece AM üretiyordu; artık ikisi ayrı ve doğru.)
            msg = self.generate_barrage_noise(num_samples, 1.0).real
            msg = msg / (np.max(np.abs(msg)) + 1e-9)
            freq_dev = 5.0e3
            phase = 2 * np.pi * freq_dev * np.cumsum(msg) / self.sample_rate
            return (amplitude * np.exp(1j * phase)).astype(np.complex64)

        elif wave_type == "Gürültü Modüleli AM":
            # GERÇEK AM: bir taşıyıcının GENLİĞİ gürültüyle modüle edilir -> DEĞİŞKEN ZARF.
            env = 0.5 + 0.5 * np.abs(self.generate_barrage_noise(num_samples, 1.0))  # pozitif zarf
            env = env / (np.max(env) + 1e-9)
            carrier = np.exp(1j * 2 * np.pi * 5000 * t_offset)
            return (amplitude * env * carrier).astype(np.complex64)

        else:
            return (amplitude * np.exp(1j * 2 * np.pi * 1e3 * t_offset)).astype(np.complex64)
