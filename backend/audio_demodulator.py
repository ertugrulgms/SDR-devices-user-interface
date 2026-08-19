import numpy as np
import scipy.signal as signal


class StreamingDemodulator:
    """Gerçek hoparlör dinlemesi için durumlu (stateful), kesintisiz (gapless) demodülatör.

    AudioDemodulator sadece grafik için 100 nokta üretir; bu sınıf ise ring buffer'dan gelen
    ardışık I/Q bloklarını tam ses hızında (≈48 kHz) demodüle eder ve blok sınırlarında tık/boşluk
    oluşmaz — tüm filtre durumları (lfilter zi), FM ayrımlayıcı hafızası, SSB karıştırıcı fazı ve
    decimation faz-ofseti bloklar arasında korunur.

    Desteklenen modlar: FM (NBFM), AM, USB, LSB. Amatör analog telsiz için tasarlanmıştır.
    Çıkış: float32, [-1, 1] aralığında, self.out_rate örnekleme hızında.
    """

    MODES = ("FM", "AM", "USB", "LSB")
    # FM: 16 kHz -> ±8 kHz kanal (NBFM ±5 kHz sapma + ses; hem 12.5 hem 25 kHz kanalı kapsar).
    _DEFAULT_BW = {"FM": 16000.0, "AM": 10000.0, "USB": 2800.0, "LSB": 2800.0}

    @staticmethod
    def _factorize_decim(q: int) -> list:
        """Decimation faktörünü küçük kademelere böler (çok-kademeli anti-alias için). Tek büyük FIR
        yüksek örnekleme hızında dar kanalı gerçekleyemez -> aliasing; kademeli seyreltme bunu çözer."""
        stages = []
        r = int(q)
        for f in (5, 4, 3, 2, 7):
            while r % f == 0 and r > 1:
                stages.append(f)
                r //= f
        if r > 1:
            stages.append(r)
        return stages or [1]

    def __init__(self, sample_rate: float, audio_rate: int = 48000, mode: str = "FM",
                 channel_bw: float = None, digital: bool = False):
        # digital=True: sayısal ses (C4FM/4FSK) için HAM FM ayrımlayıcı — de-emphasis YOK, AGC YOK,
        # geniş ses filtresi. DSD-FME / 4FSK tespiti bu ham diskriminatörü bekler.
        self.digital = bool(digital)
        self.sample_rate = float(sample_rate)
        self.audio_rate = int(audio_rate)
        # Tam sayı decimation faktörü: gerçek çıkış hızı = sample_rate / decim (gapless için
        # yeniden örnekleme yok). 2.4e6/48000 = 50 -> tam 48 kHz.
        self.decim = max(1, int(round(self.sample_rate / self.audio_rate)))
        self.out_rate = self.sample_rate / self.decim
        self.tuning_offset_hz = 0.0      # otomatik-merkezleme (worker ölçülen tepe-offseti ile günceller)
        self.set_mode(mode, channel_bw)  # filtreleri kurar + durumu sıfırlar

    # ------------------------------------------------------------------ kurulum
    def set_mode(self, mode: str, channel_bw: float = None):
        mode = (mode or "FM").upper()
        if mode not in self.MODES:
            mode = "FM"
        self.mode = mode
        self.channel_bw = float(channel_bw) if channel_bw else self._DEFAULT_BW[mode]

        # ÇOK-KADEMELİ ANTI-ALIAS SEYRELTME: decim'i küçük kademelere böl, her kademe kendi FIR'i ile
        # anti-alias'lar. Tek büyük FIR (65-tap) yüksek örnekleme hızında (ör. 20 MHz) dar kanalı
        # gerçekleyemez -> tüm banttan gürültü aliasing ile sese sızardı (asıl 'ses gürültü geliyor' hatası).
        self._dec_stage_b = []
        self._dec_stage_f = []
        for f in self._factorize_decim(self.decim):
            if f <= 1:
                continue
            # Bu kademe için anti-alias FIR: kesim = 0.8/f (o kademenin Nyquist'ine göre)
            ntaps = min(8 * f + 1, 161)
            self._dec_stage_b.append(signal.firwin(ntaps, 0.8 / f).astype(np.float64))
            self._dec_stage_f.append(f)

        # Kanal (channel) filtresi ARTIK out_rate'te uygulanır (kesim = kanal BW/2). out_rate'te
        # kesim/Nyquist makul olduğundan 65-tap yeterli (yüksek fs'de gerçeklenememe sorunu biter).
        aud_nyq0 = self.out_rate / 2.0
        chan_cut = min(self.channel_bw / 2.0, aud_nyq0 * 0.95)
        chan_cut = max(chan_cut, 100.0)
        self._chan_b = signal.firwin(65, chan_cut / aud_nyq0).astype(np.float64)

        # Ses-bandı alçak geçiren (out_rate'te; FM/AM temizliği için).
        # Sayısal modda geniş bırak (C4FM sembol içeriği ~kanal BW/2'ye kadar).
        aud_nyq = self.out_rate / 2.0
        aud_cut = min(self.channel_bw / 2.0 if self.digital else 3400.0, aud_nyq * 0.95)
        self._aud_b = signal.firwin(65, aud_cut / aud_nyq).astype(np.float64)

        # SSB yan-bant seçim filtresi (out_rate'te, merkezlenmiş aşamada uygulanır).
        ssb_cut = min(self.channel_bw / 2.0, aud_nyq * 0.95)
        self._ssb_b = signal.firwin(65, ssb_cut / aud_nyq).astype(np.float64)

        # AM DC-engelleme (yüksek geçiren, tek kutuplu).
        self._dc_b = np.array([1.0, -1.0])
        self._dc_a = np.array([1.0, -0.995])

        # FM de-emphasis (75 µs tek kutuplu alçak geçiren, out_rate'te).
        tau = 75e-6
        alpha = 1.0 / (1.0 + 1.0 / (tau * self.out_rate))
        self._deemph_b = np.array([alpha])
        self._deemph_a = np.array([1.0, -(1.0 - alpha)])

        self._reset_state()

    def _reset_state(self):
        self._prev_iq = None                     # FM ayrımlayıcı: önceki bloğun son örneği
        # Çok-kademeli seyreltme durumu: her kademe için filtre durumu (zi) + decimation faz-ofseti
        self._dec_stage_zi = [signal.lfilter_zi(b, 1.0).astype(np.complex128) * 0.0
                              for b in self._dec_stage_b]
        self._dec_stage_ph = [0 for _ in self._dec_stage_b]
        self._tune_phase = 0.0                   # otomatik-merkezleme karıştırıcı faz akümülatörü
        self._chan_zi = signal.lfilter_zi(self._chan_b, 1.0).astype(np.complex128) * 0.0
        self._aud_zi = signal.lfilter_zi(self._aud_b, 1.0) * 0.0
        self._ssb_zi = signal.lfilter_zi(self._ssb_b, 1.0).astype(np.complex128) * 0.0
        self._dc_zi = signal.lfilter_zi(self._dc_b, self._dc_a) * 0.0
        self._deemph_zi = signal.lfilter_zi(self._deemph_b, self._deemph_a) * 0.0
        self._ssb_phase = 0.0                    # SSB karıştırıcı faz akümülatörü
        self._agc_gain = 1.0                     # yumuşak AGC kazancı

    def reset(self):
        self._reset_state()

    def set_tuning_offset(self, offset_hz: float):
        """OTOMATİK MERKEZLEME: hedef sinyalin merkez-frekanstan sapması (Hz). Demod, sinyali DC'ye
        çeker -> operatör tam tune etmese bile temiz ses (aksi halde sinyal kanal filtresince süzülür).
        Worker bunu ölçülen tepe-offseti ile günceller."""
        self.tuning_offset_hz = float(offset_hz or 0.0)

    # ------------------------------------------------------------- yardımcılar
    def _decimate_complex(self, x: np.ndarray) -> np.ndarray:
        """Karmaşık I/Q'yu sample_rate -> out_rate'e ÇOK-KADEMELİ, faz-sürekli (gapless) seyreltir.
        Önce (varsa) hedef sinyali DC'ye kaydırır (otomatik-merkezleme), sonra kademe kademe anti-alias
        seyreltme, son olarak kanal filtresi. Yüksek örnekleme hızında aliasing'i önler."""
        # 0) Otomatik-merkezleme: hedefi DC'ye kaydır (faz-sürekli karıştırıcı)
        off = getattr(self, "tuning_offset_hz", 0.0)
        if abs(off) > 1.0:
            n0 = len(x)
            k = np.arange(n0)
            x = (x * np.exp(-1j * (2.0 * np.pi * off * k / self.sample_rate + self._tune_phase))).astype(np.complex64)
            self._tune_phase = (self._tune_phase + 2.0 * np.pi * off * n0 / self.sample_rate) % (2.0 * np.pi)

        # 1) Kademeli anti-alias seyreltme (her kademe: FIR + faz-sürekli downsample)
        y = x
        for i, f in enumerate(self._dec_stage_f):
            y, self._dec_stage_zi[i] = signal.lfilter(self._dec_stage_b[i], 1.0, y, zi=self._dec_stage_zi[i])
            n = len(y)
            idx = np.arange(self._dec_stage_ph[i], n, f)
            if len(idx):
                self._dec_stage_ph[i] = int(idx[-1] + f - n)
                y = y[idx]
            else:
                self._dec_stage_ph[i] -= n
                y = y[:0]
            if len(y) == 0:
                return y

        # 2) Kanal filtresi (out_rate'te; NBFM kanalını daralt)
        y, self._chan_zi = signal.lfilter(self._chan_b, 1.0, y, zi=self._chan_zi)
        return y

    def _agc(self, audio: np.ndarray) -> np.ndarray:
        """Yumuşak otomatik kazanç: tıklama olmadan ~0.7 tepe hedefler."""
        if len(audio) == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-9:
            target = 0.7 / peak
            # kazancı yumuşat (bloklar arası ani sıçrama = tık)
            self._agc_gain = 0.9 * self._agc_gain + 0.1 * min(target, 50.0)
        audio = audio * self._agc_gain
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- işleme
    def process(self, iq: np.ndarray) -> np.ndarray:
        """Bir I/Q bloğunu demodüle eder. Çıkış: float32 ses, self.out_rate hızında."""
        if iq is None or len(iq) < 2:
            return np.zeros(0, dtype=np.float32)
        iq = np.asarray(iq, dtype=np.complex64)

        # 1) Kanal seçimi + gapless seyreltme -> out_rate karmaşık taban bant
        y = self._decimate_complex(iq)
        if len(y) < 2:
            return np.zeros(0, dtype=np.float32)

        # 2) Moda özgü demodülasyon
        if self.mode == "FM":
            audio = self._demod_fm(y)
        elif self.mode == "AM":
            audio = self._demod_am(y)
        else:  # USB / LSB
            audio = self._demod_ssb(y)

        # 3) Ses-bandı temizleme
        audio, self._aud_zi = signal.lfilter(self._aud_b, 1.0, audio, zi=self._aud_zi)
        # Sayısal modda HAM ayrımlayıcı döndür (AGC/normalize YOK -> 4FSK seviyeleri korunur).
        if self.digital:
            return audio.astype(np.float32)
        return self._agc(audio)

    def _demod_fm(self, y: np.ndarray) -> np.ndarray:
        # Polar ayrımlayıcı; önceki bloğun son örneğini hafızada tut (gapless).
        if self._prev_iq is None:
            prev = y[:1]
        else:
            prev = np.array([self._prev_iq], dtype=y.dtype)
        ext = np.concatenate([prev, y])
        disc = np.angle(ext[1:] * np.conj(ext[:-1]))
        self._prev_iq = y[-1]
        # De-emphasis (sayısal modda UYGULANMAZ — C4FM seviyeleri bozulmasın)
        if self.digital:
            return disc
        out, self._deemph_zi = signal.lfilter(self._deemph_b, self._deemph_a, disc,
                                               zi=self._deemph_zi)
        return out

    def _demod_am(self, y: np.ndarray) -> np.ndarray:
        env = np.abs(y)
        out, self._dc_zi = signal.lfilter(self._dc_b, self._dc_a, env, zi=self._dc_zi)
        return out

    def _demod_ssb(self, y: np.ndarray) -> np.ndarray:
        # Frekans-kaydırma (Weaver) yöntemi: istenen yan bandı DC'ye taşı, alçak geçirerek
        # istenmeyen yan bandı at, geri kaydır ve gerçek kısmı al.
        # USB (pozitif frekanslar) -> aşağı kaydır (+jw0 karıştırıcıya bölünür); LSB -> yukarı.
        w0 = 2.0 * np.pi * (self.channel_bw / 2.0) / self.out_rate
        sign = 1.0 if self.mode == "USB" else -1.0  # USB'yi aşağı taşımak için exp(-jw0n)
        n = np.arange(len(y))
        # exp(-j*sign*w0*n): USB için band [0,bw] -> [-bw/2,+bw/2]; LSB için [-bw,0] -> merkez
        down = np.exp(-1j * sign * (w0 * n + self._ssb_phase))
        up = np.conj(down)  # geri kaydırma
        self._ssb_phase = (self._ssb_phase + w0 * len(y)) % (2.0 * np.pi)
        centered = y * down
        filt, self._ssb_zi = signal.lfilter(self._ssb_b, 1.0, centered, zi=self._ssb_zi)
        return np.real(filt * up)


class AudioDemodulator:
    def __init__(self, sample_rate: float, audio_rate: int = 48000):
        """
        SDR'dan gelen I/Q verisini FM (veya AM) demodüle edip sese (sayısal diziye) çevirir.
        Grafik çizimi ve sinyal analizi için kullanılır.
        :param sample_rate: SDR'ın donanımsal örnekleme hızı (örneğin 2.4e6)
        :param audio_rate: Çıkış ses hızı (örneğin 48000 Hz)
        """
        self.sample_rate = sample_rate
        self.audio_rate = audio_rate
        self.decimation_factor = int(self.sample_rate / self.audio_rate)
        if self.decimation_factor < 1:
            self.decimation_factor = 1

    @staticmethod
    def _factorize_stages(q: int) -> list:
        """Büyük bir decimation faktörünü <=5'lik kademelere böler. scipy.signal.decimate
        tek çağrıda q>13 için kararsızdır; çok kademeli decimation kararlı sonuç verir."""
        stages = []
        r = int(q)
        for f in (5, 4, 3, 2):
            while r % f == 0 and r > 1:
                stages.append(f)
                r //= f
        if r > 1:
            stages.append(r)
        return stages

    def _decimate_staged(self, x: np.ndarray) -> np.ndarray:
        """Faktörü kademelere bölerek kararlı (FIR, sıfır faz) decimation uygular."""
        y = x
        for s in self._factorize_stages(self.decimation_factor):
            # Her kademe, FIR filtre uzunluğundan yeterince uzun giriş gerektirir
            if s > 1 and len(y) > 20 * s + 1:
                y = signal.decimate(y, s, ftype='fir', zero_phase=True)
        return y

    def fm_demodulate(self, iq_samples: np.ndarray, output_points: int = 100) -> np.ndarray:
        """
        Gelen I/Q verisinden Geniş Bant FM (WBFM) veya Dar Bant FM (NBFM) demodülasyonu yapar.
        Dönüş değeri, arayüzdeki grafiğin boyutuna uyacak şekilde ayarlanmış ses dizisidir.
        :param output_points: Grafikte gösterilecek nokta sayısı (arayüz için genelde 100)
        """
        if len(iq_samples) < 2:
            return np.zeros(output_points)

        # 1. Faz Tespiti (Polar Discriminator): Mevcut örnek ile bir öncekinin eşleniğini çarparak açıyı bul.
        # Bu, anlık frekans sapmasını (FM demodülasyonu) verir.
        instantaneous_phase = np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))
        
        # 2. Decimation (Seyreltme): RF örnekleme hızından Ses örnekleme hızına (48kHz) in.
        # Büyük faktörler (ör. 2.4MHz/48kHz=50) kademeli olarak seyreltilir (kararlılık).
        audio_signal = self._decimate_staged(instantaneous_phase)
        
        # 3. Normalizasyon (Grafiği -1.0 ile 1.0 arasına sıkıştır)
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            audio_signal = audio_signal / max_val
            
        # 4. Arayüz Grafiği için veriyi yeniden örnekle (Downsample) veya kırp
        # Grafikte sabit (örn 100) nokta gösterildiği için, o boyuta getiriyoruz.
        if len(audio_signal) > output_points:
            # En son gelen 100 veriyi al veya signal.resample kullan
            audio_display = signal.resample(audio_signal, output_points)
        else:
            # Veri azsa sıfırlarla tamamla
            audio_display = np.pad(audio_signal, (0, max(0, output_points - len(audio_signal))), 'constant')
            
        return audio_display
