"""Analog Telsiz Aldatma (spec 5.2.3): GERÇEK bir ses mesajını hedef analog telsizin modülasyonuna
(NBFM veya AM) uygun şekilde baseband'e taşır. Böylece hedef alıcı, gerçek-dışı ama ANLAŞILIR bir
yayını 'geçerli' kabul edip 'yanlış duyar' (RF karıştırma 'hiç duymaz'; aldatma 'yanlış duyar').

Sentetik ton çorbası yerine operatörün seçtiği gerçek bir WAV mesajı (ör. sahte anons) yayınlanır —
modülasyon/sapma hedefin dalga-şekli özelliklerine uydurulur. Modülasyon matematiği saf ve donanımsız
birim-test edilebilir: üretilen baseband, bir FM/AM demodülatöründen geçirilince özgün sesi verir
(yani hedef telsiz onu gerçek ses olarak çalar).
"""
import wave
import numpy as np
import scipy.signal as signal

# TX tamponu tek blok olarak loop'landığından, aldatma mesajı bellek için sınırlanır. Fazlası
# kırpılır (mesaj sürekli tekrar yayınlanır). ~48 MB complex64 üst sınırı.
MAX_DECEPTION_SAMPLES = 6_000_000
# NBFM ses telsizi tipik tepe sapması (amatör dar-bant, 12.5 kHz kanal ~ ±2.5–3 kHz)
DEFAULT_NBFM_DEVIATION_HZ = 3000.0

# CTCSS (Continuous Tone-Coded Squelch System) standart alt-ses tonları (Hz). Analog amatör/PMR
# telsizler bu tonu taşımayan yayınlarda hoparlörü AÇMAZ (ton-squelch). Aldatmanın duyulabilmesi
# için hedefin kullandığı tonun sinyale bindirilmesi ŞARTTIR (spec 5.2.3 'protokol özellikleri').
CTCSS_TONES = [
    67.0, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8,
    97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8,
    136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9, 173.8, 179.9, 186.2,
    192.8, 203.5, 210.7, 218.1, 225.7, 233.6, 241.8, 250.3,
]


def load_wav_mono(path: str):
    """WAV dosyasını tek kanal float [-1, 1] + örnekleme hızı olarak yükler. Stereo -> ortalanır.
    Dönüş: (audio: np.float32, rate: int). Desteklenmeyen/bozuksa istisna fırlatır (sahte üretmez)."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"desteklenmeyen WAV örnek genişliği: {sw} bayt")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data.astype(np.float32), int(rate)


def nearest_ctcss(freq_hz: float, tol_hz: float = 2.5):
    """Ölçülen frekansa en yakın standart CTCSS tonunu döndürür (tol içinde) veya None."""
    if freq_hz is None or freq_hz <= 0:
        return None
    best = min(CTCSS_TONES, key=lambda c: abs(c - freq_hz))
    return best if abs(best - freq_hz) <= tol_hz else None


def detect_ctcss(disc_audio: np.ndarray, audio_rate: float):
    """FM AYRIMLAYICI çıkışından CTCSS alt-ses tonunu (60–260 Hz) tespit eder (ED, spec 5.1.2/5.2.3).

    Analog FM telsiz demodüle edildiğinde CTCSS tonu 60–260 Hz bandında dar bir tepe olarak görünür.
    En güçlü bileşen bulunur, standart CTCSS listesine eşlenir. Dönüş:
    {'present', 'freq_hz', 'ctcss_std_hz', 'snr_db'}. Ton yoksa present=False (sahte tespit yok)."""
    x = np.asarray(disc_audio, dtype=np.float64)
    if len(x) < int(audio_rate * 0.1):     # en az ~100 ms gerekir (düşük frekans çözünürlük)
        return {"present": False, "freq_hz": None, "ctcss_std_hz": None, "snr_db": 0.0}
    x = x - np.mean(x)
    w = np.hanning(len(x))
    F = np.abs(np.fft.rfft(x * w))
    fr = np.fft.rfftfreq(len(x), 1.0 / audio_rate)
    band = (fr >= 60.0) & (fr <= 260.0)
    if not np.any(band):
        return {"present": False, "freq_hz": None, "ctcss_std_hz": None, "snr_db": 0.0}
    seg = F[band]
    fseg = fr[band]
    k = int(np.argmax(seg))
    peak = float(seg[k])
    med = float(np.median(seg) + 1e-12)
    snr_db = 20.0 * np.log10(peak / med)
    tone_f = float(fseg[k])
    std = nearest_ctcss(tone_f)
    present = snr_db > 12.0 and std is not None       # kesin: standart tona yakın + belirgin tepe
    return {"present": bool(present), "freq_hz": round(tone_f, 1) if present else None,
            "ctcss_std_hz": std if present else None, "snr_db": round(snr_db, 1)}


def channel_deviation_budget(channel_khz: float = 12.5):
    """Kanal genişliğine göre güvenli (ses, CTCSS) tepe sapma bütçesi (Hz). Toplam, kanal maks.
    sapmasını aşmaz (splatter/komşu-kanal taşması önlenir). 12.5 kHz -> (2000, 500); 25 -> (4000, 750)."""
    if channel_khz <= 12.5:
        return 2000.0, 500.0
    return 4000.0, 750.0


class AnalogDeceptionModulator:
    """Gerçek ses mesajını NBFM/AM ile baseband'e taşıyan aldatma modülatörü."""

    def __init__(self, sample_rate_out: float):
        self.sample_rate = float(sample_rate_out)

    def _resample(self, audio: np.ndarray, rate_in: int) -> np.ndarray:
        """Ses örneğini SDR çıkış hızına yeniden örnekler (polifaz)."""
        if rate_in == self.sample_rate:
            return audio.astype(np.float64)
        # Kesirli oran -> tam sayı up/down (gcd ile)
        from math import gcd
        up = int(self.sample_rate)
        down = int(rate_in)
        g = gcd(up, down)
        up //= g
        down //= g
        # Çok büyük up/down oranlarını sınırla (kararlılık); gerekirse iki aşamalı
        return signal.resample_poly(audio.astype(np.float64), up, down)

    @staticmethod
    def _preemphasis(x: np.ndarray, fs: float, tau: float = 75e-6) -> np.ndarray:
        """FM ses telsizi TX standardı: 1. derece pre-emphasis (6 dB/okt, köşe ~1/(2π·τ)).
        Alıcının de-emphasis'iyle eşleşir -> doğal ses tonu. y[n]=x[n]-a·x[n-1]."""
        a = float(np.exp(-1.0 / (fs * tau)))
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - a * x[:-1]
        m = float(np.max(np.abs(y)))
        return y / m if m > 0 else y

    def modulate(self, audio: np.ndarray, rate_in: int, mode: str = "NBFM",
                 deviation_hz: float = DEFAULT_NBFM_DEVIATION_HZ, amplitude: float = 0.9,
                 max_samples: int = MAX_DECEPTION_SAMPLES, ctcss_hz: float = 0.0,
                 ctcss_dev_hz: float = 500.0, preemphasis: bool = True) -> np.ndarray:
        """Ses mesajını baseband complex64 aldatma sinyaline çevirir (hedef modülasyonuna uygun).

        NBFM: sabit zarf, faz = 2π·∫f_dev  (dar-bant FM ses telsizi). f_dev = sapma·ses(+CTCSS tonu).
        AM:   değişken zarf, (yarım-derinlik) genlik modülasyonu (AM ses telsizi).

        CTCSS (ctcss_hz > 0): hedefin ton-squelch'ini AÇMAK için alt-ses tonu FM sapmasına bindirilir
        (NBFM'de). Ton yoksa hedef hoparlörü açılmaz -> aldatma duyulmaz (spec 5.2.3 kritik nokta).
        Toplam tepe sapma (ses + CTCSS) kanal maks. sapmasını aşmamalı (splatter); çağıran ayarlar.
        Mesaj çok uzunsa max_samples'a kırpılır (donanımda sürekli tekrar loop'lanır)."""
        audio = np.asarray(audio, dtype=np.float64)
        if audio.size < 2:
            return np.zeros(0, dtype=np.complex64)
        mode = (mode or "NBFM").upper()

        # Pre-emphasis (yalnızca FM; ses hızında, resample öncesi -> ucuz + standart)
        if preemphasis and mode != "AM":
            audio = self._preemphasis(audio, float(rate_in))

        # SDR hızına yeniden örnekle + normalize ([-1,1])
        x = self._resample(audio, rate_in)
        peak = float(np.max(np.abs(x)))
        if peak > 1e-9:
            x = x / peak
        if len(x) > max_samples:
            x = x[:max_samples]

        if mode == "AM":
            # AM: pozitif zarf (0.15..1.0), taşıyıcı baseband DC'de -> alıcı zarf-dedektörü sesi çıkarır
            # (CTCSS FM-tabanlıdır; AM ses telsizinde alt-ses squelch tipik değil -> AM'de eklenmez)
            env = 0.575 + 0.425 * x
            buf = env.astype(np.complex64)
        else:  # NBFM (ses amatör telsizi varsayılanı)
            # Anlık FREKANS SAPMASI (Hz): ses + (varsa) CTCSS alt-ses tonu
            f_dev = float(deviation_hz) * x
            if ctcss_hz and ctcss_hz > 0.0:
                t = np.arange(len(x)) / self.sample_rate
                f_dev = f_dev + float(ctcss_dev_hz) * np.sin(2.0 * np.pi * float(ctcss_hz) * t)
            phase = 2.0 * np.pi * np.cumsum(f_dev) / self.sample_rate
            buf = np.exp(1j * phase).astype(np.complex64)

        # Tepe genliğe ölçekle (DAC güvenliği çağıran tarafça enforce edilir)
        buf = (amplitude * buf / (np.max(np.abs(buf)) + 1e-12)).astype(np.complex64)
        return buf
