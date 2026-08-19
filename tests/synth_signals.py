"""Sentetik modülasyon üreteçleri — sınıflandırıcı kalibrasyonu ve testleri için.
Hepsi kompleks baseband; bilinen SNR'de gürültü eklenebilir. Donanım gerektirmez."""
import numpy as np


def add_noise(x: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    p_sig = np.mean(np.abs(x) ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = np.sqrt(p_noise / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return (x + noise).astype(np.complex64)


def _upsample(symbols, sps):
    return np.repeat(symbols, sps)


def gen_am(n, fs, fm=2e3, mod_index=0.7):
    t = np.arange(n) / fs
    msg = np.cos(2 * np.pi * fm * t)
    return (1.0 + mod_index * msg).astype(np.complex64)          # reel zarf, faz 0


def gen_fm(n, fs, fm=2e3, freq_dev=60e3):
    t = np.arange(n) / fs
    msg = np.cos(2 * np.pi * fm * t)
    phase = 2 * np.pi * freq_dev * np.cumsum(msg) / fs
    return np.exp(1j * phase).astype(np.complex64)               # sabit zarf, FM


def gen_psk(n, fs, M, sps=32, rng=None):
    rng = rng or np.random.default_rng()
    nsym = n // sps + 1
    syms = rng.integers(0, M, nsym)
    const = np.exp(1j * 2 * np.pi * np.arange(M) / M)
    x = _upsample(const[syms], sps)[:n]
    return x.astype(np.complex64)                                # sabit zarf, faz modülasyonu


def gen_qam16(n, fs, sps=32, rng=None):
    rng = rng or np.random.default_rng()
    nsym = n // sps + 1
    levels = np.array([-3, -1, 1, 3])
    I = rng.choice(levels, nsym)
    Q = rng.choice(levels, nsym)
    s = (I + 1j * Q) / np.sqrt(10.0)
    x = _upsample(s, sps)[:n]
    return x.astype(np.complex64)                                # değişken zarf + faz


def gen_fsk(n, fs, M=2, sps=32, sep=30e3, rng=None):
    rng = rng or np.random.default_rng()
    nsym = n // sps + 1
    syms = rng.integers(0, M, nsym)
    freqs = (np.arange(M) - (M - 1) / 2.0) * sep
    inst = _upsample(freqs[syms], sps)[:n]
    phase = 2 * np.pi * np.cumsum(inst) / fs                     # sürekli faz FSK
    return np.exp(1j * phase).astype(np.complex64)               # sabit zarf, frekans modülasyonu


def gen_ofdm(n, fs, n_fft=64, n_cp=16, rng=None):
    """CP'li OFDM: her sembol = siklik prefix (gövdenin son n_cp örneği) + n_fft'lik IFFT.
    CP, gövdenin kopyası olduğu için N_fft gecikmede güçlü otokorelasyon verir (OFDM imzası)."""
    rng = rng or np.random.default_rng()
    L = n_fft + n_cp
    nsym = n // L + 1
    out = []
    for _ in range(nsym):
        # rastgele QPSK alt-taşıyıcılar
        data = ((rng.integers(0, 2, n_fft) * 2 - 1) + 1j * (rng.integers(0, 2, n_fft) * 2 - 1))
        sym = np.fft.ifft(data) * np.sqrt(n_fft)
        out.append(np.concatenate([sym[-n_cp:], sym]))   # CP + gövde
    return np.concatenate(out)[:n].astype(np.complex64)


def gen_fhss(n, fs, hop_len=256, n_freqs=6, rng=None):
    """Frekans Atlamalı Yayılı Spektrum (FHSS): sinyal, hop_len örnekte bir farklı taşıyıcı
    frekansına sıçrar. Zaman-frekans düzleminde ayrık atlamalar -> FHSS imzası."""
    rng = rng or np.random.default_rng()
    freqs = np.linspace(-0.35, 0.35, n_freqs) * fs
    nhops = n // hop_len + 1
    out = []
    phase = 0.0
    for _ in range(nhops):
        f = rng.choice(freqs)
        t = np.arange(hop_len) / fs
        seg = np.exp(1j * (2 * np.pi * f * t + phase))
        phase = float(np.angle(seg[-1]))          # faz sürekliliği
        out.append(seg)
    return np.concatenate(out)[:n].astype(np.complex64)


# {etiket: üreteç} — kalibrasyon ve test için ortak sözlük
GENERATORS = {
    "AM":    lambda n, fs, rng: gen_am(n, fs),
    "FM":    lambda n, fs, rng: gen_fm(n, fs),
    "BPSK":  lambda n, fs, rng: gen_psk(n, fs, 2, rng=rng),
    "QPSK":  lambda n, fs, rng: gen_psk(n, fs, 4, rng=rng),
    "8PSK":  lambda n, fs, rng: gen_psk(n, fs, 8, rng=rng),
    "16QAM": lambda n, fs, rng: gen_qam16(n, fs, rng=rng),
    "2FSK":  lambda n, fs, rng: gen_fsk(n, fs, 2, rng=rng),
}
