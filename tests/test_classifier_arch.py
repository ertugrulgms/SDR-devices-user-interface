"""Sınıflandırıcı mimari iyileştirmeleri — uzman eleştirilerinin düzeltildiğini kilitler.

  #1  FHSS: HoppingHistoryTracker zaman-geçmişinden atlama tespiti (tek blok değil).
  #2  OFDM: yüksek fs'te dar-göreli geniş-bant OFDM artık elenmez; CP penceresi fs ile ölçeklenir.
      Enerji-tetiklemeli akışta get_snapshot yoksa canlı iq'ya düşülür.
Donanımsız/deterministik.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.signal_classifier import SignalClassifier, HoppingHistoryTracker


class TestHoppingHistoryTracker(unittest.TestCase):
    def test_stationary_signal_is_not_fhss(self):
        # Sabit frekanslı sinyal: her kare aynı tepe -> atlama yok
        tr = HoppingHistoryTracker(window_sec=2.0, min_snr_db=10.0)
        t = 0.0
        for _ in range(60):
            tr.update(t, peak_norm=0.5, snr_db=25.0)   # sabit tepe, güçlü
            t += 1 / 30.0
        label, hops, info = tr.detect()
        self.assertEqual(label, "Yok", f"sabit sinyal FHSS sanıldı: {info}")

    def test_hopping_signal_is_fhss(self):
        # Frekans atlayan sinyal: her kare farklı kanal -> FHSS
        tr = HoppingHistoryTracker(window_sec=2.0, min_snr_db=10.0)
        rng = np.random.default_rng(0)
        t = 0.0
        for _ in range(60):
            tr.update(t, peak_norm=float(rng.uniform(0.05, 0.95)), snr_db=25.0)
            t += 1 / 30.0
        label, hops, info = tr.detect()
        self.assertTrue(label.startswith("FHSS"), f"atlayan sinyal yakalanamadı: {info}")
        self.assertGreaterEqual(info["distinct"], 4)

    def test_noise_below_snr_is_ignored(self):
        # Zayıf (gürültü) kareler eşiğin altında -> sayılmaz, FHSS demez
        tr = HoppingHistoryTracker(window_sec=2.0, min_snr_db=10.0)
        rng = np.random.default_rng(1)
        t = 0.0
        for _ in range(60):
            tr.update(t, peak_norm=float(rng.uniform(0, 1)), snr_db=3.0)   # hep gürültü
            t += 1 / 30.0
        label, hops, info = tr.detect()
        self.assertEqual(label, "Yok")

    def test_window_evicts_old_samples(self):
        tr = HoppingHistoryTracker(window_sec=1.0)
        tr.update(0.0, 0.5, 25.0)
        tr.update(5.0, 0.5, 25.0)                 # 5 sn sonra -> eski örnek düşmeli
        self.assertEqual(len(tr._hist), 1)


class TestOfdmHighSampleRate(unittest.TestCase):
    """Uzman #2: yüksek fs'te dar-göreli geniş-bant OFDM elenmemeli."""

    @staticmethod
    def _make_ofdm(fs, n, occ_hz, seed=1):
        rng = np.random.default_rng(seed)
        nfft, ncp = 64, 16
        occ_bins = max(4, int(nfft * occ_hz / fs))
        nsym = n // (nfft + ncp) + 1
        out = []
        for _ in range(nsym):
            X = np.zeros(nfft, dtype=complex)
            idx = np.r_[0:occ_bins // 2, nfft - occ_bins // 2:nfft]
            X[idx] = (rng.integers(0, 2, len(idx)) * 2 - 1) + 1j * (rng.integers(0, 2, len(idx)) * 2 - 1)
            t = np.fft.ifft(X)
            out.append(np.r_[t[-ncp:], t])
        x = np.concatenate(out)[:n].astype(np.complex64)
        return x / (np.max(np.abs(x)) + 1e-9) * 0.5

    def test_wideband_ofdm_in_high_fs_detected(self):
        # 56 MHz örneklemede 20 MHz OFDM (~%36 doluluk) -> OFDM (eski kod "Tek Taşıyıcı" derdi)
        fs = 56e6
        x = self._make_ofdm(fs, 32768, occ_hz=20e6)
        clf = SignalClassifier(fs)
        label, prom, nfft = clf.detect_multiplex(x)
        self.assertTrue(label.startswith("OFDM"), f"yüksek fs OFDM kaçtı: {label} prom={prom}")

    def test_narrowband_psk_not_false_ofdm(self):
        # Aşırı örneklenmiş dar-bant tek-ton benzeri -> OFDM DEĞİL (yanlış pozitif olmamalı)
        fs = 2.4e6
        t = np.arange(8192) / fs
        x = np.exp(1j * 2 * np.pi * 50e3 * t).astype(np.complex64)   # dar-bant CW
        clf = SignalClassifier(fs)
        label, _, _ = clf.detect_multiplex(x)
        self.assertEqual(label, "Tek Taşıyıcı")


if __name__ == "__main__":
    unittest.main()
