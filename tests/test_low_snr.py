"""Düşük-SNR sağlamlığı testleri (uzman eleştirileri). Zayıf sinyal varlığı, BW şişmesi,
self-masking, çoklama SNR kapısı. Donanımsız/deterministik."""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.dsp_processor import DSPProcessor, NoiseFloorTracker
from backend.signal_classifier import SignalClassifier


class TestLowSnrSpectrum(unittest.TestCase):
    def setUp(self):
        self.d = DSPProcessor(fft_size=2048)
        self.fs = 2.4e6

    def test_pure_noise_no_false_positive(self):
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 1e-3, 2048) + 1j * rng.normal(0, 1e-3, 2048)).astype(np.complex64)
        _, fft = self.d.compute_fft_dbm(noise)
        info = self.d.analyze_spectrum(fft, self.fs)               # tek kare, varsayılan eşik
        self.assertIn("Sinyal Yok", info["signal_class"])

    def test_bw_not_inflated_at_low_snr(self):
        # Zayıf DAR-bant sinyal + gürültü: BW gürültüyle şişmemeli (uzman #2). Eşik-üstü bin yayılımı.
        # Deterministik seed (önceden seed'siz np.random -> flaky idi; gürültü tepesi ara sıra eşiği aşıp
        # testi rastgele düşürüyordu).
        rng = np.random.default_rng(0)
        t = np.arange(2048) / self.fs
        iq = (0.05 * np.exp(1j * 2 * np.pi * 200e3 * t)
              + 0.01 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))).astype(np.complex64)
        _, fft = self.d.compute_fft_dbm(iq)
        info = self.d.analyze_spectrum(fft, self.fs)
        # İşgal BW makul dar olmalı (tüm bant 2.4 MHz değil)
        self.assertLess(info["occupied_bw_hz"], 1.0e6)

    def test_wideband_self_masking_detected_with_history(self):
        # Band önce boş, sonra TÜM pencereyi kaplayan sinyal -> tarihsel-min ile SNR yüksek kalmalı
        tr = NoiseFloorTracker()
        tr.update(np.full(2048, -80.0))                            # boş band öğrenildi
        fft = np.full(2048, -35.0)                                 # dev geniş-bant sinyal (band %100)
        nf = tr.update(fft)
        info = self.d.analyze_spectrum(fft, 20e6, noise_floor=nf)
        self.assertNotIn("Sinyal Yok", info["signal_class"])       # medyan körlenirdi; tarihsel-min gördü
        self.assertGreater(info["occupied_bw_hz"], 10e6)           # geniş-bant BW ölçüldü

    def test_present_db_param_controls_sensitivity(self):
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 1e-3, 2048) + 1j * rng.normal(0, 1e-3, 2048)).astype(np.complex64)
        _, fft = self.d.compute_fft_dbm(noise)
        # Yüksek eşik -> gürültü kesin yok; çok düşük eşik -> tek karede gürültü sızabilir (bu yüzden
        # worker zaman-ortalama besleyip düşük eşik kullanır). Parametre çalışmalı:
        self.assertIn("Sinyal Yok", self.d.analyze_spectrum(fft, self.fs, present_db=8.0)["signal_class"])


class TestMultiplexSnrGate(unittest.TestCase):
    def test_low_snr_multiplex_is_undetermined_not_single_carrier(self):
        # Uzman #4: zayıf sinyalde OFDM CP erir -> yanlış "Tek Taşıyıcı" yerine "Belirlenemedi"
        clf = SignalClassifier(2.4e6)
        noise = (np.random.default_rng(3).standard_normal(8192)
                 + 1j * np.random.default_rng(4).standard_normal(8192)).astype(np.complex64) * 0.01
        res = clf.classify(noise, check_hopping=False)
        # düşük SNR -> çoklama "Tek Taşıyıcı" iddia ETMEMELİ
        self.assertNotEqual(res.get("multiplex"), "Tek Taşıyıcı")


if __name__ == "__main__":
    unittest.main()
