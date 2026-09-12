"""SignalClassifier (AMC) birim testleri — sentetik sinyallerle.

KAPSAM (şimdilik): SADECE AM ve FM (analog). Dijital modülasyon türleri (PSK/QAM/OFDM/
FSK/GMSK/CSS) test cihazı olmadığı için sınıflandırıcıda KAPALIDIR; ilgili testler de
aşağıda yorum satırındadır (ileride dijital açılınca geri açılacak).

Doğrulama stratejisi:
  * Yeterli SNR'de (>=15 dB) AM -> "AM (Analog-Genlik)", FM -> "FM (Analog-Frekans)".
  * Saf gürültüde dürüstçe "Belirlenemedi" dönmeli (asla uydurma sınıf).
  * Güven skoru [0,1] aralığında olmalı.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.signal_classifier import SignalClassifier
from tests.synth_signals import GENERATORS, add_noise

FS = 2.4e6
N = 8192


# Modülasyon çıktı etiketleri. FM/2FSK -> "FM/FSK (Frekans Mod.)" grubu döner (analog FM ↔ sayısal
# FSK ayrımı SESLE, worker._refine_fm_fsk ile yapılır; sınıflandırıcı özellikten güvenilir ayıramaz).
_EXPECTED = {
    "AM":    "AM (Analog-Genlik)",
    "FM":    "FM/FSK (Frekans Mod.)",
    "2FSK":  "FM/FSK (Frekans Mod.)",
    "BPSK":  "BPSK (Sayısal-Faz)",
    "QPSK":  "QPSK (Sayısal-Faz)",
    "8PSK":  "8PSK (Sayısal-Faz)",
    "16QAM": "QAM (Sayısal-Genlik)",
}


def _expected(label):
    return _EXPECTED[label]


def _majority_vote(label, snr_db, trials=9, seed=0):
    rng = np.random.default_rng(seed)
    clf = SignalClassifier(FS)
    votes = {}
    for _ in range(trials):
        x = add_noise(GENERATORS[label](N, FS, rng), snr_db, rng)
        mod = clf.classify(x)["modulation"]
        votes[mod] = votes.get(mod, 0) + 1
    return max(votes, key=votes.get)


class TestModulationClassification(unittest.TestCase):
    def test_am_high_snr(self):
        self.assertEqual(_majority_vote("AM", 18), _expected("AM"))

    def test_fm_high_snr(self):
        self.assertEqual(_majority_vote("FM", 18), _expected("FM"))

    # --- DİJİTAL TÜRLER (ampirik kalibrasyonla açıldı — 13-25 dB'de %100 ayrım) ---------
    def test_bpsk_high_snr(self):
        self.assertEqual(_majority_vote("BPSK", 18), _expected("BPSK"))

    def test_qpsk_high_snr(self):
        self.assertEqual(_majority_vote("QPSK", 18), _expected("QPSK"))

    def test_8psk_high_snr(self):
        self.assertEqual(_majority_vote("8PSK", 18), _expected("8PSK"))

    def test_qam_high_snr(self):
        self.assertEqual(_majority_vote("16QAM", 18), _expected("16QAM"))

    def test_fsk_maps_to_frequency_family(self):
        # 2FSK sabit-zarf açı modülasyonu -> "FM/FSK" grubu (analog/sayısal ayrımı sesle)
        self.assertEqual(_majority_vote("2FSK", 18), _expected("2FSK"))


class TestHonesty(unittest.TestCase):
    def test_pure_noise_is_belirlenemedi(self):
        # Sinyal yok, saf gürültü -> düşük SNR -> "Belirlenemedi"
        rng = np.random.default_rng(3)
        noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64)
        res = SignalClassifier(FS).classify(noise)
        self.assertIn("Belirlenemedi", res["modulation"])

    def test_confidence_in_range(self):
        rng = np.random.default_rng(4)
        clf = SignalClassifier(FS)
        for label in ("AM", "FM"):
            x = add_noise(GENERATORS[label](N, FS, rng), 18, rng)
            c = clf.classify(x)["confidence"]
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_no_crash_short_input(self):
        res = SignalClassifier(FS).classify(np.zeros(64, dtype=np.complex64))
        self.assertIn("Belirlenemedi", res["modulation"])


# --- ÇOKLAMA (OFDM) ve EKKT (FHSS) TESTLERİ ŞİMDİLİK KAPALI --------------------------
# Dijital çoklama/yayılı-spektrum tespiti sınıflandırıcıda kapalı olduğundan bu testler de
# devre dışı. Dijital türler açılınca (signal_classifier.classify içindeki blokla birlikte)
# aşağıdaki TestMultiplex / TestEKKT sınıflarını geri açın.
#
# class TestMultiplex(unittest.TestCase):
#     ...  (OFDM CP otokorelasyon, Tek Taşıyıcı ayrımı)
# class TestEKKT(unittest.TestCase):
#     ...  (FHSS frekans-atlama tespiti)


if __name__ == "__main__":
    unittest.main()
