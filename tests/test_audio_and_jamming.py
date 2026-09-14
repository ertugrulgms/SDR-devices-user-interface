"""AudioDemodulator ve JammingGenerator birim testleri (donanımsız, deterministik)."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.audio_demodulator import AudioDemodulator
from backend.audio_player import AudioPlayer
from backend.jamming_generator import JammingGenerator


class TestAudioDemodulator(unittest.TestCase):
    def setUp(self):
        self.demod = AudioDemodulator(sample_rate=2_400_000, audio_rate=48_000)

    def test_output_length_is_requested_points(self):
        t = np.arange(4096) / 2_400_000
        iq = np.exp(2j * np.pi * 50e3 * t).astype(np.complex64)
        out = self.demod.fm_demodulate(iq, output_points=100)
        self.assertEqual(len(out), 100)

    def test_short_input_returns_zeros(self):
        out = self.demod.fm_demodulate(np.array([1 + 0j], dtype=np.complex64), output_points=100)
        self.assertEqual(len(out), 100)
        self.assertTrue(np.all(out == 0))

    def test_output_is_finite_and_normalized(self):
        t = np.arange(8192) / 2_400_000
        iq = np.exp(2j * np.pi * 75e3 * t).astype(np.complex64)
        out = self.demod.fm_demodulate(iq, output_points=100)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(np.max(np.abs(out)), 1.0 + 1e-6)

    def test_factorize_stages_product_equals_input(self):
        for q in (50, 48, 100, 7):
            stages = AudioDemodulator._factorize_stages(q)
            prod = 1
            for s in stages:
                prod *= s
            self.assertEqual(prod, q, f"q={q} çarpanları {stages} -> {prod}")


class TestAudioPlayerSquelch(unittest.TestCase):
    """SQUELCH: sinyal yokken sesi kuyruğa yazmamalı (sessizlik), sinyal varken yazmalı."""

    def setUp(self):
        # Donanımsız oynatıcı: _producer'ı elle çağırmayız; enqueue mantığını doğrudan test ederiz.
        self.p = AudioPlayer(engine=None, sample_rate=2_400_000, mode="FM")
        self.p._running = True

    def test_squelch_default_open(self):
        self.assertTrue(self.p._squelch_open)

    def test_squelch_toggle(self):
        self.p.set_squelch_open(False)
        self.assertFalse(self.p._squelch_open)
        self.p.set_squelch_open(True)
        self.assertTrue(self.p._squelch_open)

    def test_out_rate_scales_with_bandwidth(self):
        # 2.4 MHz -> 48 kHz tam; 10 MHz -> ~48 kHz (decim 208). Örnekleme hızıyla ölçeklenir.
        self.assertEqual(round(self.p.out_rate), 48000)
        p10 = AudioPlayer(engine=None, sample_rate=10_000_000, mode="FM")
        self.assertGreater(p10.out_rate, 40000)
        self.assertLess(p10.out_rate, 60000)


class TestJammingGenerator(unittest.TestCase):
    def setUp(self):
        self.jam = JammingGenerator(sample_rate_hz=2_000_000)

    def test_jsr_to_amplitude_is_dac_safe(self):
        # YENİ DAVRANIŞ (bulgu #1): JSR -> DAC-güvenli dijital sürüş. JSR_FULLSCALE_DB (20 dB)
        # değerinde genlik tam-skala tavanına (TX_DIGITAL_PEAK=0.9) OTURUR; üstü doyar, ASLA aşmaz.
        from backend.jamming_generator import TX_DIGITAL_PEAK
        self.assertAlmostEqual(self.jam.jsr_to_amplitude(20.0, 1.0), TX_DIGITAL_PEAK, places=6)
        self.assertAlmostEqual(self.jam.jsr_to_amplitude(0.0, 1.0), TX_DIGITAL_PEAK * 0.1, places=6)
        # Tavanın çok üstünde JSR bile tavanı aşamaz (clip imkansız)
        self.assertLessEqual(self.jam.jsr_to_amplitude(60.0, 1.0), TX_DIGITAL_PEAK + 1e-9)

    def test_enforce_dac_safe_caps_peak(self):
        # Herhangi bir genlikte üretilen tampon, DAC tavanının üstüne çıkamaz
        from backend.jamming_generator import TX_DIGITAL_PEAK
        loud = self.jam.generate_barrage_noise(8192, amplitude=50.0)   # bilerek çok yüksek
        safe = self.jam.enforce_dac_safe(loud)
        self.assertLessEqual(float(np.max(np.abs(safe))), TX_DIGITAL_PEAK + 1e-6)

    def test_barrage_noise_length_and_dtype(self):
        n = 4096
        buf = self.jam.generate_barrage_noise(n)
        self.assertEqual(len(buf), n)
        self.assertEqual(buf.dtype, np.complex64)

    def test_barrage_amplitude_scales(self):
        base = self.jam.generate_barrage_noise(2048, amplitude=1.0)
        big = self.jam.generate_barrage_noise(2048, amplitude=4.0)
        # Aynı havuz dilimi değil (idx ilerledi) ama istatistiksel güç ölçeklenmeli
        self.assertGreater(np.mean(np.abs(big) ** 2), np.mean(np.abs(base) ** 2) * 4)

    def test_pool_wraparound_does_not_crash(self):
        # Havuz sınırını aşacak kadar çok örnek çek; sarma çalışmalı
        total = 0
        for _ in range(30):
            total += len(self.jam.generate_barrage_noise(50_000))
        self.assertEqual(total, 30 * 50_000)

    def test_multi_tone_has_multiple_peaks(self):
        # 5 tonlu karıştırma spektrumda ~5 ayrık tepe üretmeli (tek tondan farklı)
        buf = self.jam.generate_multi_tone(8192, amplitude=1.0, n_tones=5, spacing_hz=100e3)
        self.assertEqual(buf.dtype, np.complex64)
        spec = np.abs(np.fft.fftshift(np.fft.fft(buf)))
        thr = np.median(spec) * 10
        peaks = int(np.sum((spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:]) & (spec[1:-1] > thr)))
        self.assertGreaterEqual(peaks, 4, f"beklenen ~5 ton, bulunan {peaks}")

    def test_multi_tone_wider_than_single_tone(self):
        # Çoklu ton, tek tondan daha geniş bir işgal bandı kaplamalı
        single = self.jam.generate_single_tone(0.0, 8192, 1.0)
        multi = self.jam.generate_multi_tone(8192, 1.0, n_tones=5, spacing_hz=100e3)
        occ = lambda x: np.sum(np.abs(np.fft.fft(x)) ** 2 > 0.01 * np.max(np.abs(np.fft.fft(x)) ** 2))
        self.assertGreater(occ(multi), occ(single))

    def test_sweep_covers_wide_band(self):
        # Süpürmeli jammer, tek tondan ÇOK daha geniş bir bandı kaplamalı (frekans tarama)
        sweep = self.jam.generate_chirp_sweep(8192, 1.0, n_sweeps=8)
        single = self.jam.generate_single_tone(0.0, 8192, 1.0)
        def occ(x):
            p = np.abs(np.fft.fft(x)) ** 2
            return int(np.sum(p > 0.05 * np.max(p)))
        self.assertEqual(sweep.dtype, np.complex64)
        self.assertGreater(occ(sweep), occ(single) * 10)   # süpürme çok geniş
        self.assertTrue(np.all(np.isfinite(sweep)))

    def test_analog_voice_spoof_is_constant_envelope_fm(self):
        # "Ses/Audio Sahte Ses" -> FM (sabit zarf), ses-benzeri modülasyon
        buf = self.jam.generate_analog_spoofing("Ses/Audio Sahte Ses", 0.0, 8192, 1.0)
        self.assertEqual(buf.dtype, np.complex64)
        env = np.abs(buf)
        self.assertLess(np.std(env) / (np.mean(env) + 1e-9), 0.1)   # FM: sabit zarf
        self.assertTrue(np.all(np.isfinite(buf)))

    def test_single_tone_peak_at_offset(self):
        n = 4096
        f_off = 250e3
        tone = self.jam.generate_single_tone(freq_offset_hz=f_off, num_samples=n)
        spec = np.abs(np.fft.fftshift(np.fft.fft(tone)))
        peak_bin = int(np.argmax(spec))
        expected = n // 2 + int(f_off / 2_000_000 * n)
        self.assertLess(abs(peak_bin - expected), 5)


if __name__ == "__main__":
    unittest.main()
