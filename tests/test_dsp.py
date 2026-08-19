"""DSPProcessor birim testleri — YÖN BULMA (AoA/nirengi) HARİÇ.

Kapsam: FFT güç spektrumu (Welch), kanal gücü, ölçülen spektrum analizi
(işgal bant genişliği / SNR / spektral düzlük). Bu fonksiyonlar saf ve
deterministiktir; donanım gerektirmez.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dsp_processor import DSPProcessor


class TestComputeFFT(unittest.TestCase):
    def setUp(self):
        self.dsp = DSPProcessor(fft_size=2048)
        self.fs = 1_000_000.0

    def _tone(self, f_offset, n=2048, amp=1.0):
        t = np.arange(n) / self.fs
        return (amp * np.exp(2j * np.pi * f_offset * t)).astype(np.complex64)

    def _raw_fft(self, iq):
        # compute_fft_dbm -> (EMA-yumuşatılmış görüntü, ham spektrum). Analiz/tepe testleri
        # için ham (ikinci) döndürülene bakarız; yumuşatma zaman-durumuna bağlıdır.
        smoothed, raw = self.dsp.compute_fft_dbm(iq)
        return smoothed, raw

    def test_returns_two_arrays_of_fft_size(self):
        smoothed, raw = self._raw_fft(self._tone(100e3))
        self.assertEqual(len(smoothed), 2048)
        self.assertEqual(len(raw), 2048)

    def test_peak_at_known_tone(self):
        # +200 kHz'lik ton; fftshift'li eksende beklenen bin merkez + offset oranı
        f = 200e3
        _, raw = self._raw_fft(self._tone(f))
        peak_bin = int(np.argmax(raw))
        expected_bin = 2048 // 2 + int(f / self.fs * 2048)
        self.assertLess(abs(peak_bin - expected_bin), 40,
                        f"Tepe {peak_bin}, beklenen ~{expected_bin}")

    def test_tone_stands_above_noise_floor(self):
        _, raw = self._raw_fft(self._tone(150e3, amp=1.0))
        self.assertGreater(raw.max() - np.median(raw), 15.0,
                           "Ton gürültü tabanının en az 15 dB üstünde olmalı")

    def test_no_nan_or_inf(self):
        smoothed, raw = self._raw_fft(self._tone(0.0))
        self.assertTrue(np.all(np.isfinite(raw)))
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_short_input_path(self):
        # seg*2'den kısa girişte tek-FFT dalı çalışır, yine fft_size döndürür
        short = self._tone(10e3, n=256)
        _, raw = self._raw_fft(short)
        self.assertEqual(len(raw), 2048)
        self.assertTrue(np.all(np.isfinite(raw)))


class TestChannelPower(unittest.TestCase):
    def setUp(self):
        self.dsp = DSPProcessor(fft_size=2048)

    def test_power_scaling_of_constant_amplitude(self):
        # Sabit |z|=A sinyalin ortalama gücü A^2 -> dBFS = 10log10(A^2) (sahte +30 ofset yok)
        A = 0.5
        z = np.full(4096, A + 0j, dtype=np.complex64)
        expected = 10 * np.log10(A ** 2)
        self.assertAlmostEqual(self.dsp.compute_channel_power_dbm(z), round(expected, 1), places=1)

    def test_higher_amplitude_gives_higher_power(self):
        weak = np.full(1024, 0.1 + 0j, dtype=np.complex64)
        strong = np.full(1024, 1.0 + 0j, dtype=np.complex64)
        self.assertGreater(self.dsp.compute_channel_power_dbm(strong),
                           self.dsp.compute_channel_power_dbm(weak))


class TestAnalyzeSpectrum(unittest.TestCase):
    def setUp(self):
        self.dsp = DSPProcessor(fft_size=2048)
        self.fs = 2_000_000.0

    def test_pure_noise_is_reported_as_no_signal(self):
        # Düşük seviyeli beyaz gürültü: SNR < 6 dB -> "Sinyal Yok"
        rng = np.random.default_rng(0)
        noise = (rng.normal(0, 1e-3, 2048) + 1j * rng.normal(0, 1e-3, 2048)).astype(np.complex64)
        _, fft_dbm = self.dsp.compute_fft_dbm(noise)
        info = self.dsp.analyze_spectrum(fft_dbm, self.fs)
        self.assertEqual(info["occupied_bw_hz"], 0.0)
        self.assertIn("Sinyal Yok", info["signal_class"])

    def test_strong_tone_has_positive_occupied_bw_and_high_snr(self):
        t = np.arange(2048) / self.fs
        tone = np.exp(2j * np.pi * 300e3 * t).astype(np.complex64)
        _, fft_dbm = self.dsp.compute_fft_dbm(tone)
        info = self.dsp.analyze_spectrum(fft_dbm, self.fs)
        self.assertGreater(info["snr_db"], 6.0)
        self.assertGreater(info["occupied_bw_hz"], 0.0)
        # Dar bir ton, örnekleme bandının küçük bir kısmını işgal etmeli
        self.assertLess(info["occupied_bw_hz"], self.fs * 0.5)

    def test_narrowband_tone_lower_flatness_than_wideband_noise(self):
        t = np.arange(2048) / self.fs
        tone = np.exp(2j * np.pi * 100e3 * t).astype(np.complex64)
        rng = np.random.default_rng(1)
        noise = (rng.normal(0, 1, 2048) + 1j * rng.normal(0, 1, 2048)).astype(np.complex64)
        f_tone = self.dsp.analyze_spectrum(self.dsp.compute_fft_dbm(tone)[1], self.fs)
        f_noise = self.dsp.analyze_spectrum(self.dsp.compute_fft_dbm(noise)[1], self.fs)
        # Her ikisi de yeterli SNR üretmeyebilir; sadece flatness tanımlıysa kıyasla
        if f_tone["flatness"] is not None and f_noise["flatness"] is not None:
            self.assertLess(f_tone["flatness"], f_noise["flatness"])


if __name__ == "__main__":
    unittest.main()
