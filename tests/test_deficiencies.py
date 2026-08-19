"""Denetimde bulunan eksikliklerin GİDERİLDİĞİNİ kilitleyen testler.

Kapsam: dBm güç kalibrasyonu, AGC, DDC kanal izolasyonu, kütüphane-tabanlı protokol ID,
kapalı-çevrim akıllı karıştırma (odak), sinyal istihbarat logu. Donanımsız/deterministik.
"""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from backend.dsp_processor import DSPProcessor
from backend.signal_classifier import SignalClassifier
from backend.jamming_generator import JammingGenerator
from backend.gnss_spoofer import GNSSSpoofer
from backend.tx_engine import TxWaveformBuilder
from backend.mission_logger import MissionLogger, _DATA_DIR
from backend.sdr_worker import SDRWorker


class TestPowerCalibration(unittest.TestCase):
    def setUp(self):
        self.d = DSPProcessor(fft_size=2048)
        self.d.set_cal_offset(0.0)          # temiz başla

    def tearDown(self):
        self.d.set_cal_offset(0.0)

    def test_single_point_calibration_maps_to_known_dbm(self):
        iq = (0.5 * np.exp(1j * 2 * np.pi * 0.1 * np.arange(4096))).astype(np.complex64)
        self.d.calibrate_power(iq, known_dbm=-50.0)
        self.assertAlmostEqual(self.d.compute_channel_power_dbm(iq), -50.0, delta=0.2)

    def test_offset_is_persistent(self):
        self.d.set_cal_offset(-44.0)
        d2 = DSPProcessor(fft_size=2048)                 # yeni örnek dosyadan okumalı
        self.assertAlmostEqual(d2.cal_offset_db, -44.0, places=3)

    def test_snr_unaffected_by_calibration(self):
        # Kalibrasyon mutlak ölçeği kaydırır ama SNR (fark) DEĞİŞMEZ
        iq = (np.exp(1j * 2 * np.pi * 0.1 * np.arange(4096)) + 0.01 * np.random.randn(4096)).astype(np.complex64)
        s1 = self.d.analyze_spectrum(self.d.compute_fft_dbm(iq)[1], 2.4e6)["snr_db"]
        self.d.set_cal_offset(-40.0)
        s2 = self.d.analyze_spectrum(self.d.compute_fft_dbm(iq)[1], 2.4e6)["snr_db"]
        self.assertAlmostEqual(s1, s2, delta=0.5)


class TestAGC(unittest.TestCase):
    def setUp(self):
        self.w = SDRWorker()
        self.w.use_hardware = True

        class _Eng:
            def set_gain(self, v): self.g = v
        self.w.engine = _Eng()
        self.w.agc_enabled = True
        self.w._last_agc_time = 0.0

    def test_agc_reduces_gain_near_saturation(self):
        self.w.gain_db = 50.0
        self.w._service_agc(0.95)                        # doygunluk yakın
        self.assertLess(self.w.gain_db, 50.0)

    def test_agc_raises_gain_when_weak(self):
        self.w.gain_db = 30.0
        self.w._service_agc(0.02)                        # zayıf
        self.assertGreater(self.w.gain_db, 30.0)

    def test_agc_disabled_does_nothing(self):
        self.w.agc_enabled = False
        self.w.gain_db = 40.0
        self.w._service_agc(0.99)
        self.assertEqual(self.w.gain_db, 40.0)

    def test_agc_respects_bounds(self):
        from backend.sdr_worker import RX_GAIN_MAX_DB
        self.w.gain_db = RX_GAIN_MAX_DB
        self.w._service_agc(0.02)                        # yükseltmeye çalış ama tavanda
        self.assertLessEqual(self.w.gain_db, RX_GAIN_MAX_DB)


class TestDDCChannelization(unittest.TestCase):
    def test_off_center_signal_isolated(self):
        # Merkez-dışı sinyal DDC ile izole edilip merkezdekiyle aynı sınıflanmalı
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.synth_signals import GENERATORS
        FS, N = 2.4e6, 8192
        clf = SignalClassifier(FS)
        rng = np.random.default_rng(2)
        x = GENERATORS["FM"](N, FS, rng)
        t = np.arange(N) / FS
        x_off = (x * np.exp(1j * 2 * np.pi * 600e3 * t)).astype(np.complex64)
        c = clf.classify(x, check_hopping=False)["modulation"]
        o = clf.classify(x_off, check_hopping=False)["modulation"]
        self.assertEqual(c, o)


class TestProtocolLibrary(unittest.TestCase):
    def setUp(self):
        self.clf = SignalClassifier(56e6)

    def test_wifi_vs_bluetooth_by_bandwidth(self):
        # Aynı bantta (2.4 GHz) BW+mux/ekkt ile Wi-Fi vs Bluetooth ayrımı
        wifi = self.clf.guess_protocol(2437, {"multiplex": "OFDM (Çok Taşıyıcı)", "occupied_bw_hz": 20e6})
        bt = self.clf.guess_protocol(2440, {"ekkt": "FHSS", "occupied_bw_hz": 1e6})
        self.assertIn("Wi-Fi", wifi)
        self.assertIn("Bluetooth", bt)

    def test_fm_broadcast_by_band_and_bw(self):
        p = self.clf.guess_protocol(98, {"modulation": "FM/FSK", "occupied_bw_hz": 180e3})
        self.assertIn("FM Broadcast", p)

    def test_out_of_band_is_honest(self):
        p = self.clf.guess_protocol(50.0, {"occupied_bw_hz": 10e3})
        self.assertIn("Belirlenemedi", p)


class TestSmartJammingFocus(unittest.TestCase):
    def setUp(self):
        fs = 20e6
        self.b = TxWaveformBuilder(JammingGenerator(fs), GNSSSpoofer(fs), fs, block_samples=8192)

    @staticmethod
    def _occ(buf):
        p = np.abs(np.fft.fft(buf)) ** 2
        return int(np.sum(p > 0.05 * np.max(p)))

    def test_focus_concentrates_power(self):
        wide, _ = self.b.build({"mode": "BARRAGE", "jsr_db": 15})
        focused, meta = self.b.build({"mode": "BARRAGE", "jsr_db": 15,
                                      "focus_bw_hz": 2e6, "focus_offset_hz": 0.0})
        self.assertLess(self._occ(focused), self._occ(wide))
        self.assertIn("focus_bw_hz", meta)

    def test_focus_ignored_if_bw_covers_whole_band(self):
        # Odak bandı tüm bandı kaplıyorsa odak uygulanmaz (profil bw kullanılır)
        _, meta = self.b.build({"mode": "BARRAGE", "jsr_db": 15, "focus_bw_hz": 19e6})
        self.assertNotIn("focus_bw_hz", meta)


class TestSignalLog(unittest.TestCase):
    def setUp(self):
        self.db = f"t_siglog_{os.getpid()}.db"
        self.logger = MissionLogger(db_name=self.db)

    def tearDown(self):
        self.logger.close()
        p = os.path.join(_DATA_DIR, self.db)
        if os.path.exists(p):
            os.remove(p)

    def test_signal_log_persists(self):
        self.logger.log_signal(2437.0, "OFDM (Çok Taşıyıcı)", "OFDM", "Yok",
                               "olası Wi-Fi 2.4 GHz, ~20.0 MHz", 20e6, 25.0, 0.7)
        cur = self.logger.conn.cursor()
        cur.execute("SELECT frequency_mhz, modulation, protocol FROM signal_logs")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 2437.0)
        self.assertIn("Wi-Fi", rows[0][2])


if __name__ == "__main__":
    unittest.main()
