"""GNSSSpoofer, MissionLogger ve HardwareController birim testleri.
Donanım/seri-port gerektirmez; SQLite geçici dosyada, seri port yoksa başarısızlık yolu test edilir."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.gnss_spoofer import GNSSSpoofer
from backend.mission_logger import MissionLogger, _DATA_DIR
from backend.hardware_controller import HardwareController


class TestGNSSSpoofer(unittest.TestCase):
    def setUp(self):
        self.g = GNSSSpoofer(sample_rate_hz=10e6)

    def test_gold_code_length_and_polarity(self):
        code = self.g.generate_gold_code(1)
        self.assertEqual(len(code), 1023)                 # GPS C/A: 1023 çip
        self.assertTrue(set(np.unique(code)).issubset({-1, 1}))

    def test_gold_code_is_balanced(self):
        # Dengeli C/A kodu: +1 ve -1 sayısı neredeyse eşit (|toplam| küçük)
        code = self.g.generate_gold_code(1)
        self.assertLessEqual(abs(int(np.sum(code))), 65)

    def test_different_prn_different_code(self):
        self.assertFalse(np.array_equal(self.g.generate_gold_code(1), self.g.generate_gold_code(2)))

    def test_gold_code_cached(self):
        self.assertIs(self.g.generate_gold_code(3), self.g.generate_gold_code(3))

    def test_unknown_prn_falls_back(self):
        code = self.g.generate_gold_code(99)              # desteklenmeyen -> PRN 1
        self.assertEqual(len(code), 1023)

    def test_baseband_shape_and_finite(self):
        iq = self.g.synthesize_baseband_signal(prn_id=1, num_samples=4096, amplitude=1.0, doppler_hz=1000.0)
        self.assertEqual(len(iq), 4096)
        self.assertTrue(np.iscomplexobj(iq))
        self.assertTrue(np.all(np.isfinite(iq)))
        self.assertGreater(np.max(np.abs(iq)), 0.0)

    def test_doppler_from_coords_physical_range(self):
        # Fiziksel Doppler (bulgu #10): gerçek GPS L1 durağan-alıcı bandı ±~5 kHz içinde olmalı.
        d = self.g.calculate_doppler_from_coords("39.92,32.85")
        self.assertGreaterEqual(d, -5000.0)
        self.assertLessEqual(d, 5000.0)

    def test_doppler_is_deterministic_and_geometric(self):
        # Aynı koordinat aynı Doppler'i vermeli (deterministik, hash-rastgele değil).
        d1 = self.g.calculate_doppler_from_coords("39.92,32.85")
        d2 = self.g.calculate_doppler_from_coords("39.92,32.85")
        self.assertEqual(d1, d2)
        # Zenit yakını (0,0) -> LOS hızı ~0 -> Doppler ~0 (geometrik tutarlılık)
        self.assertAlmostEqual(self.g.calculate_doppler_from_coords("0,0"), 0.0, places=3)

    def test_doppler_invalid_coords_default(self):
        self.assertEqual(self.g.calculate_doppler_from_coords("bozuk"), 1250.0)

    def test_nav_data_polar_and_preamble(self):
        nav = self.g.generate_nav_data(num_bits=50, seed=1)
        self.assertTrue(set(np.unique(nav)).issubset({-1.0, 1.0}))
        # TLM preamble (10001011 -> polar) ilk 8 bit
        np.testing.assert_array_equal(nav[:8], [1, -1, -1, -1, 1, -1, 1, 1])

    def test_baseband_with_nav_data_finite(self):
        nav = self.g.generate_nav_data(seed=1)
        iq = self.g.synthesize_baseband_signal(1, 8192, 1.0, 1000.0, code_phase_chips=100, nav_data=nav)
        self.assertEqual(len(iq), 8192)
        self.assertTrue(np.all(np.isfinite(iq)))
        self.assertGreater(np.max(np.abs(iq)), 0.0)

    def test_all_gnss_services_defined(self):
        # Şartname 5.2.4: GPS/GLONASS/Galileo/Beidou servisleri, hepsi L-bandı frekansları
        svc = GNSSSpoofer.GNSS_SERVICES
        for name in ("GPS L1", "GPS L2", "GPS L5", "GLONASS L1", "GALILEO E1", "BEIDOU B1"):
            self.assertIn(name, svc)
        for name, f in svc.items():
            self.assertTrue(1100 < f < 1700, f"{name} L-bandı dışı: {f}")
        self.assertGreaterEqual(len(svc), 12)   # 12+ servis

    def test_multi_satellite_combines_prns(self):
        # 4 uydu toplamı: tek uydudan farklı (çoklu PRN karışımı) ve geçerli
        multi = self.g.synthesize_multi_satellite([1, 2, 3, 4], 8192, amplitude=1.0)
        single = self.g.synthesize_baseband_signal(1, 8192, 1.0)
        self.assertEqual(len(multi), 8192)
        self.assertTrue(np.all(np.isfinite(multi)))
        self.assertFalse(np.array_equal(multi, single))
        self.assertGreater(np.max(np.abs(multi)), 0.0)


class TestMissionLogger(unittest.TestCase):
    def setUp(self):
        self.db_name = f"test_logs_{os.getpid()}.db"
        self.logger = MissionLogger(db_name=self.db_name)

    def tearDown(self):
        self.logger.close()
        for fn in (self.db_name, "test_report.csv"):
            p = os.path.join(_DATA_DIR, fn)
            if os.path.exists(p):
                os.remove(p)

    def test_log_target_and_export(self):
        self.logger.log_target(freq_mhz=100.0, x_km=1.2, y_km=3.4, tx_mode="BARRAGE", notes="test")
        path = self.logger.export_to_csv("test_report.csv")
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("100.0", content)                   # loglanan frekans CSV'de

    def test_log_df_accuracy_no_crash(self):
        self.logger.log_df_accuracy(reference_deg=90.0, measured_deg=88.5, rms_error_deg=1.5, sample_count=10)

    def test_export_empty_returns_path(self):
        path = self.logger.export_to_csv("test_report.csv")
        self.assertTrue(path.endswith(".csv"))


class TestHardwareController(unittest.TestCase):
    def test_connect_fails_gracefully_without_port(self):
        hw = HardwareController(port="/dev/nonexistent_ttyXYZ")
        self.assertFalse(hw.connect())                    # port yok -> False, çökmeme
        self.assertFalse(hw.is_connected)

    def test_initial_angle_zero(self):
        hw = HardwareController(port="/dev/nonexistent_ttyXYZ")
        self.assertEqual(hw.get_angle(), 0.0)

    def test_scan_commands_no_crash_when_disconnected(self):
        hw = HardwareController(port="/dev/nonexistent_ttyXYZ")
        hw.start_scan()                                   # bağlı değilken sessiz no-op
        hw.stop_scan()
        hw.disconnect()


if __name__ == "__main__":
    unittest.main()
