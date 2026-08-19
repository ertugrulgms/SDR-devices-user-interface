"""UI birim testleri — headless (offscreen Qt). Gerçek ekran/donanım gerekmez.

Kapsam: widget oluşturma (smoke), analiz/pusula/spektrum güncelleme metodları ve
_to_dms açı biçimlendirme. Amaç: en az test edilen katmanda temel regresyon koruması.
"""
import os
import sys
import unittest
import numpy as np

# GUI'yi görünmez (offscreen) modda çalıştır — import'lardan ÖNCE ayarlanmalı
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from ui.widgets import ControlPanel, AnalysisPanel, SpectrumWidget, DFPanel, PPIWidget
from ui.main_window import _to_dms


class TestWidgetSmoke(unittest.TestCase):
    """Widget'lar hatasız oluşturulabilmeli (offscreen)."""

    def test_all_widgets_construct(self):
        for cls in (ControlPanel, AnalysisPanel, SpectrumWidget, DFPanel, PPIWidget):
            w = cls()
            self.assertIsNotNone(w)


class TestAnalysisPanel(unittest.TestCase):
    def setUp(self):
        self.panel = AnalysisPanel()

    def test_update_field_changes_value(self):
        self.panel.update_field("modulation", "FM/FSK (Frekans Mod.)")
        field = self.panel.analysis_data["modulation"]
        self.assertIn("FM/FSK", field.widget.text())

    def test_unknown_field_id_no_crash(self):
        # Geçersiz alan id -> sessizce yok sayılmalı, çökmemeli
        self.panel.update_field("olmayan_alan", "x")

    def test_expected_fields_exist(self):
        for fid in ("bandwidth", "power", "signal_type", "modulation", "protocol", "multiplex", "ekkt"):
            self.assertIn(fid, self.panel.analysis_data)


class TestPPIWidget(unittest.TestCase):
    def setUp(self):
        self.c = PPIWidget()

    def test_update_with_fix_sets_label(self):
        nodes = {"NODE-MAIN": {"pos": [0, 0, 0], "azimuth_deg": 56.3, "elevation_deg": 18.4,
                               "is_self": True, "stale": False}}
        self.c.update_ppi({"df_nodes": nodes, "df_fix": True, "df_position_xyz_m": [300, 200, 120],
                           "df_target_bearing_deg": 56.3, "df_target_range_m": 360.6,
                           "df_target_elevation_deg": 18.4})
        self.assertIn("HEDEF", self.c.lbl_target.text())

    def test_update_without_fix_shows_waiting(self):
        self.c.update_ppi({})
        self.assertIn("BEKLENİYOR", self.c.lbl_target.text().upper())


class TestSpectrumWidget(unittest.TestCase):
    def test_update_spectrum_sets_data(self):
        sw = SpectrumWidget()
        freqs = np.linspace(94, 100, 2048)
        dbm = np.full(2048, -20.0)
        sw.update_spectrum(freqs, dbm)                 # çökmemeli
        x = sw.signal_line.getData()[0]
        self.assertEqual(len(x), 2048)


class TestToDms(unittest.TestCase):
    def test_positive_angles(self):
        self.assertEqual(_to_dms(45.5), "45° 30'")
        self.assertEqual(_to_dms(120.0), "120° 00'")
        self.assertEqual(_to_dms(0.0), "0° 00'")

    def test_fractional_minutes(self):
        # 100.25° = 100° 15'
        self.assertEqual(_to_dms(100.25), "100° 15'")


if __name__ == "__main__":
    unittest.main()
