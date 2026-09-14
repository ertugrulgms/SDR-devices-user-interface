"""Parametre çıkarımı (şartname 5.1.2) testleri: genişletilmiş çoklama (FDMA/DSSS/TDMA/OFDM)
+ DSSS EKKT + taşıyıcı frekansı + sembol hızı gösterimi. Donanımsız/deterministik."""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from backend.signal_classifier import SignalClassifier
from tests import synth_signals  # noqa (GENERATORS)
from tests.synth_signals import GENERATORS


class TestMultiplexClassification(unittest.TestCase):
    def setUp(self):
        self.N = 8192
        self.rng = np.random.default_rng(0)

    def _clf(self, fs):
        return SignalClassifier(fs)

    def test_single_carrier_not_false_multi(self):
        # QPSK / FM / AM tek taşıyıcıdır -> yanlış FDMA/OFDM olmamalı
        clf = self._clf(2.4e6)
        for name in ("QPSK", "FM", "AM", "BPSK", "16QAM"):
            mux, dsss, _ = clf.analyze_multiplex_ekkt(GENERATORS[name](self.N, 2.4e6, self.rng))
            self.assertEqual(mux, "Tek Taşıyıcı", f"{name} -> {mux}")

    def test_fdma_detected(self):
        fs = 10e6; t = np.arange(self.N) / fs
        def nb(f, seed):
            r = np.random.default_rng(seed)
            sym = r.choice([1, -1, 1j, -1j], self.N // 100 + 1)
            return np.repeat(sym, 100)[:self.N] * np.exp(1j * 2 * np.pi * f * t)
        fdma = sum(nb(f, s) for f, s in [(-3e6, 1), (-1e6, 2), (1e6, 3), (3e6, 4)]).astype(np.complex64)
        fdma += (np.random.randn(self.N) + 1j * np.random.randn(self.N)) * 0.02
        mux, _, _ = self._clf(fs).analyze_multiplex_ekkt(fdma)
        self.assertIn("FDMA", mux)   # "olası FDMA (...)" — dürüst etiket (bağımsız vericiler de olabilir)

    def test_dsss_detected_and_flags_ekkt(self):
        dsss = (np.random.default_rng(1).choice([-1, 1], self.N).astype(float) + 0j).astype(np.complex64)
        mux, dsss_flag, _ = self._clf(2.4e6).analyze_multiplex_ekkt(dsss)
        self.assertIn("DSSS", mux)
        self.assertTrue(dsss_flag)                       # EKKT = DSSS işaretlenir

    def test_tdma_burst_detected(self):
        sig = GENERATORS["QPSK"](self.N, 2.4e6, self.rng).copy()
        sig[self.N // 2:] = 0                            # yarısı kapalı (burst)
        mux, _, _ = self._clf(2.4e6).analyze_multiplex_ekkt(sig)
        self.assertIn("TDMA", mux)

    def test_ofdm_still_detected(self):
        fs = 10e6; nfft, ncp = 64, 16
        rng = np.random.default_rng(2); out = []
        for _ in range(self.N // (nfft + ncp) + 1):
            X = (rng.integers(0, 2, nfft) * 2 - 1) + 1j * (rng.integers(0, 2, nfft) * 2 - 1)
            ti = np.fft.ifft(X); out.append(np.r_[ti[-ncp:], ti])
        ofdm = np.concatenate(out)[:self.N].astype(np.complex64)
        mux, _, _ = self._clf(fs).analyze_multiplex_ekkt(ofdm)
        self.assertTrue(mux.startswith("OFDM"), mux)


class TestParameterFieldsInPanel(unittest.TestCase):
    def test_panel_has_carrier_and_digital_fields(self):
        from ui.widgets import AnalysisPanel
        p = AnalysisPanel()
        for fid in ("carrier", "bandwidth", "power", "signal_type", "modulation",
                    "protocol", "multiplex", "ekkt", "digital"):
            self.assertIn(fid, p.analysis_data, f"panelde '{fid}' alanı yok")


if __name__ == "__main__":
    unittest.main()
