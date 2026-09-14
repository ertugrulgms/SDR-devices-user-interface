"""Yön Bulma + Konum (DF) testleri — GENLİK TABANLI DF + 3B LOB üçgenleme + ağ füzyonu.

Sentetik iq2/iq3 faz-DF'nin kaldırıldığı, gerçek çok-düğümlü DF'nin çalıştığı doğrulanır.
Donanımsız/deterministik.
"""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from backend.direction_finding import (azel_to_unit, bearing_from_positions, triangulate_lob,
                                       AmplitudeDFEstimator, NodeBearingStore, load_node_registry)
from backend.sdr_worker import SDRWorker


class TestGeometry(unittest.TestCase):
    def test_north_reference(self):
        np.testing.assert_allclose(azel_to_unit(0, 0), [0, 1, 0], atol=1e-9)   # 0°=Kuzey(+y)
        np.testing.assert_allclose(azel_to_unit(90, 0), [1, 0, 0], atol=1e-9)  # 90°=Doğu(+x)

    def test_elevation_component(self):
        u = azel_to_unit(0, 90)                          # tam yukarı
        np.testing.assert_allclose(u, [0, 0, 1], atol=1e-9)

    def test_bearing_from_positions_roundtrip(self):
        az, rng, el = bearing_from_positions([0, 0, 0], [0, 100, 0])
        self.assertAlmostEqual(az, 0.0, places=1)        # kuzeydeki hedef -> 0°
        self.assertAlmostEqual(rng, 100.0, places=1)


class TestTriangulation(unittest.TestCase):
    def test_3d_recovers_airborne_source(self):
        nodes = {"A": [0, 0, 0], "B": [500, 0, 0], "C": [250, 433, 0]}
        src = np.array([300.0, 200.0, 150.0])            # havada
        pos, dirs = [], []
        for p in nodes.values():
            az, rng, el = bearing_from_positions(p, src)
            pos.append(p)
            dirs.append(azel_to_unit(az, el))
        est, res, fix, _cross = triangulate_lob(pos, dirs)
        self.assertTrue(fix)
        self.assertLess(np.linalg.norm(est - src), 1.0)  # <1 m hata
        self.assertLess(res, 1.0)

    def test_2d_when_no_elevation(self):
        # Yükseklik yoksa yer düzlemi çözülür (z=0)
        nodes = {"A": [0, 0, 0], "B": [500, 0, 0]}
        src = np.array([250.0, 300.0, 0.0])
        dirs = [azel_to_unit(*bearing_from_positions(p, src)[::2][:1] + (0.0,)) for p in nodes.values()]
        # yukarıdaki karmaşık; doğrudan azimutla:
        dirs = []
        for p in nodes.values():
            az, _, _ = bearing_from_positions(p, src)
            dirs.append(azel_to_unit(az, 0.0))
        est, res, fix, _cross = triangulate_lob(list(nodes.values()), dirs)
        self.assertTrue(fix)
        self.assertEqual(est[2], 0.0)
        self.assertLess(np.hypot(est[0] - 250, est[1] - 300), 1.0)

    def test_single_lob_no_fix(self):
        est, res, fix, _cross = triangulate_lob([[0, 0, 0]], [azel_to_unit(45, 0)])
        self.assertFalse(fix)

    def test_near_parallel_lobs_rejected_gdop(self):
        # İki düğüm ~aynı yönü gösteriyor (ışınlar neredeyse paralel) -> kötü GDOP: küçük açı hatası
        # devasa konum hatası verir. "Güvenli ama yanlış" konum yerine fix=False dönmeli.
        nodes = [[0, 0, 0], [10, 0, 0]]           # çok kısa taban
        dirs = [azel_to_unit(1.0, 0.0), azel_to_unit(2.0, 0.0)]   # ~1° ayrık -> neredeyse paralel
        est, res, fix, cross = triangulate_lob(nodes, dirs)
        self.assertFalse(fix)                     # zayıf geometri reddedildi
        self.assertLess(cross, 5.0)

    def test_good_geometry_reports_crossing_angle(self):
        # Dik kesişen iki LOB -> yüksek kesişim açısı + fix
        nodes = [[0, 0, 0], [500, 0, 0]]
        src = np.array([250.0, 300.0, 0.0])
        dirs = [azel_to_unit(bearing_from_positions(p, src)[0], 0.0) for p in nodes]
        est, res, fix, cross = triangulate_lob(nodes, dirs)
        self.assertTrue(fix)
        self.assertGreater(cross, 30.0)           # iyi geometri (geniş kesişim)


class TestAmplitudeEstimator(unittest.TestCase):
    def test_peak_azimuth_found(self):
        est = AmplitudeDFEstimator()
        for az in range(0, 360, 2):
            d = ((az - 137 + 180) % 360) - 180
            amp = -90 + 40 * np.cos(np.radians(d)) ** 8 if abs(d) < 90 else -90
            est.update(az, amp, now=0.0)
        b, pk, conf, n = est.bearing(now=0.0)
        self.assertIsNotNone(b)
        self.assertLess(abs(((b - 137 + 180) % 360) - 180), 3.0)   # <3° hata
        self.assertGreater(conf, 0.5)

    def test_insufficient_data_returns_none(self):
        est = AmplitudeDFEstimator()
        est.update(10, -50, now=0.0)
        b, _, _, _ = est.bearing(now=0.0)
        self.assertIsNone(b)

    def test_stale_samples_pruned(self):
        est = AmplitudeDFEstimator(decay_sec=2.0)
        for az in (10, 20, 30):
            est.update(az, -50, now=0.0)
        b, _, _, n = est.bearing(now=100.0)               # çok sonra -> hepsi bayat
        self.assertEqual(n, 0)


class TestNodeStore(unittest.TestCase):
    def test_valid_json_accepted(self):
        s = NodeBearingStore()
        self.assertTrue(s.update_from_json({"id": "NODE-2", "azimuth_deg": 137.5,
                                            "elevation_deg": 12.0, "amp_dbm": -52.0}))
        act = s.active_bearings()
        self.assertIn("NODE-2", act)
        self.assertAlmostEqual(act["NODE-2"]["azimuth_deg"], 137.5)

    def test_invalid_json_rejected(self):
        s = NodeBearingStore()
        self.assertFalse(s.update_from_json({"id": "X"}))          # azimut yok
        self.assertFalse(s.update_from_json({"azimuth_deg": 90}))  # id yok
        self.assertFalse(s.update_from_json({"id": "X", "azimuth_deg": "abc"}))

    def test_stale_bearing_excluded_from_active(self):
        s = NodeBearingStore(stale_sec=0.0)
        s.update_from_json({"id": "NODE-2", "azimuth_deg": 90})
        import time as _t
        _t.sleep(0.01)
        self.assertNotIn("NODE-2", s.active_bearings())           # bayat -> füzyona girmez
        self.assertIn("NODE-2", s.snapshot())                     # ama izlemede görünür (stale=True)


class TestWorkerDFIntegration(unittest.TestCase):
    def test_network_bearings_triangulate_in_worker(self):
        w = SDRWorker()
        # Donanımdan bağımsız olsun: bu makineye bir ESP32 enkoder TAKILIYSA worker onu otomatik
        # bağlar ve kendi self-kerterizini (sinyalsiz -> None) üretip enjekte edilen NODE-MAIN
        # kerterizini siler. Test 3 AĞ kerterizini üçgenlemeyi doğruluyor -> encoder'ı ayır.
        w.hw_ctrl.disconnect()
        w.hw_ctrl.is_connected = False
        src = [300.0, 200.0, 120.0]
        for nid, cfg in w.df_registry.items():
            az, rng, el = bearing_from_positions(cfg["pos"], src)
            w.node_store.update_from_json({"id": nid, "azimuth_deg": az, "elevation_deg": el,
                                           "amp_dbm": -55.0})
        df = w._run_direction_finding(np.ones(2048, dtype=np.complex64) * 0.01)
        self.assertTrue(df["fix"])
        self.assertEqual(df["active_count"], 3)
        est = df["position_xyz_m"]
        self.assertLess(abs(est[0] - 300), 2.0)
        self.assertLess(abs(est[1] - 200), 2.0)
        self.assertLess(abs(est[2] - 120), 2.0)

    def test_no_synthetic_iq_channels(self):
        # Sentetik iq2/iq3 kaldırıldı: worker artık bu alanları üretmemeli
        w = SDRWorker()
        self.assertFalse(hasattr(w, "network_data"))     # eski sahte ağ verisi kaldırıldı


class TestDFFixLogging(unittest.TestCase):
    def test_df_fix_persists_with_node_bearings(self):
        import os
        from backend.mission_logger import MissionLogger, _DATA_DIR
        db = f"t_dffixlog_{os.getpid()}.db"
        lg = MissionLogger(db_name=db)
        try:
            lg.log_df_fix(433.0, [300.0, 200.0, 120.0], 0.5, 3, 56.3, 360.6, 18.4,
                          {"NODE-2": {"azimuth_deg": 315.0, "elevation_deg": 23.0}})
            cur = lg.conn.cursor()
            cur.execute("SELECT pos_x_m, node_count, nodes_json FROM df_fix_logs")
            rows = cur.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], 3)
            self.assertIn("NODE-2", rows[0][2])          # düğüm kerterizi JSON olarak saklandı
        finally:
            lg.close()
            p = os.path.join(_DATA_DIR, db)
            if os.path.exists(p):
                os.remove(p)


class TestDFFixesRegression(unittest.TestCase):
    """Denetimde bulunan DF eksikliklerinin düzeltildiğini kilitler."""

    def test_amplitude_df_resets_on_retune(self):
        # Frekans değişince eski azimut-genlik haritası (kerteriz) sıfırlanmalı
        w = SDRWorker()
        for az in range(0, 360, 5):
            w.self_amp_df.update(az, -50.0, now=0.0)
        self.assertGreater(len(w.self_amp_df._bins), 0)
        w.set_frequency(433.0)
        self.assertEqual(len(w.self_amp_df._bins), 0)

    def test_unknown_node_id_filtered(self):
        # Registry'de olmayan id füzyona/PPI'ya girmemeli (merkeze çizilip yanıltmamalı)
        w = SDRWorker()
        w.node_store.update_from_json({"id": "GHOST-9", "azimuth_deg": 45.0})
        df = w._run_direction_finding(np.ones(2048, dtype=np.complex64) * 0.01)
        self.assertNotIn("GHOST-9", df["nodes"])

    def test_network_node_bearing_geometry(self):
        import backend.network_node as nn
        az, el = nn.bearing_to([0, 0, 0], [0, 100, 0])   # kuzeydeki hedef
        self.assertAlmostEqual(az, 0.0, places=1)
        self.assertAlmostEqual(el, 0.0, places=1)

    def _fake_encoder(self, w):
        class _HW:
            is_connected = True
            def __init__(self): self.a = 0.0
            def get_angle(self): return self.a
        hw = _HW()
        w.hw_ctrl = hw
        return hw

    def test_snr_gate_blocks_noise_pollution(self):
        # Şartname senaryosu: kaynaklar SIRAYLA yayın yapar -> hedef çoğu zaman KAPALI. Kaynak
        # sustuğunda gürültü tepesi genlik-DF haritasını kirletip sahte kerteriz üretmemeli.
        w = SDRWorker()
        hw = self._fake_encoder(w)
        # 1) Sinyal VAR: 137° civarı ışın taraması (yüksek SNR) -> kerteriz oluşur
        for az in range(110, 165, 2):
            hw.a = az
            d = ((az - 137 + 180) % 360) - 180
            peak = -90 + 40 * np.cos(np.radians(d)) ** 8 if abs(d) < 90 else -90
            fft = np.full(2048, -100.0); fft[1000] = peak
            w._last_fft_dbm = fft
            w._run_direction_finding(np.ones(2048, np.complex64) * 0.01, {"snr_db": peak + 100})
        b1, _, _, _ = w.self_amp_df.bearing()
        self.assertIsNotNone(b1)
        self.assertLess(abs(((b1 - 137 + 180) % 360) - 180), 3.0)
        bins_before = len(w.self_amp_df._bins)
        # 2) Sinyal YOK: farklı azimutlarda eşik-altı SNR -> HARİTAYA yeni bin EKLENMEMELİ
        for az in range(200, 260, 2):
            hw.a = az
            fftn = np.full(2048, -100.0); fftn[1500] = -96.0
            w._last_fft_dbm = fftn
            w._run_direction_finding(np.ones(2048, np.complex64) * 0.01, {"snr_db": 4.0})
        self.assertEqual(len(w.self_amp_df._bins), bins_before)   # kirlenme yok
        b2, _, _, _ = w.self_amp_df.bearing()
        self.assertLess(abs(((b2 - 137 + 180) % 360) - 180), 3.0)  # kerteriz korundu

    def test_robust_peak_power_integrates_window(self):
        # Tek-bin max'e göre tepe etrafı entegre güç, izole gürültü binine daha dayanıklı olmalı
        w = SDRWorker()
        fft = np.full(2048, -100.0)
        fft[1000] = -50.0; fft[999] = -52.0; fft[1001] = -52.0   # gerçek tepe (yayılmış)
        p = w._robust_peak_power_dbm(fft)
        self.assertGreater(p, -60.0)     # tepe civarı güç yüksek
        self.assertLess(p, -49.0)        # ama tek-bin -50'yi aşmaz (ortalama)

    def test_df_locks_target_channel_not_stronger_interferer(self):
        # HEDEF-KANAL (uzman P0.3): hedef GÜNEYDE (180°), MERKEZDE. Bandın başka yerinde SÜREKLİ ve
        # DAHA GÜÇLÜ bir parazit var. Ana DF, GLOBAL tepeye (parazit) DEĞİL hedef kanalına kilitlenmeli.
        w = SDRWorker()
        hw = self._fake_encoder(w)
        for az in range(150, 211, 2):
            hw.a = az
            d = ((az - 180 + 180) % 360) - 180
            tgt = -90 + 40 * np.cos(np.radians(d)) ** 8 if abs(d) < 90 else -90   # hedef ışını (merkez)
            fft = np.full(2048, -100.0)
            fft[1024] = tgt          # HEDEF: merkez (tune edilen frekans)
            fft[1400] = -35.0        # PARAZİT: merkez-dışı, hedeften GÜÇLÜ, açıdan bağımsız
            w._last_fft_dbm = fft
            w._run_direction_finding(np.ones(2048, np.complex64) * 0.01, {"snr_db": 60.0})
        b, _, _, _ = w.self_amp_df.bearing()
        self.assertIsNotNone(b)
        self.assertLess(abs(((b - 180 + 180) % 360) - 180), 6.0)   # güneye kilitlendi (parazite kaymadı)

    def test_target_power_ignores_far_interferer(self):
        # _target_power_dbm: merkezdeki ZAYIF hedefi ölçmeli, ±80 kHz DIŞINDAKİ güçlü parazite atlamamalı
        w = SDRWorker(); w.set_bandwidth(2.4)
        fft = np.full(2048, -100.0)
        fft[1024] = -55.0        # merkez hedef (zayıf)
        fft[1600] = -30.0        # uzak parazit (çok güçlü, ±80 kHz dışında)
        p = w._target_power_dbm(fft)
        self.assertLess(p, -45.0)    # hedefi (~-55) ölçtü, parazite (-30) atlamadı


class TestProtocolSymbolRate(unittest.TestCase):
    def test_baud_distinguishes_dmr(self):
        from backend.signal_classifier import SignalClassifier
        clf = SignalClassifier(2.4e6)
        # UHF telsiz bandı + PSK + DMR baud (~4.8 kBd) -> DMR/P25 etiketi
        p = clf.guess_protocol(450.0, {"modulation": "PSK (Sayısal-Faz)",
                                       "occupied_bw_hz": 12e3, "symbol_rate_hz": 4800.0})
        self.assertIn("DMR", p)


class TestPPIWidget(unittest.TestCase):
    def test_ppi_updates_and_resets(self):
        from ui.widgets import PPIWidget
        w = PPIWidget()
        nodes = {"NODE-MAIN": {"pos": [0, 0, 0], "azimuth_deg": 56.3, "elevation_deg": 18.4,
                               "is_self": True, "stale": False}}
        w.update_ppi({"df_nodes": nodes, "df_fix": True, "df_position_xyz_m": [300, 200, 120],
                      "df_target_bearing_deg": 56.3, "df_target_range_m": 360.6,
                      "df_target_elevation_deg": 18.4})
        self.assertIn("HEDEF", w.lbl_target.text())
        w.update_ppi({})                                 # boş -> çökmemeli
        self.assertIn("BEKLENİYOR", w.lbl_target.text())


if __name__ == "__main__":
    unittest.main()
