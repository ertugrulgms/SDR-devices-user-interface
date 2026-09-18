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
        w.self_amp_df.set_freq(None)   # bu test SNR/kirlenme kapısını sınar (pattern değil); sentetik
                                       # cos^8 hüzme + dar yay pattern-eşleşmeye uygun değil -> kapat
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
        w.self_amp_df.set_freq(None)   # hedef-kanal seçimini sınar (pattern değil); sentetik dar yay
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


class TestPatternMatchedDF(unittest.TestCase):
    """ÖLÇÜLEN VNA pattern'iyle pattern-eşleştirmeli DF (şartname 5.1.4). data/antenna_pattern.json
    yoksa testler dürüstçe atlanır (skip). Örnekler GERÇEK pattern'den üretilir (test-zamanı, ÇALIŞMA
    ZAMANI DEĞİL) + gürültü -> kerteriz geri kazanımı ve centroid'e üstünlük doğrulanır."""

    @classmethod
    def setUpClass(cls):
        from backend.antenna_pattern import shared_pattern
        cls.ap = shared_pattern()
        if not cls.ap.available():
            raise unittest.SkipTest("data/antenna_pattern.json yok — pattern testleri atlandı")

    def _sweep_from_pattern(self, freq_hz, true_bearing, noise_db, rng, step=5.0):
        ang, pref, fb, pk = self.ap.pattern_at(freq_hz)
        ae = np.concatenate([ang - 360, ang, ang + 360]); pe = np.concatenate([pref, pref, pref])
        encs = np.arange(0, 360, step)
        power = np.interp((true_bearing - encs + pk) % 360, ae, pe) + rng.normal(0, noise_db, len(encs))
        return encs, power

    def test_pattern_beats_centroid_on_directional_freq(self):
        rng = np.random.default_rng(1)
        f = 1575e6                                    # yönlü frekans (ön/arka ~14 dB)
        perr, cerr = [], []
        for _ in range(40):
            tb = rng.uniform(0, 360)
            encs, power = self._sweep_from_pattern(f, tb, 2.0, rng)
            m = self.ap.match(encs, power, f)
            self.assertIsNotNone(m)
            perr.append(abs(((m["bearing_deg"] - tb + 180) % 360) - 180))
            est = AmplitudeDFEstimator(use_pattern=False)
            for e, p in zip(encs, power):
                est.update(e, p, now=0.0)
            cb, _, _, _ = est.bearing(now=0.0)
            cerr.append(abs(((cb - tb + 180) % 360) - 180))
        prms = float(np.sqrt(np.mean(np.square(perr))))
        crms = float(np.sqrt(np.mean(np.square(cerr))))
        self.assertLess(prms, 5.0, f"pattern RMS {prms:.1f}° çok yüksek")
        self.assertLess(prms, crms, f"pattern ({prms:.1f}°) centroid'i ({crms:.1f}°) geçemedi")

    def test_low_frontback_flagged_not_quality_ok(self):
        # 915 MHz civarı ön/arka düşük -> quality_ok False olmalı (çağıran σ'yı şişirir/centroid'e düşer)
        rng = np.random.default_rng(2)
        encs, power = self._sweep_from_pattern(915e6, 120.0, 2.0, rng)
        m = self.ap.match(encs, power, 915e6)
        self.assertIsNotNone(m)
        self.assertFalse(m["quality_ok"])

    def test_estimator_uses_pattern_when_freq_set(self):
        rng = np.random.default_rng(3)
        est = AmplitudeDFEstimator(freq_hz=1900e6)
        encs, power = self._sweep_from_pattern(1900e6, 47.0, 1.5, rng)
        for e, p in zip(encs, power):
            est.update(e, p, now=0.0)
        b, _, _, _ = est.bearing(now=0.0)
        self.assertEqual(est._last_method, "pattern")
        self.assertLess(abs(((b - 47 + 180) % 360) - 180), 5.0)
        self.assertLess(est.last_sigma(), 8.0)         # pattern σ centroid tabanından küçük

    def test_falls_back_to_centroid_without_freq(self):
        est = AmplitudeDFEstimator(freq_hz=None)       # frekans yok -> pattern kapalı
        for az in range(0, 360, 5):
            d = ((az - 200 + 180) % 360) - 180
            est.update(az, -90 + 40 * np.cos(np.radians(d)) ** 8 if abs(d) < 90 else -90, now=0.0)
        b, _, _, _ = est.bearing(now=0.0)
        self.assertEqual(est._last_method, "centroid")
        self.assertLess(abs(((b - 200 + 180) % 360) - 180), 5.0)

    def test_weighted_triangulation_downweights_uncertain_node(self):
        # İki hassas (küçük σ) + bir çok belirsiz/yanlış (büyük σ) kerteriz -> ağırlıklı çözüm
        # hassaslara yakın olmalı (belirsiz olan konumu bozmamalı).
        pos = [[0, 0, 0], [500, 0, 0], [250, 400, 0]]
        # Gerçek hedef (250, 200). İlk iki düğüm doğru bakar, üçüncü 40° yanlış.
        tgt = np.array([250.0, 200.0, 0.0])
        dirs, sig = [], []
        for i, p in enumerate(pos):
            az, _, _ = bearing_from_positions(p, tgt)
            if i == 2:
                az += 40.0; sig.append(30.0)          # yanlış + belirsiz
            else:
                sig.append(1.0)                        # hassas
            dirs.append(azel_to_unit(az, 0.0))
        x_w, _, _, _ = triangulate_lob(pos, dirs, sigmas=sig)
        x_u, _, _, _ = triangulate_lob(pos, dirs)      # eşit ağırlık
        err_w = np.linalg.norm(x_w[:2] - tgt[:2])
        err_u = np.linalg.norm(x_u[:2] - tgt[:2])
        self.assertLess(err_w, err_u)                  # ağırlıklı, eşit-ağırlıktan daha isabetli

    # --- İLERİ-YAY KAPISI TESTLERİ (arkada alan yok senaryosu) ---------------------------
    def test_forward_gate_off_is_identical(self):
        # Kapı KAPALI iken davranış birebir aynı olmalı (gate None -> tam tur)
        rng = np.random.default_rng(20)
        encs, power = self._sweep_from_pattern(1575e6, 70.0, 1.5, rng)
        m_off = self.ap.match(encs, power, 1575e6)
        m_none = self.ap.match(encs, power, 1575e6, arc_center=None, arc_half=None)
        self.assertEqual(m_off["bearing_deg"], m_none["bearing_deg"])

    def test_forward_gate_rejects_rear_peak(self):
        # Hedef önde (137°), ARKADA çok güçlü sahte tepe (330°). Kapı açıkken kerteriz önde kalmalı.
        est = AmplitudeDFEstimator(use_pattern=False)
        for az in range(0, 360, 5):
            d = ((az - 137 + 180) % 360) - 180
            est.update(az, -90 + 40 * np.cos(np.radians(d)) ** 8 if abs(d) < 90 else -90, now=0.0)
        for az in range(310, 350, 5):
            est.update(az, -20.0, now=0.0)              # arkada çok güçlü parazit
        est.set_forward_gate(140.0, 60.0)              # ileri yay 80..200
        b_gated, _, _, _ = est.bearing(now=0.0)
        self.assertLess(abs(((b_gated - 137 + 180) % 360) - 180), 6.0)   # önde kaldı
        est.clear_forward_gate()
        b_open, _, _, _ = est.bearing(now=0.0)
        self.assertGreater(abs(((b_open - 137 + 180) % 360) - 180), 60.0)  # kapalıyken arkaya kaydı

    def test_forward_gate_enables_pattern_on_narrow_sweep(self):
        # Köşe istasyon yalnızca ~90° tarasa: kapı YOKken kapsama<180 -> pattern kapalı; kapı VARken açık.
        rng = np.random.default_rng(21)
        ang, pref, fb, pk = self.ap.pattern_at(1575e6)
        ae = np.concatenate([ang - 360, ang, ang + 360]); pe = np.concatenate([pref, pref, pref])
        encs = np.arange(0, 125, 5.0)                  # ~120° dar yay
        power = np.interp((40.0 - encs + pk) % 360, ae, pe) + rng.normal(0, 1.5, len(encs))
        self.assertFalse(self.ap.match(encs, power, 1575e6)["quality_ok"])          # yaysız: kapsama düşük
        m = self.ap.match(encs, power, 1575e6, arc_center=60.0, arc_half=60.0)
        self.assertTrue(m["quality_ok"])                                            # yaylı: pattern açık
        self.assertLess(abs(((m["bearing_deg"] - 40 + 180) % 360) - 180), 5.0)

    def test_forward_gate_bearing_stays_in_arc(self):
        # Yay dışına düşen bir çözüm dönmemeli (arka kaynakta bile bearing yay içinde kalır)
        rng = np.random.default_rng(22)
        ang, pref, fb, pk = self.ap.pattern_at(1575e6)
        ae = np.concatenate([ang - 360, ang, ang + 360]); pe = np.concatenate([pref, pref, pref])
        encs = np.arange(0, 125, 5.0)
        power = np.interp((220.0 - encs + pk) % 360, ae, pe) + rng.normal(0, 1.5, len(encs))  # arka kaynak
        m = self.ap.match(encs, power, 1575e6, arc_center=60.0, arc_half=60.0)
        diff = abs(((m["bearing_deg"] - 60.0 + 180) % 360) - 180)
        self.assertLessEqual(diff, 60.0 + 1e-6)        # bearing 0..120 yay içinde

    # --- UZMAN İNCELEMESİYLE EKLENEN KAPI/DAYANIKLILIK TESTLERİ ---------------------------
    def test_circular_boundary_bearings(self):
        # 0°/359° dolanma sınırında kerteriz doğru geri kazanılmalı
        rng = np.random.default_rng(10)
        for tb in (0.0, 1.0, 179.0, 180.0, 181.0, 359.0):
            encs, power = self._sweep_from_pattern(1575e6, tb, 1.0, rng)
            m = self.ap.match(encs, power, 1575e6)
            self.assertIsNotNone(m)
            err = abs(((m["bearing_deg"] - tb + 180) % 360) - 180)
            self.assertLess(err, 5.0, f"tb={tb} -> {m['bearing_deg']} (hata {err:.1f}°)")

    def test_incomplete_sweep_low_coverage_not_quality_ok(self):
        # Sadece dar bir yay (~85°) tarandıysa pattern eşleştirme GÜVENİLMEZ -> quality_ok False
        rng = np.random.default_rng(11)
        ang, pref, fb, pk = self.ap.pattern_at(1575e6)
        ae = np.concatenate([ang - 360, ang, ang + 360]); pe = np.concatenate([pref, pref, pref])
        encs = np.arange(100, 185, 5.0)                # ~85° yay
        power = np.interp((140.0 - encs + pk) % 360, ae, pe) + rng.normal(0, 1.0, len(encs))
        m = self.ap.match(encs, power, 1575e6)
        self.assertIsNotNone(m)
        self.assertLess(m["coverage_deg"], 180.0)
        self.assertFalse(m["quality_ok"])              # kapsama yetersiz -> güvenilmez

    def test_estimator_falls_back_to_centroid_on_partial_sweep(self):
        # Dar yayda estimator pattern'i KULLANMAMALI, centroid'e düşmeli (gerçek fallback — P0)
        est = AmplitudeDFEstimator(freq_hz=1575e6)
        rng = np.random.default_rng(12)
        ang, pref, fb, pk = self.ap.pattern_at(1575e6)
        ae = np.concatenate([ang - 360, ang, ang + 360]); pe = np.concatenate([pref, pref, pref])
        for az in np.arange(100, 185, 5.0):
            p = float(np.interp((140.0 - az + pk) % 360, ae, pe)) + rng.normal(0, 1.0)
            est.update(az, p, now=0.0)
        est.bearing(now=0.0)
        self.assertIn("centroid", est.last_method())   # pattern DEĞİL
        self.assertGreater(est.last_coverage(), 0.0)

    def test_out_of_range_frequency_rejected(self):
        # Kalibrasyon aralığı (300 MHz–3 GHz) dışında pattern KULLANILMAMALI (uç-noktaya sıçrama yok)
        self.assertFalse(self.ap.in_range(150e6))
        self.assertFalse(self.ap.in_range(4000e6))
        self.assertIsNone(self.ap.pattern_at(150e6))
        rng = np.random.default_rng(13)
        # 1575 MHz sweep'ini 150 MHz frekansla eşleştirmeye çalış -> None (aralık dışı)
        encs, power = self._sweep_from_pattern(1575e6, 47.0, 1.0, rng)
        self.assertIsNone(self.ap.match(encs, power, 150e6))

    def test_frequency_interpolation_between_slices(self):
        # İki kalibre dilim arasındaki frekans, ikisinin GÜÇ-ortalamasına yakın bir pattern vermeli
        f = np.array(self.ap._freqs)
        i = len(f) // 2
        f_mid = float((f[i] + f[i + 1]) / 2.0)
        _, p_mid, _, _ = self.ap.pattern_at(f_mid)
        _, p0, _, _ = self.ap.pattern_at(float(f[i]))
        _, p1, _, _ = self.ap.pattern_at(float(f[i + 1]))
        # güç alanında ortalama (normalize öncesi kabaca); interpolasyon iki uç ARASINDA olmalı
        lin_mid = 10 ** (p_mid / 10); lin0 = 10 ** (p0 / 10); lin1 = 10 ** (p1 / 10)
        expected = 0.5 * (lin0 + lin1); expected /= expected.max()
        self.assertLess(float(np.mean(np.abs(lin_mid - expected))), 0.05)

    def test_power_scaling_invariance(self):
        # Ölçülen eğri mutlak seviyede ×10 / ÷10 ölçeklenirse (kaynak gücü/mesafe) kerteriz DEĞİŞMEMELİ
        rng = np.random.default_rng(14)
        encs, power = self._sweep_from_pattern(2400e6, 88.0, 1.0, rng)
        b_ref = self.ap.match(encs, power, 2400e6)["bearing_deg"]
        for shift_db in (-20.0, +20.0):                # dB toplamı = doğrusal ×0.01 / ×100
            b = self.ap.match(encs, power + shift_db, 2400e6)["bearing_deg"]
            self.assertLess(abs(((b - b_ref + 180) % 360) - 180), 0.5)

    def test_main_aux_matcher_parity(self):
        # Merkez (backend) ve AUX (bağımsız kopya) AYNI girdide AYNI sonucu vermeli (bakım riski kapısı)
        import importlib.util
        spec = importlib.util.spec_from_file_location("aux_station_mod",
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aux_station.py"))
        aux = importlib.util.module_from_spec(spec)
        sys.modules["aux_station_mod"] = aux
        try:
            spec.loader.exec_module(aux)
        except SystemExit:
            pass
        auxpat = aux._aux_pattern()
        if not auxpat.available():
            self.skipTest("aux pattern yüklenemedi")
        rng = np.random.default_rng(15)
        for f in (1575e6, 1900e6, 2400e6):
            encs, power = self._sweep_from_pattern(f, 123.0, 1.5, rng)
            mb = self.ap.match(encs, power, f)
            ma = auxpat.match(encs, power, f)
            self.assertIsNotNone(mb); self.assertIsNotNone(ma)
            self.assertAlmostEqual(mb["bearing_deg"], ma["bearing_deg"], places=2,
                                   msg=f"{f/1e6:.0f} MHz: merkez≠aux bearing")
            self.assertAlmostEqual(mb["sigma_deg"], ma["sigma_deg"], places=2)
            self.assertEqual(mb["quality_ok"], ma["quality_ok"])


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
