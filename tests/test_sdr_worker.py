"""SDRWorker birim testleri — DONANIMSIZ (start() çağrılmaz, gerçek USRP gerekmez).

Kapsam: TX baseband üretimi (her karıştırma modu) ve durum yönetimi (frekans/gain/bant/
DF modu/TX tetikleme). run() döngüsü ve gerçek RX donanım gerektirdiği için burada test
edilmez; saf mantık ve durum geçişleri izole edilir.
"""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QThread/pyqtSignal için bir Qt uygulaması gerekir. QApplication kullanıyoruz (QCoreApplication
# değil) ki aynı test oturumundaki UI testleriyle (QWidget) çakışmasın; offscreen ile başsız.
from PyQt6.QtWidgets import QApplication
_app = QApplication.instance() or QApplication(sys.argv)

from backend.sdr_worker import SDRWorker, LOOK_THROUGH_ENDED_WINDOWS


class TestTxBuffer(unittest.TestCase):
    """_build_tx_buffer: her mod için geçerli baseband I/Q tamponu üretmeli."""

    def setUp(self):
        self.w = SDRWorker()

    def _build(self, mode, **params):
        self.w.tx_params.update({"mode": mode, "jsr_db": 15.0})
        self.w.tx_params.update(params)
        return self.w._build_tx_buffer()

    def test_all_modes_produce_valid_buffer(self):
        from backend.jamming_generator import TX_DIGITAL_PEAK
        modes = ["BARRAGE", "SINGLE_TONE", "MULTI_TONE", "SWEEP", "ANALOG_SPOOF",
                 "GNSS_SPOOF", "LOOK_THROUGH", "WIFI_JAMMING"]
        for mode in modes:
            buf = self._build(mode, coords="39.9,32.8", wave_type="Sinüs Dalga (Tone)")
            self.assertEqual(buf.dtype, np.complex64, f"{mode}: dtype")
            self.assertGreater(len(buf), 0, f"{mode}: boş")
            self.assertTrue(buf.flags["C_CONTIGUOUS"], f"{mode}: contiguous değil")
            self.assertTrue(np.all(np.isfinite(buf)), f"{mode}: NaN/Inf")
            self.assertGreater(np.max(np.abs(buf)), 0.0, f"{mode}: tümü sıfır")
            # KRİTİK #1: donanıma giden tampon DAC tavanını aşamaz
            self.assertLessEqual(float(np.max(np.abs(buf))), TX_DIGITAL_PEAK + 1e-6, f"{mode}: DAC clip")

    def test_spot_is_bandlimited_noise_not_cw_tone(self):
        # 5.2.1 düzeltmesi: "tekli" (SPOT) artık CW ton DEĞİL, hedef kanalına yoğunlaşmış GÜRÜLTÜ
        # (gürültü tabanını yükseltir -> Shannon-etkin). Zarf DEĞİŞKEN (gürültü), spektrum yayılmış.
        buf = self._build("SPOT")
        env = np.abs(buf)
        self.assertGreater(np.std(env) / (np.mean(env) + 1e-9), 0.1)   # değişken zarf (gürültü)
        p = np.abs(np.fft.fft(buf)) ** 2
        self.assertLess(float(np.max(p) / np.sum(p)), 0.2)             # tek çizgi değil (yayılmış)

    def test_look_through_buffer_is_continuous(self):
        # YENİ TASARIM (bulgu #3): look-through tamponu artık SÜREKLİ baraj gürültüsüdür;
        # aç/kapa (T/R) donanımda _service_look_through ile yapılır, tamponda sıfır kuyruk YOK.
        buf = self._build("LOOK_THROUGH", duty_percent=50.0)
        half = len(buf) // 2
        self.assertGreater(np.max(np.abs(buf[:half])), 0.0)        # ilk yarı dolu
        self.assertGreater(np.max(np.abs(buf[half:])), 0.0)        # ikinci yarı da dolu (sürekli)


class TestStateManagement(unittest.TestCase):
    """Donanımsız durum geçişleri: setter'lar iç durumu doğru güncellemeli."""

    def setUp(self):
        self.w = SDRWorker()

    def test_set_frequency(self):
        self.w.set_frequency(100.5)
        self.assertEqual(self.w.center_freq_mhz, 100.5)

    def test_set_gain(self):
        self.w.set_gain(55)
        self.assertEqual(self.w.gain_db, 55.0)

    def test_set_bandwidth_syncs_sample_rate(self):
        self.w.set_bandwidth(20.0)
        self.assertEqual(self.w.sample_rate, 20e6)
        self.assertEqual(self.w.bandwidth_mhz, 20.0)   # ikisi senkron olmalı

    def test_set_bandwidth_rebuilds_engines(self):
        old_jam = self.w.jam_gen
        old_clf = self.w.classifier
        self.w.set_bandwidth(10.0)
        self.assertIsNot(self.w.jam_gen, old_jam)        # yeni sample_rate ile yeniden kurulmalı
        self.assertIsNot(self.w.classifier, old_clf)

    def test_trigger_and_stop_tx(self):
        self.w.trigger_tx({"mode": "BARRAGE", "jsr_db": 20.0})
        self.assertTrue(self.w.tx_active)
        self.assertEqual(self.w.tx_params["mode"], "BARRAGE")
        self.w.stop_tx()
        self.assertFalse(self.w.tx_active)
        self.assertEqual(self.w.tx_params["mode"], "NONE")

    def test_df_mode_and_manual_angles(self):
        self.w.set_df_mode(auto=False)
        self.assertFalse(self.w.is_auto_df)
        self.w.set_manual_angles((10.0, 20.0, 30.0))
        self.assertEqual(self.w.manual_angles, (10.0, 20.0, 30.0))

    def test_antenna_port(self):
        self.w.set_antenna("RX2")
        self.assertEqual(self.w.antenna_port, "RX2")

    def test_gnss_spoof_forces_l1_frequency(self):
        # GNSS aldatma seçilince taşıyıcı otomatik GPS L1'e (1575.42 MHz) gitmeli
        self.w.set_frequency(2400.0)
        self.w.trigger_tx({"mode": "GNSS_SPOOF", "gnss_code": "GPS L1 C/A", "coords": "39.9,32.8"})
        self.assertAlmostEqual(self.w.center_freq_mhz, 1575.42, places=2)
        self.w.stop_tx()

    def test_multi_tone_mode_state(self):
        self.w.trigger_tx({"mode": "MULTI_TONE", "jsr_db": 15.0})
        self.assertTrue(self.w.tx_active)
        self.assertEqual(self.w.tx_params["mode"], "MULTI_TONE")
        # üretilen tampon çok-ton olmalı (tek frekanstan geniş)
        self.assertGreater(np.max(np.abs(self.w.tx_pre_gen)), 0.0)
        self.w.stop_tx()

    def test_tx_gain_is_set_and_clamped_from_payload(self):
        # KRİTİK #2: TX RF kazancı artık payload'dan gelir ve B200mini üst sınırına kırpılır
        from backend.sdr_worker import TX_GAIN_MAX_DB
        self.w.trigger_tx({"mode": "BARRAGE", "jsr_db": 15.0, "tx_gain_db": 55.0})
        self.assertEqual(self.w.tx_gain_db, 55.0)
        self.w.stop_tx()
        self.w.trigger_tx({"mode": "BARRAGE", "jsr_db": 15.0, "tx_gain_db": 200.0})  # aşırı
        self.assertEqual(self.w.tx_gain_db, TX_GAIN_MAX_DB)                          # tavana kırpıldı
        self.w.stop_tx()

    def test_wifi_jamming_retunes_to_band(self):
        # ORTA #9: Wi-Fi engelleme gerçek baraj -> 2.4 GHz seçilince kanal 6 (2437 MHz)'e geçer
        self.w.set_frequency(100.0)
        self.w.trigger_tx({"mode": "WIFI_JAMMING", "wifi_band": "2.4 GHz (802.11 b/g/n)"})
        self.assertAlmostEqual(self.w.center_freq_mhz, 2437.0, places=1)
        self.w.stop_tx()

    def test_classify_snapshot_falls_back_without_engine(self):
        # Donanım/engine yokken get_classify_snapshot canlı iq'ya düşmeli (çökmemeli)
        fallback = np.ones(2048, dtype=np.complex64)
        out = self.w._get_classify_snapshot(fallback)
        self.assertIs(out, fallback)

    def test_hop_tracker_reset_on_retune(self):
        # Frekans değişince FHSS zaman-geçmişi sıfırlanmalı (eski band anlamsız)
        self.w.hop_tracker.update(0.0, 0.5, 25.0)
        self.assertGreater(len(self.w.hop_tracker._hist), 0)
        self.w.set_frequency(1800.0)
        self.assertEqual(len(self.w.hop_tracker._hist), 0)

    def test_look_through_tr_state_machine(self):
        # KRİTİK #3: T/R durum makinesi yayın penceresinde başlamalı; dinleme süresi dolunca
        # (donanım yokken engine çağrısı atlanır ama _lt_tx_on bayrağı yine de dönmeli mantığı
        # donanıma bağlı olduğundan, burada yalnızca tetiklemede yayın fazından başladığını doğrularız)
        self.w.trigger_tx({"mode": "LOOK_THROUGH", "duty_percent": 50.0, "look_time_ms": 40.0})
        self.assertTrue(self.w._lt_tx_on)          # tetikleme -> yayın penceresinden başlar
        self.w.stop_tx()
        self.assertFalse(self.w._lt_tx_on)         # durdurunca dinleme/kapalı


class _FakeTxEngine:
    """Look-through kapalı-çevrim testi için sahte engine (T/R + buffer çağrılarını sayar)."""
    def __init__(self):
        self.tx_active = True
        self.tx_set_calls = []
        self.buffer_sets = 0
    def set_tx_active(self, on): self.tx_active = bool(on); self.tx_set_calls.append(bool(on))
    def set_tx_buffer(self, buf): self.buffer_sets += 1
    def set_tx_gain(self, g): pass
    def set_frequency(self, f): pass
    def set_gain(self, g): pass
    def set_antenna(self, a): pass


class TestLookThroughClosedLoop(unittest.TestCase):
    """5.2.2: arabakış — SÜREKLİ bastırma. Operatör karıştırmayı seçtiğinde, kısa dinleme (peek)
    penceresi karıştırmayı DURDURMAZ; yalnızca ekran + gücü tespit edilen banda odaklama (refocus)
    için bilgi toplar. Her peek sonrası jam'a geri dönülür. Donanımsız (sahte engine).
    (Not: eski 'hedef yoksa STANDBY' davranışı kaldırıldı — kısa/güvenilmez peek ölçümü ya da
    aralıklı hedef yayını karıştırmayı erkenden kesiyordu.)"""

    def setUp(self):
        self.w = SDRWorker()
        self.w.use_hardware = True
        self.w.engine = _FakeTxEngine()
        # KLASİK arabakış (oto-tarama KAPALI): tek frekans, sürekli bastırma. Oto-tarama açıkken
        # hedef susunca band taranır (ayrı testler); bu sınıf klasik sürekli davranışı doğrular.
        self.w.trigger_tx({"mode": "LOOK_THROUGH", "jam_sec": 5.0, "listen_sec": 2.0,
                           "lt_auto_scan": False})

    def _advance(self, present, bw_hz=0.0, off_hz=0.0):
        """Bir dinleme penceresini simüle et: durumu LISTEN yap, süresini geçir, servisi çağır."""
        self.w._lt_present = present
        self.w._lt_bw_hz = bw_hz
        self.w._lt_peak_offset_hz = off_hz
        self.w._lt_state = "LISTEN"              # dinleme penceresindeyiz
        self.w._lt_tx_on = False
        self.w._lt_phase_start = 0.0             # listen süresi dolmuş say
        self.w._service_look_through()

    def test_absent_target_keeps_jamming(self):
        # Ardışık BOŞ pencerelerde bile karıştırma SÜRER (standby yok) -> her peek sonrası TX açık
        for _ in range(LOOK_THROUGH_ENDED_WINDOWS + 2):
            self._advance(present=False)
        self.assertTrue(self.w._lt_active)         # karıştırma AKTİF kalır
        self.assertTrue(self.w.engine.tx_active)   # her peek sonrası TX'e geri dönülür
        self.assertTrue(self.w._lt_tx_on)          # jam penceresinde

    def test_jamming_stays_continuous(self):
        # Hedef var/yok karışık gelse de karıştırma her zaman sürdürülür (kesinti yok)
        for present in (True, False, False, True, False):
            self._advance(present=present, bw_hz=12.5e3 if present else 0.0)
            self.assertTrue(self.w._lt_active)
            self.assertTrue(self.w.engine.tx_active)

    def test_power_refocus_on_target_move(self):
        # Hedef offseti belirgin kayınca TX tamponu yeniden odaklı üretilmeli (set_tx_buffer çağrısı)
        before = self.w.engine.buffer_sets
        self._advance(present=True, bw_hz=15e3, off_hz=0.0)
        self._advance(present=True, bw_hz=15e3, off_hz=400e3)   # 400 kHz kaydı -> refocus
        self.assertGreater(self.w.engine.buffer_sets, before)
        self.assertAlmostEqual(self.w.tx_params["focus_offset_hz"], 400e3)


class TestRunLoopPayload(unittest.TestCase):
    """run() döngüsü SİMÜLASYON modunda (donanımsız) geçerli bir payload üretmeli.
    _init_hardware False'a zorlanır -> gerçek USRP gerekmez, simülasyon yolu çalışır."""

    def test_payload_has_all_expected_keys(self):
        from PyQt6.QtCore import QTimer
        w = SDRWorker()
        w._init_hardware = lambda: False          # simülasyon modunu zorla (donanım yok)
        payloads = []
        w.data_ready.connect(lambda p: payloads.append(p))
        w.start()
        QTimer.singleShot(1500, lambda: (w.stop(), _app.quit()))
        _app.exec()

        self.assertGreater(len(payloads), 0, "run() döngüsü payload üretmedi")
        p = payloads[-1]
        for key in ("x_freqs", "fft_dbm", "audio_y", "angles", "amps", "target_pos",
                    "freq_bounds", "bandwidth_mhz", "gain_db", "occupied_bw_hz",
                    "signal_class", "snr_db", "modulation", "multiplex", "ekkt", "protocol"):
            self.assertIn(key, p, f"payload eksik anahtar: {key}")
        # Spektrum dizileri doğru boyutta
        self.assertEqual(len(p["fft_dbm"]), 2048)
        self.assertEqual(len(p["angles"]), 3)


if __name__ == "__main__":
    unittest.main()
