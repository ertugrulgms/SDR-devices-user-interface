"""RF bant tarama / sinyal tespiti testleri (şartname 5.1.1).

Tespit GERÇEK ölçülen FFT'den çalışır; sinyal yoksa hiçbir şey uydurmaz. Donanımsız/deterministik.
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
from backend.sdr_worker import SDRWorker, SCAN_CONFIRM_HITS
from backend.mission_logger import MissionLogger, _DATA_DIR


class TestDetectPeaks(unittest.TestCase):
    def setUp(self):
        self.d = DSPProcessor(fft_size=2048)

    def test_finds_peaks_at_correct_frequency(self):
        fft = np.full(2048, -80.0)
        fft[600] = -40.0
        peaks = self.d.detect_peaks(fft, center_mhz=2400.0, bandwidth_mhz=20.0, thresh_db=10.0)
        self.assertEqual(len(peaks), 1)
        freq, pwr, snr = peaks[0]
        self.assertAlmostEqual(freq, 2400.0 + (600 - 1024) / 2048 * 20.0, places=2)
        self.assertEqual(pwr, -40.0)
        self.assertGreater(snr, 35)

    def test_no_signal_no_detection(self):
        # Düz gürültü tabanı -> HİÇBİR tespit (sahte üretmez)
        self.assertEqual(self.d.detect_peaks(np.full(2048, -75.0), 2400.0, 20.0), [])

    def test_sorted_by_power(self):
        fft = np.full(2048, -80.0)
        fft[400] = -55.0; fft[1600] = -35.0
        peaks = self.d.detect_peaks(fft, 900.0, 10.0)
        self.assertEqual(len(peaks), 2)
        self.assertGreater(peaks[0][1], peaks[1][1])   # en güçlü önce

    def test_threshold_respected(self):
        fft = np.full(2048, -80.0)
        fft[500] = -75.0                                # yalnızca 5 dB üstü -> eşiğin (10 dB) altında
        self.assertEqual(self.d.detect_peaks(fft, 900.0, 10.0, thresh_db=10.0), [])


class TestDetectSignalsAndNoiseFloor(unittest.TestCase):
    def setUp(self):
        from backend.dsp_processor import NoiseFloorTracker
        self.d = DSPProcessor(fft_size=2048)
        self.NF = NoiseFloorTracker

    def test_wideband_island_gives_bandwidth(self):
        # Geniş-bant sinyal (düz masa) TEK tespit + doğru bant genişliği (ada genişliği)
        fft = np.full(2048, -80.0)
        fft[512:1536] = -35.0                                # bandın yarısı dolu (10 MHz @20MHz)
        nf = self.NF().update(np.full(2048, -80.0))          # önceden öğrenilmiş taban
        # aynı taban ölçekli; doğrudan detect_signals'a boş-band tabanı ver
        det = self.d.detect_signals(fft, np.full(2048, -80.0), 2400.0, 20.0, thresh_db=10.0)
        self.assertEqual(len(det), 1)                        # 50 sahte tepe DEĞİL, tek ada
        self.assertAlmostEqual(det[0][3], 10.0, delta=1.0)   # bant genişliği ~10 MHz

    def test_noise_floor_historical_min_remembers(self):
        tr = self.NF()
        tr.update(np.full(2048, -80.0))                      # boş band öğrenildi
        nf = tr.update(np.full(2048, -35.0))                 # dev sinyal geldi
        self.assertLess(float(np.median(nf)), -60.0)         # taban -80'e yakın kaldı (maskeleme yok)

    def test_narrowband_first_frame_detected(self):
        # Dar-bant sinyal (band çoğu boş) İLK karede yakalanır (persantil taban)
        FS, N = 2.4e6, 2048
        t = np.arange(N) / FS
        iq = (0.3 * np.exp(1j * 2 * np.pi * 250e3 * t)
              + 0.003 * (np.random.randn(N) + 1j * np.random.randn(N))).astype(np.complex64)
        fftd, _ = self.d.compute_fft_dbm(iq)
        nf = self.NF().update(fftd)
        self.assertGreater(len(self.d.detect_signals(fftd, nf, 900.0, 2.4)), 0)


class TestScannerStateMachine(unittest.TestCase):
    def setUp(self):
        self.w = SDRWorker()
        self.w.use_hardware = True

        class _Eng:
            def set_frequency(self, hz): self.f = hz
        self.w.engine = _Eng()

    def test_start_scan_sets_state(self):
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 2500.0)
        self.assertTrue(self.w.scan_active)
        self.assertAlmostEqual(self.w.scan_start_mhz, 400.0)
        self.assertAlmostEqual(self.w.scan_stop_mhz, 2500.0)
        self.assertEqual(self.w.center_freq_mhz, 400.0)      # başlangıca tune etti

    def test_scan_detects_and_advances(self):
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 400.0)                   # tek frekans -> revisit aynı merkez
        # Gerçekçi bir sinyal adası (çok bin geniş) simüle et
        fft = np.full(2048, -80.0); fft[1000:1030] = -30.0   # ~30 bin genişliğinde ada
        # ONAY: aynı frekansta SCAN_CONFIRM_HITS tur görülünce listeye girer (tek-kare gürültü elenir)
        for _ in range(SCAN_CONFIRM_HITS):
            self.w.scan_cursor_mhz = 400.0
            self.w._scan_settle_until = 0.0                  # oturmayı atla
            self.w._last_raw_fft = fft
            self.w._last_raw_fft_center = self.w.center_freq_mhz
            self.w._service_scan()
        self.assertGreater(len(self.w.scan_detections), 0)   # onaylı tespit kaydedildi
        # Tespit bant genişliği taşıyor (dar+geniş bant motoru)
        det = list(self.w.scan_detections.values())[0]
        self.assertIn("bw_mhz", det)

    def test_scan_advances_cursor(self):
        # Tarama, her adımda bir sonraki merkez frekansa geçmeli (süpürme).
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 2500.0)
        self.w._scan_settle_until = 0.0
        self.w._last_raw_fft = np.full(2048, -80.0)          # sinyal gerekmez, sadece ilerleme
        self.w._last_raw_fft_center = self.w.center_freq_mhz
        start_freq = self.w.center_freq_mhz
        self.w._service_scan()
        self.assertGreater(self.w.center_freq_mhz, start_freq)  # sonraki frekansa geçti

    def test_single_frame_real_signal_detected_immediately(self):
        # PTT/anlık sinyal: CONFIRM_HITS=1 -> gerçek bir sinyal (belirgin ada) TEK karede listeye girer
        # (bas-konuş telsizi susmadan yakalanır). Hayalet-selini CFAR+eşik eler, temporal onaya gerek yok.
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 2500.0)
        self.w._scan_settle_until = 0.0
        fft = np.full(2048, -80.0); fft[1000:1030] = -30.0   # gerçek sinyal adası (çevresi düşük)
        self.w._last_raw_fft = fft
        self.w._last_raw_fft_center = self.w.center_freq_mhz
        self.w._service_scan()                               # tek tur
        self.assertGreater(len(self.w.scan_detections), 0)   # anında tespit (PTT kaçmaz)

    def test_flat_raised_floor_rejected_single_frame(self):
        # HAYALET ELEME (CONFIRM=1 olsa da): güçlü taşıyıcının yükselttiği DÜZ dalgalı taban -> CFAR
        # yerel-belirginlik ile elenir; onay mekanizması olmadan da flood OLUŞMAZ.
        self.w.set_bandwidth(2.4)
        self.w.start_scan_rf(440.0, 450.0)
        self.w._scan_settle_until = 0.0
        rng = np.random.default_rng(7)
        self.w._last_raw_fft = -40.0 + rng.standard_normal(2048) * 4.0   # düz yükselmiş taban (sinyal yok)
        self.w._last_raw_fft_center = self.w.center_freq_mhz
        self.w._service_scan()
        self.assertLessEqual(len(self.w.scan_detections), 2)   # sel yok (CFAR eledi); en çok birkaç kalıntı

    def test_scan_no_self_masking_wideband(self):
        # SELF-MASKING (uzman #1): band önce boş, sonra TÜM pencereyi kaplayan dev sinyal gelir;
        # tarihsel-min taban sayesinde yine de tespit edilmeli (medyan tabanı bunu maskelerdi).
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 400.0)                   # tek frekans (revisit aynı merkez)
        self.w._scan_settle_until = 0.0
        self.w._last_raw_fft = np.full(2048, -80.0)          # 1) boş band -> düşük taban öğrenilir
        self.w._last_raw_fft_center = self.w.center_freq_mhz
        self.w._service_scan()
        # 2) dev geniş-bant sinyal (band %100 dolu); onay için CONFIRM tur besle
        for _ in range(SCAN_CONFIRM_HITS):
            self.w.scan_cursor_mhz = 400.0                   # aynı merkeze dön
            self.w._scan_settle_until = 0.0
            self.w._last_raw_fft = np.full(2048, -35.0)
            self.w._last_raw_fft_center = self.w.center_freq_mhz
            self.w._service_scan()
        self.assertGreater(len(self.w.scan_detections), 0)   # medyan körlenirdi; tarihsel-min yakaladı

    def test_scan_no_signal_no_detection(self):
        self.w.set_bandwidth(20.0)
        self.w.start_scan_rf(400.0, 2500.0)
        self.w._scan_settle_until = 0.0
        self.w._last_raw_fft = np.full(2048, -78.0)          # düz gürültü
        self.w._last_raw_fft_center = self.w.center_freq_mhz
        self.w._service_scan()
        self.assertEqual(len(self.w.scan_detections), 0)     # sahte tespit YOK

    def test_stop_scan(self):
        self.w.start_scan_rf(400.0, 2500.0)
        self.w.stop_scan_rf()
        self.assertFalse(self.w.scan_active)


class TestCFARProminence(unittest.TestCase):
    """CFAR yerel-belirginlik: güçlü taşıyıcının yükselttiği DÜZ gürültü tabanı yalancı tespit
    üretmemeli (gerçek dış sinyal yokken), gerçek TEPE korunmalı."""

    def setUp(self):
        self.d = DSPProcessor(fft_size=2048)

    def test_flat_raised_floor_rejected(self):
        # Güçlü taşıyıcının desense ettiği DÜZ dalgalı taban (-40 ort). Tarihsel-min ~-53'te takılı.
        # Gerçek sinyal YOK -> CFAR ile tespit sayısı çok düşük olmalı (eski davranış yüzlerce üretirdi).
        rng = np.random.default_rng(7)
        carpet = -40.0 + rng.standard_normal(2048) * 4.0
        nf = np.full(2048, -53.0)
        old = self.d.detect_signals(carpet, nf, 1260.0, 10.0, 10.0, prominence_db=0.0)
        new = self.d.detect_signals(carpet, nf, 1260.0, 10.0, 10.0, prominence_db=12.0)
        self.assertGreater(len(old), 50)      # CFAR'sız: düz taban yüzlerce hayalet
        self.assertLessEqual(len(new), 3)      # CFAR'lı: düz taban elenir (hayalet ~0)

    def test_real_peak_survives_cfar(self):
        # Çevresi düşük gürültü (-70), tek gerçek TEPE (+20) -> CFAR onu KORUMALI.
        rng = np.random.default_rng(3)
        fft = np.full(2048, -70.0) + rng.standard_normal(2048) * 1.0
        fft[600:604] = 20.0
        nf = np.full(2048, -73.0)
        r = self.d.detect_signals(fft, nf, 1260.0, 10.0, 10.0, prominence_db=12.0)
        self.assertGreaterEqual(len(r), 1)     # gerçek tepe korunur
        self.assertTrue(any(abs(f - (1260.0 + (602 - 1024) / 2048 * 10.0)) < 0.05 for f, *_ in r))

    def test_wide_signal_bypasses_cfar(self):
        # Pencereyi dolduran GENİŞ sinyal (self-masking): yerel taban güvenilmez -> CFAR atlanmalı,
        # tarihsel-min sayesinde yine tespit edilmeli.
        fft = np.full(2048, -35.0)             # tüm pencere dolu (geniş sinyal)
        nf = np.full(2048, -80.0)              # tarihsel-min düşük taban hatırlıyor
        r = self.d.detect_signals(fft, nf, 1260.0, 10.0, 10.0, prominence_db=12.0)
        self.assertGreaterEqual(len(r), 1)     # geniş sinyal CFAR'a rağmen yakalanır


class TestDetectionLogging(unittest.TestCase):
    def test_detection_persists(self):
        db = f"t_det_{os.getpid()}.db"
        lg = MissionLogger(db_name=db)
        try:
            lg.log_detection(2437.5, -42.0, 25.0)
            cur = lg.conn.cursor()
            cur.execute("SELECT frequency_mhz, power_dbfs, snr_db FROM detection_logs")
            rows = cur.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 2437.5)
        finally:
            lg.close()
            p = os.path.join(_DATA_DIR, db)
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    unittest.main()
