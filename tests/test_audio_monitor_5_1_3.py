"""spec 5.1.3 SİNYAL İZLEME/DİNLEME testleri: gerçek ses demodülasyonu (FM/AM/USB/LSB), gapless
akış sürekliliği, sinyal izleme/takip (süreklilik + parametre geçmişi), sayısal ses (4FSK/C4FM
tespiti) ve harici DSD-FME köprüsü. Donanımsız/deterministik."""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.audio_demodulator import StreamingDemodulator
from backend.signal_monitor import SignalMonitor
from backend.digital_voice import (FourFSKDetector, DSDFMEDecoder, find_dsd_binary,
                                   classify_fm_or_fsk)
from backend.audio_player import AudioPlayer


FS = 2.4e6


def _fm_iq(msg, dev, fs=FS):
    msg = msg / (np.max(np.abs(msg)) + 1e-9)
    return np.exp(1j * 2 * np.pi * dev * np.cumsum(msg) / fs).astype(np.complex64)


def _demod_digital(iq, fs=FS):
    d = StreamingDemodulator(fs, mode="FM", digital=True, channel_bw=12500.0)
    out = []
    for i in range(0, len(iq), 1 << 16):
        out.append(d.process(iq[i:i + (1 << 16)]))
    return np.concatenate(out), d.out_rate


class TestAnalogDigitalDiscriminator(unittest.TestCase):
    """5.1.2: FM/FSK 'Ortak' -> sese demodüle edip analog/sayısal KESİN ayrım (classify_fm_or_fsk)."""
    def setUp(self):
        self.n = int(FS * 0.2)
        self.t = np.arange(self.n) / FS

    def _verdict(self, iq):
        rng = np.random.default_rng(1)
        iq = iq + (0.02 * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))).astype(np.complex64)
        disc, rate = _demod_digital(iq)
        return classify_fm_or_fsk(disc, rate)

    def test_voice_fm_is_analog(self):
        rng = np.random.default_rng(0)
        msg = (np.sin(2 * np.pi * 300 * self.t) + 0.6 * np.sin(2 * np.pi * 900 * self.t)
               + 0.4 * np.sin(2 * np.pi * 1700 * self.t) + 0.2 * rng.standard_normal(self.n))
        self.assertIs(self._verdict(_fm_iq(msg, 3000))["is_digital"], False)

    def test_tone_fm_is_analog(self):
        self.assertIs(self._verdict(_fm_iq(np.sin(2 * np.pi * 1000 * self.t), 3000))["is_digital"], False)

    def test_c4fm_is_digital(self):
        rng = np.random.default_rng(2)
        sps = int(FS / 4800)
        lv = np.array([-3, -1, 1, 3.0]) / 3.0
        syms = np.repeat(lv[rng.integers(0, 4, self.n // sps + 1)], sps)[:self.n]
        self.assertIs(self._verdict(_fm_iq(syms, 1944))["is_digital"], True)

    def test_2fsk_is_digital(self):
        rng = np.random.default_rng(3)
        sps = int(FS / 4800)
        lv = np.array([-1.0, 1.0])
        syms = np.repeat(lv[rng.integers(0, 2, self.n // sps + 1)], sps)[:self.n]
        self.assertIs(self._verdict(_fm_iq(syms, 2500))["is_digital"], True)

    def test_short_input_is_uncertain(self):
        self.assertIsNone(classify_fm_or_fsk(np.zeros(100), 48000)["is_digital"])


class TestWorkerAnalogDigitalRefine(unittest.TestCase):
    def test_worker_refines_fm_fsk_to_analog_and_digital(self):
        from backend.sdr_worker import SDRWorker
        w = SDRWorker(); w.sample_rate = FS; w._last_peak_offset_hz = 0.0
        n = int(FS * 0.2); t = np.arange(n) / FS; rng = np.random.default_rng(0)
        voice = _fm_iq(np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 1600 * t)
                       + 0.2 * rng.standard_normal(n), 3000)
        sps = int(FS / 4800); lv = np.array([-3, -1, 1, 3.0]) / 3.0
        c4fm = _fm_iq(np.repeat(lv[rng.integers(0, 4, n // sps + 1)], sps)[:n], 1944)
        w._clf_result = {"modulation": "FM/FSK (Frekans Mod.)"}
        w._refine_fm_fsk(voice)
        self.assertIn("Analog", w._clf_result.get("analog_digital", ""))
        w._clf_result = {"modulation": "FM/FSK (Frekans Mod.)"}
        w._refine_fm_fsk(c4fm)
        self.assertIn("Sayısal", w._clf_result.get("analog_digital", ""))


def _dominant_freq(audio, rate):
    if len(audio) < 32:
        return -1.0
    w = np.hanning(len(audio))
    sp = np.abs(np.fft.rfft(audio * w))
    fr = np.fft.rfftfreq(len(audio), 1.0 / rate)
    return float(fr[np.argmax(sp[1:]) + 1])


def _run_chunked(demod, iq, chunk):
    demod.reset()
    out = []
    for i in range(0, len(iq), chunk):
        out.append(demod.process(iq[i:i + chunk]))
    return np.concatenate(out) if out else np.zeros(0)


class TestStreamingDemodulator(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(int(FS * 0.2)) / FS
        self.tone = 1000.0

    def test_out_rate_integer_decimation(self):
        d = StreamingDemodulator(FS, audio_rate=48000, mode="FM")
        self.assertEqual(d.out_rate, 48000.0)   # 2.4e6/50 = 48k tam

    def test_fm_recovers_tone(self):
        ph = 2 * np.pi * 3000 * np.cumsum(np.sin(2 * np.pi * self.tone * self.t)) / FS
        fm = np.exp(1j * ph).astype(np.complex64)
        d = StreamingDemodulator(FS, mode="FM")
        a = _run_chunked(d, fm, 65536)
        self.assertAlmostEqual(_dominant_freq(a, d.out_rate), self.tone, delta=60)

    def test_am_recovers_tone(self):
        am = ((1.0 + 0.7 * np.sin(2 * np.pi * self.tone * self.t))).astype(np.complex64)
        d = StreamingDemodulator(FS, mode="AM")
        a = _run_chunked(d, am, 65536)
        self.assertAlmostEqual(_dominant_freq(a, d.out_rate), self.tone, delta=60)

    def test_usb_lsb_recover_tone(self):
        usb = np.exp(1j * 2 * np.pi * self.tone * self.t).astype(np.complex64)
        lsb = np.exp(-1j * 2 * np.pi * self.tone * self.t).astype(np.complex64)
        du = StreamingDemodulator(FS, mode="USB")
        dl = StreamingDemodulator(FS, mode="LSB")
        self.assertAlmostEqual(_dominant_freq(_run_chunked(du, usb, 65536), du.out_rate), self.tone, delta=60)
        self.assertAlmostEqual(_dominant_freq(_run_chunked(dl, lsb, 65536), dl.out_rate), self.tone, delta=60)

    def test_sideband_rejection(self):
        # USB sinyali LSB modunda büyük ölçüde bastırılmalı
        usb = np.exp(1j * 2 * np.pi * self.tone * self.t).astype(np.complex64)
        p_usb = np.mean(_run_chunked(StreamingDemodulator(FS, mode="USB"), usb, 65536) ** 2)
        p_lsb = np.mean(_run_chunked(StreamingDemodulator(FS, mode="LSB"), usb, 65536) ** 2)
        self.assertLess(p_lsb, 0.25 * p_usb)

    def test_gapless_continuity(self):
        # Filtre/ayrımlayıcı blok sınırlarında sürekli olmalı (AGC hariç birebir aynı)
        ph = 2 * np.pi * 3000 * np.cumsum(np.sin(2 * np.pi * self.tone * self.t)) / FS
        fm = np.exp(1j * ph).astype(np.complex64)
        d = StreamingDemodulator(FS, mode="FM")
        d._agc = lambda a: a.astype(np.float32)   # AGC'yi kimlik yap
        one = _run_chunked(d, fm, len(fm))
        many = _run_chunked(d, fm, 20000)
        m = min(len(one), len(many))
        self.assertLess(np.max(np.abs(one[200:m] - many[200:m])), 1e-6)


class TestSignalMonitor(unittest.TestCase):
    def test_lock_and_continuity_events(self):
        m = SignalMonitor(drop_grace_sec=1.5)
        m.lock(433.9)
        t = 0.0
        for _ in range(10):   # 5 sn var
            m.update(t, True, carrier_mhz=433.9, power_dbm=-50, bw_hz=12500, snr_db=20)
            t += 0.5
        events = []
        for _ in range(6):    # 3 sn yok -> drop
            e = m.update(t, False, snr_db=2); t += 0.5
            if e:
                events.append(e)
        for _ in range(6):    # tekrar var -> reacquire
            e = m.update(t, True, carrier_mhz=433.9, power_dbm=-52, bw_hz=12500, snr_db=18); t += 0.5
            if e:
                events.append(e)
        self.assertIn("drop", events)
        self.assertIn("reacquire", events)
        s = m.status(now=t)
        self.assertTrue(s["locked"])
        self.assertGreaterEqual(s["drops"], 1)
        self.assertGreaterEqual(s["reacquires"], 1)
        self.assertGreater(s["uptime_pct"], 40.0)

    def test_unlocked_status(self):
        m = SignalMonitor()
        self.assertFalse(m.status()["locked"])

    def test_freq_drift_measured(self):
        m = SignalMonitor()
        m.lock(100.0)
        t = 0.0
        for f in (100.000, 100.001, 100.002, 100.003):
            m.update(t, True, carrier_mhz=f, power_dbm=-40, bw_hz=1000, snr_db=15); t += 0.5
        self.assertAlmostEqual(m.status(now=t)["freq_drift_khz"], 3.0, delta=0.1)


class TestFourFSKDetector(unittest.TestCase):
    def setUp(self):
        self.det = FourFSKDetector(48000.0)

    @staticmethod
    def _c4fm(baud, noise=0.02, seed=0, n=2000):
        r = np.random.default_rng(seed)
        sps = int(48000.0 / baud)
        lv = np.array([-3, -1, 1, 3.0])
        d = np.repeat(lv[r.integers(0, 4, n)], sps).astype(float)
        return d / np.std(d) + noise * r.standard_normal(len(d))

    def test_c4fm_detected_with_symbol_rate(self):
        r = self.det.detect(self._c4fm(4800))
        self.assertTrue(r["is_4fsk"])
        self.assertAlmostEqual(r["symbol_rate_hz"], 4800.0, delta=200)

    def test_c4fm_noisy_detected(self):
        self.assertTrue(self.det.detect(self._c4fm(4800, noise=0.15, seed=3))["is_4fsk"])

    def test_analog_fm_not_4fsk(self):
        t = np.arange(20000) / 48000.0
        two = np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 2500 * t)
        one = np.sin(2 * np.pi * 1200 * t)
        self.assertFalse(self.det.detect(two)["is_4fsk"])
        self.assertFalse(self.det.detect(one)["is_4fsk"])   # tonluk kontrolü

    def test_noise_not_4fsk(self):
        noise = np.random.default_rng(0).standard_normal(20000)
        self.assertFalse(self.det.detect(noise)["is_4fsk"])


class TestDSDFMEDecoder(unittest.TestCase):
    def test_availability_matches_binary(self):
        d = DSDFMEDecoder()
        self.assertEqual(d.available(), find_dsd_binary() is not None)

    def test_install_hint_nonempty(self):
        self.assertIn("dsd-fme", DSDFMEDecoder().install_hint())

    def test_feed_before_start_is_noop(self):
        d = DSDFMEDecoder()
        d.feed(np.zeros(100, dtype=np.float32))   # patlamamalı
        self.assertFalse(d.is_running())

    def test_start_without_binary_reports_false(self):
        d = DSDFMEDecoder()
        d.binary = None
        self.assertFalse(d.start())
        self.assertIsNotNone(d.last_error)


class _FakeEngine:
    """get_stream_new sağlayan sahte SDR motoru (donanımsız test)."""
    def __init__(self, fs):
        t = np.arange(int(fs * 0.05)) / fs
        ph = 2 * np.pi * 3000 * np.cumsum(np.sin(2 * np.pi * 1000 * t)) / fs
        self.iq = np.exp(1j * ph).astype(np.complex64)
        self.reset_calls = 0

    def reset_audio_cursor(self):
        self.reset_calls += 1

    def get_stream_new(self, n):
        return self.iq[:n]


class TestAudioPlayer(unittest.TestCase):
    def test_construction_and_out_rate(self):
        ap = AudioPlayer(_FakeEngine(FS), FS, mode="FM")
        self.assertEqual(ap.out_rate, 48000)
        self.assertTrue(ap.is_muted())
        self.assertEqual(ap.mode, "FM")

    def test_demod_pipeline_produces_audio(self):
        eng = _FakeEngine(FS)
        ap = AudioPlayer(eng, FS, mode="FM")
        audio = ap.demod.process(eng.get_stream_new(1 << 16))
        self.assertGreater(len(audio), 0)
        self.assertGreater(np.sqrt(np.mean(audio ** 2)), 0.0)

    def test_set_mode_and_digital(self):
        ap = AudioPlayer(_FakeEngine(FS), FS)
        ap.set_mode("USB")
        self.assertEqual(ap.mode, "USB")
        info = ap.set_digital(True)
        self.assertIn("enabled", info)
        self.assertIn("dsd_available", info)
        self.assertTrue(ap.digital_enabled)
        ap.set_digital(False)
        self.assertFalse(ap.digital_enabled)

    def test_start_without_engine_api_fails_gracefully(self):
        ap = AudioPlayer(object(), FS)   # get_stream_new yok
        self.assertFalse(ap.start())
        self.assertIsNotNone(ap.last_error)


if __name__ == "__main__":
    unittest.main()
