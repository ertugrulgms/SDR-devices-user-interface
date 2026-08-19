"""spec 5.2.3 ANALOG TELSİZ ALDATMA — gerçek ses mesajını hedef modülasyonuna (NBFM/AM) taşıma.

Aldatma sinyali, bir FM/AM demodülatöründen geçirilince özgün ses mesajını GERİ VERİR — yani hedef
analog telsiz onu gerçek (ama gerçek-dışı içerikli) ses olarak çalar ('yanlış duyar'). Tamamen
deterministik/donanımsız; DAC güvenliği ve modülasyon doğruluğu kilitlenir.
"""
import os
import sys
import wave
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.audio_deception import (AnalogDeceptionModulator, load_wav_mono,
                                     MAX_DECEPTION_SAMPLES, DEFAULT_NBFM_DEVIATION_HZ,
                                     detect_ctcss, nearest_ctcss, channel_deviation_budget,
                                     CTCSS_TONES)
from backend.audio_demodulator import StreamingDemodulator
from backend.tx_engine import TxWaveformBuilder, WAV_DECEPTION_WAVE
from backend.jamming_generator import JammingGenerator, TX_DIGITAL_PEAK
from backend.gnss_spoofer import GNSSSpoofer

FS = 2.4e6
ARATE = 48000


def _tone(freq, dur=0.15, rate=ARATE, amp=0.8):
    t = np.arange(int(rate * dur)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant(audio, rate):
    w = np.hanning(len(audio))
    sp = np.abs(np.fft.rfft(audio * w))
    fr = np.fft.rfftfreq(len(audio), 1.0 / rate)
    return float(fr[np.argmax(sp[1:]) + 1])


def _fm_demod(buf, mode="FM"):
    d = StreamingDemodulator(FS, mode=mode)
    out = []
    for i in range(0, len(buf), 65536):
        out.append(d.process(buf[i:i + 65536]))
    return np.concatenate(out), d.out_rate


class TestDeceptionModulator(unittest.TestCase):
    def setUp(self):
        self.mod = AnalogDeceptionModulator(FS)

    def test_nbfm_is_constant_envelope(self):
        buf = self.mod.modulate(_tone(1000), ARATE, mode="NBFM", deviation_hz=3000)
        env = np.abs(buf)
        self.assertLess(np.std(env) / (np.mean(env) + 1e-9), 0.02)   # FM -> sabit zarf

    def test_nbfm_target_plays_real_audio(self):
        # Hedef telsiz (FM demod) aldatma sinyalinden ÖZGÜN ses tonunu geri çıkarmalı
        buf = self.mod.modulate(_tone(1200), ARATE, mode="NBFM", deviation_hz=3000)
        rec, rate = _fm_demod(buf, "FM")
        self.assertAlmostEqual(_dominant(rec, rate), 1200.0, delta=60)

    def test_am_variable_envelope_and_recovers(self):
        buf = self.mod.modulate(_tone(1000), ARATE, mode="AM")
        env = np.abs(buf)
        self.assertGreater(np.std(env) / (np.mean(env) + 1e-9), 0.1)  # AM -> değişken zarf
        rec, rate = _fm_demod(buf, "AM")
        self.assertAlmostEqual(_dominant(rec, rate), 1000.0, delta=60)

    def test_message_capped_to_max_samples(self):
        long_audio = np.tile(_tone(500, dur=1.0), 20)   # uzun mesaj
        buf = self.mod.modulate(long_audio, ARATE, mode="NBFM", max_samples=100000)
        self.assertLessEqual(len(buf), 100000)

    def test_empty_audio_returns_empty(self):
        self.assertEqual(len(self.mod.modulate(np.zeros(1), ARATE)), 0)


class TestCtcss(unittest.TestCase):
    """CTCSS ton-squelch: hedef, doğru alt-ses tonu olmadan hoparlörü AÇMAZ (5.2.3 kritik)."""

    def _raw_disc(self, buf):
        d = StreamingDemodulator(FS, mode="FM", digital=True, channel_bw=16000)  # ham ayrımlayıcı
        out = []
        for i in range(0, len(buf), 65536):
            out.append(d.process(buf[i:i + 65536]))
        return np.concatenate(out), d.out_rate

    def test_ctcss_tone_present_and_detected(self):
        mod = AnalogDeceptionModulator(FS)
        dev, cdev = channel_deviation_budget(12.5)
        buf = mod.modulate(_tone(1000, dur=0.4), ARATE, mode="NBFM",
                           deviation_hz=dev, ctcss_hz=88.5, ctcss_dev_hz=cdev)
        disc, rate = self._raw_disc(buf)
        res = detect_ctcss(disc, rate)
        self.assertTrue(res["present"])
        self.assertEqual(res["ctcss_std_hz"], 88.5)      # standart tona eşlendi

    def test_no_ctcss_not_detected(self):
        mod = AnalogDeceptionModulator(FS)
        buf = mod.modulate(_tone(1000, dur=0.4), ARATE, mode="NBFM", deviation_hz=2000, ctcss_hz=0.0)
        disc, rate = self._raw_disc(buf)
        self.assertFalse(detect_ctcss(disc, rate)["present"])

    def test_total_deviation_within_channel_budget(self):
        # Ses + CTCSS tepe sapması 12.5 kHz kanal maksimumunu (±2.5 kHz) aşmamalı (splatter)
        mod = AnalogDeceptionModulator(FS)
        dev, cdev = channel_deviation_budget(12.5)
        buf = mod.modulate(_tone(1000, dur=0.3), ARATE, mode="NBFM",
                           deviation_hz=dev, ctcss_hz=88.5, ctcss_dev_hz=cdev, preemphasis=False)
        z = buf.astype(np.complex128)
        inst = np.angle(z[1:] * np.conj(z[:-1])) * FS / (2 * np.pi)
        self.assertLess(np.percentile(np.abs(inst), 99.5), 2600.0)

    def test_nearest_ctcss_mapping(self):
        self.assertEqual(nearest_ctcss(88.4), 88.5)
        self.assertEqual(nearest_ctcss(100.1), 100.0)
        self.assertIsNone(nearest_ctcss(500.0))          # standart dışı -> None

    def test_channel_budget_values(self):
        self.assertEqual(channel_deviation_budget(12.5), (2000.0, 500.0))
        self.assertEqual(channel_deviation_budget(25.0), (4000.0, 750.0))

    def test_ctcss_still_recovers_voice(self):
        # CTCSS eklenince özgün ses (1 kHz) yine demodüle edilebilmeli (aldatma anlaşılır kalır)
        mod = AnalogDeceptionModulator(FS)
        buf = mod.modulate(_tone(1000, dur=0.4), ARATE, mode="NBFM",
                           deviation_hz=2000, ctcss_hz=88.5, ctcss_dev_hz=500)
        rec, rate = _fm_demod(buf, "FM")   # de-emphasis'li normal FM demod
        self.assertAlmostEqual(_dominant(rec, rate), 1000.0, delta=80)


class TestWavLoader(unittest.TestCase):
    def _write_wav(self, audio, rate=ARATE, sampwidth=2, nch=1):
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(sampwidth)
            w.setframerate(rate)
            if sampwidth == 2:
                w.writeframes((audio * 32767).astype("<i2").tobytes())
            elif sampwidth == 1:
                w.writeframes(((audio * 127) + 128).astype(np.uint8).tobytes())
        return path

    def test_load_16bit_mono(self):
        path = self._write_wav(_tone(1000))
        try:
            audio, rate = load_wav_mono(path)
            self.assertEqual(rate, ARATE)
            self.assertAlmostEqual(_dominant(audio, rate), 1000.0, delta=30)
        finally:
            os.remove(path)

    def test_load_stereo_averages(self):
        mono = _tone(800)
        stereo = np.repeat(mono, 2)   # L=R
        path = self._write_wav(stereo, nch=2)
        try:
            audio, rate = load_wav_mono(path)
            self.assertAlmostEqual(_dominant(audio, rate), 800.0, delta=30)
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        with self.assertRaises(Exception):
            load_wav_mono("/nonexistent/does_not_exist.wav")


class TestTxEngineDeception(unittest.TestCase):
    def setUp(self):
        self.b = TxWaveformBuilder(JammingGenerator(FS), GNSSSpoofer(FS), FS, block_samples=40960)

    def test_wav_deception_dac_safe_and_recovers(self):
        buf, meta = self.b.build({"mode": "ANALOG_SPOOF", "wave_type": WAV_DECEPTION_WAVE,
                                  "decept_audio": _tone(1000), "decept_audio_rate": ARATE,
                                  "decept_mod": "NBFM", "decept_deviation_hz": 3000, "jsr_db": 20})
        self.assertLessEqual(float(np.max(np.abs(buf))), TX_DIGITAL_PEAK + 1e-6)
        self.assertEqual(meta.get("decept_mod"), "NBFM")
        rec, rate = _fm_demod(buf, "FM")
        self.assertAlmostEqual(_dominant(rec, rate), 1000.0, delta=60)

    def test_falls_back_to_synthetic_without_audio(self):
        # WAV seçili ama ses verilmemiş -> sentetik üretime düşer (çökmez, DAC-güvenli)
        buf, _ = self.b.build({"mode": "ANALOG_SPOOF", "wave_type": WAV_DECEPTION_WAVE, "jsr_db": 15})
        self.assertGreater(len(buf), 0)
        self.assertLessEqual(float(np.max(np.abs(buf))), TX_DIGITAL_PEAK + 1e-6)

    def test_synthetic_modes_still_work(self):
        for wave_type in ["Sinüs Dalga (Tone)", "Ses/Audio Sahte Ses",
                          "Gürültü Modüleli AM", "Gürültü Modüleli FM"]:
            buf, _ = self.b.build({"mode": "ANALOG_SPOOF", "wave_type": wave_type, "jsr_db": 30})
            self.assertLessEqual(float(np.max(np.abs(buf))), TX_DIGITAL_PEAK + 1e-6)


class TestWorkerDeceptionLoad(unittest.TestCase):
    def test_worker_loads_wav_on_trigger(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        _ = QApplication.instance() or QApplication(sys.argv)
        from backend.sdr_worker import SDRWorker
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(ARATE)
            w.writeframes((_tone(1000) * 32767).astype("<i2").tobytes())
        try:
            w = SDRWorker()
            w.trigger_tx({"mode": "ANALOG_SPOOF", "wave_type": WAV_DECEPTION_WAVE,
                          "decept_audio_file": path, "decept_mod": "NBFM"})
            self.assertIn("decept_audio", w.tx_params)
            self.assertGreater(len(w.tx_params["decept_audio"]), 0)
            w.stop_tx()
        finally:
            os.remove(path)

    def test_worker_missing_wav_no_fake_audio(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        _ = QApplication.instance() or QApplication(sys.argv)
        from backend.sdr_worker import SDRWorker
        w = SDRWorker()
        w.trigger_tx({"mode": "ANALOG_SPOOF", "wave_type": WAV_DECEPTION_WAVE,
                      "decept_audio_file": "/nonexistent.wav", "decept_mod": "NBFM"})
        # Dosya yok -> decept_audio konmamalı (sentetik/sahte ses zorlanmaz)
        self.assertNotIn("decept_audio", w.tx_params)
        w.stop_tx()


if __name__ == "__main__":
    unittest.main()
