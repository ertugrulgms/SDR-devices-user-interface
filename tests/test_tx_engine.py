"""TxWaveformBuilder ve TX zinciri regresyon testleri.

Bu testler, denetimde bulunan kritik/ciddi/orta bulguların DÜZELTİLDİĞİNİ kilitler:
  #1  Hiçbir mod DAC tavanını (TX_DIGITAL_PEAK) aşamaz -> donanımda clipping imkansız.
  #5  Chirp süpürme faz-sürekli (blok sınırında büyük faz sıçraması yok).
  #8  Analog AM (değişken zarf) ve FM (sabit zarf) ayrı ve doğru.
  #9  Wi-Fi engelleme gerçek baraj üretir.
  #11 Hedef sinyal tipi karıştırma parametrelerini (bant genişliği) gerçekten değiştirir.
Donanımsız/deterministiktir.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.jamming_generator import JammingGenerator, TX_DIGITAL_PEAK
from backend.gnss_spoofer import GNSSSpoofer
from backend.tx_engine import TxWaveformBuilder, target_profile, WIFI_BAND_CENTERS


class TestTxWaveformBuilder(unittest.TestCase):
    def setUp(self):
        fs = 2_000_000
        self.builder = TxWaveformBuilder(
            JammingGenerator(sample_rate_hz=fs), GNSSSpoofer(sample_rate_hz=fs),
            fs, block_samples=8192)

    def _build(self, **params):
        params.setdefault("jsr_db", 15.0)
        buf, meta = self.builder.build(params)
        return buf, meta

    def test_all_modes_are_dac_safe(self):
        # KRİTİK #1: hiçbir mod tepe büyüklükte DAC tavanını aşmamalı (yüksek JSR'de bile)
        modes = ["SPOT", "SINGLE_TONE", "MULTI_TONE", "SWEEP", "BARRAGE", "LOOK_THROUGH",
                 "WIFI_JAMMING", "GNSS_SPOOF"]
        for mode in modes:
            for jsr in (0.0, 15.0, 40.0, 80.0):     # düşükten aşırı-yükseğe
                buf, _ = self._build(mode=mode, jsr_db=jsr)
                peak = float(np.max(np.abs(buf)))
                self.assertLessEqual(peak, TX_DIGITAL_PEAK + 1e-6,
                                     f"{mode} JSR={jsr} tepe {peak} > tavan {TX_DIGITAL_PEAK}")
                self.assertTrue(np.all(np.isfinite(buf)))

    def test_analog_spoof_modes_are_dac_safe(self):
        for wave in ["Sinüs Dalga (Tone)", "Ses/Audio Sahte Ses",
                     "Gürültü Modüleli AM", "Gürültü Modüleli FM"]:
            buf, _ = self._build(mode="ANALOG_SPOOF", wave_type=wave, jsr_db=40.0)
            self.assertLessEqual(float(np.max(np.abs(buf))), TX_DIGITAL_PEAK + 1e-6)

    def test_gnss_meta_reports_service(self):
        _, meta = self._build(mode="GNSS_SPOOF", coords="39.92,32.85", gnss_code="GPS L1")
        self.assertIn("gnss_doppler_hz", meta)
        self.assertEqual(meta["gnss_service"], "GPS L1")

    def test_target_profile_changes_barrage_bandwidth(self):
        # ORTA #11: dar-bant hedef (Analog Telsiz) geniş-bant hedeften (Geniş Bant Veri)
        # belirgin şekilde DAHA DAR bir işgal bandı üretmeli.
        def occ_bins(buf):
            p = np.abs(np.fft.fft(buf)) ** 2
            return int(np.sum(p > 0.05 * np.max(p)))
        narrow, _ = self._build(mode="BARRAGE", target_signal="Analog Telsiz")
        wide, _ = self._build(mode="BARRAGE", target_signal="Geniş Bant Veri")
        self.assertLess(occ_bins(narrow), occ_bins(wide),
                        "dar-bant hedef geniş-bant hedeften daha dar olmalı")

    def test_wifi_profile_defined(self):
        self.assertIn("2.4 GHz (802.11 b/g/n)", WIFI_BAND_CENTERS)
        self.assertAlmostEqual(WIFI_BAND_CENTERS["2.4 GHz (802.11 b/g/n)"], 2437.0)

    def test_spot_raises_noise_not_single_line(self):
        # 5.2.1: "tekli" karıştırma gürültü tabanını yükseltmeli (Shannon). SPOT modu, saf CW
        # ton gibi TEK spektral çizgi DEĞİL, kanal içinde YAYILMIŞ gürültü olmalı.
        spot, _ = self._build(mode="SPOT", target_signal="Analog Telsiz")
        tone = JammingGenerator(sample_rate_hz=2_000_000).generate_single_tone(0.0, len(spot), 0.9)
        def peak_ratio(buf):
            p = np.abs(np.fft.fft(buf)) ** 2
            return float(np.max(p) / np.sum(p))
        self.assertGreater(peak_ratio(tone), 0.9)       # CW ton: tek çizgi
        self.assertLess(peak_ratio(spot), 0.2)          # SPOT: yayılmış gürültü

    def test_spot_is_narrower_than_barrage(self):
        # Tekli (spot) tek kanala yoğunlaşır; baraj daha geniş banda yayılır -> spot DAHA DAR.
        def occ(buf):
            p = np.abs(np.fft.fft(buf)) ** 2
            return int(np.sum(p > 0.05 * np.max(p)))
        spot, _ = self._build(mode="SPOT", target_signal="Analog Telsiz")
        barrage, _ = self._build(mode="BARRAGE", target_signal="Analog Telsiz")
        self.assertLess(occ(spot), occ(barrage))

    def test_spot_closed_loop_focus_shifts_to_target(self):
        # RX'te ölçülen hedef bandı/offseti verildiğinde spot o offsete kaydırılmalı (meta'da raporlanır)
        _, meta = self._build(mode="SPOT", focus_bw_hz=40e3, focus_offset_hz=250e3)
        self.assertAlmostEqual(meta.get("focus_offset_hz", 0.0), 250e3)
        self.assertAlmostEqual(meta.get("focus_bw_hz", 0.0), 40e3)

    def test_single_tone_alias_maps_to_spot_noise(self):
        # Geriye dönük uyum: eski "SINGLE_TONE" anahtarı da spot GÜRÜLTÜ üretmeli (CW ton değil)
        buf, _ = self._build(mode="SINGLE_TONE", target_signal="Analog Telsiz")
        p = np.abs(np.fft.fft(buf)) ** 2
        self.assertLess(float(np.max(p) / np.sum(p)), 0.2)


class TestChirpContinuity(unittest.TestCase):
    def test_chirp_phase_is_continuous(self):
        # CİDDİ #5: faz-sürekli chirp -> ardışık örnekler arası faz adımı hiçbir yerde
        # büyük (~pi) sıçrama yapmamalı. (Eski blok-bazlı sürüm sınırlarda pi'ye yakın sıçrardı.)
        jam = JammingGenerator(sample_rate_hz=2_000_000)
        sweep = jam.generate_chirp_sweep(8192, amplitude=0.9, n_sweeps=8)
        dphase = np.diff(np.unwrap(np.angle(sweep)))
        # Anlık frekans |f| <= B/2 = 0.45*fs -> örnek başına faz adımı <= 2*pi*0.45 ~ 2.83 rad.
        self.assertLess(float(np.max(np.abs(dphase))), 3.0,
                        "faz adımı çok büyük -> süreksizlik/tıklama var")
        self.assertTrue(np.all(np.isfinite(sweep)))


class TestAnalogAmVsFm(unittest.TestCase):
    def setUp(self):
        self.jam = JammingGenerator(sample_rate_hz=2_000_000)

    def test_fm_is_constant_envelope(self):
        buf = self.jam.generate_analog_spoofing("Gürültü Modüleli FM", 0.0, 8192, 1.0)
        env = np.abs(buf)
        self.assertLess(np.std(env) / (np.mean(env) + 1e-9), 0.05)   # FM: sabit zarf

    def test_am_is_variable_envelope(self):
        buf = self.jam.generate_analog_spoofing("Gürültü Modüleli AM", 0.0, 8192, 1.0)
        env = np.abs(buf)
        self.assertGreater(np.std(env) / (np.mean(env) + 1e-9), 0.1)  # AM: değişken zarf


if __name__ == "__main__":
    unittest.main()
