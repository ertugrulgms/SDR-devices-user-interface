"""Çok-sistem GNSS aldatma testleri (şartname 5.2.4: en çok serviste aldatma).

Her GNSS servisinin KENDİ SİSTEMİNİN yapısıyla üretildiğini kilitler:
  - GPS: CDMA C/A Gold kod (1023 çip, 1.023 Mçip/s), BPSK.
  - GLONASS: 511-çip ortak m-dizisi + FDMA (çok-taşıyıcı).
  - Galileo: 4092/5115 çip kod + BOC (bölünmüş spektrum).
  - Beidou: 2046-çip Gold kod + BOC/BPSK.
Donanımsız/deterministik.
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.gnss_spoofer import GNSSSpoofer


class TestCodeGenerators(unittest.TestCase):
    def setUp(self):
        self.g = GNSSSpoofer(sample_rate_hz=10e6)

    def test_glonass_code_length_and_polarity(self):
        code = self.g.generate_glonass_code()
        self.assertEqual(len(code), 511)                  # GLONASS C/A: 511 çip
        self.assertTrue(set(np.unique(code)).issubset({-1, 1}))

    def test_glonass_code_is_shared(self):
        # GLONASS'ta tüm uydular AYNI kodu kullanır (FDMA ile ayrılır)
        self.assertIs(self.g.generate_glonass_code(), self.g.generate_glonass_code())

    def test_beidou_code_length(self):
        code = self.g.generate_beidou_code(1)
        self.assertEqual(len(code), 2046)                 # Beidou B1I: 2046 çip
        self.assertTrue(set(np.unique(code)).issubset({-1, 1}))

    def test_beidou_prns_differ(self):
        self.assertFalse(np.array_equal(self.g.generate_beidou_code(1), self.g.generate_beidou_code(2)))

    def test_galileo_code_length(self):
        code = self.g.generate_galileo_code(1, 4092)
        self.assertEqual(len(code), 4092)                 # Galileo E1: 4092 çip
        self.assertTrue(set(np.unique(code)).issubset({-1, 1}))

    def test_mseq_is_maximal_length_balanced(self):
        # m-dizisi dengeli olmalı: +1 ve -1 sayısı ~eşit (|toplam| küçük)
        code = self.g.generate_glonass_code()
        self.assertLessEqual(abs(int(np.sum(code))), 33)


class TestSpecTable(unittest.TestCase):
    def test_all_13_services_have_spec(self):
        for svc in GNSSSpoofer.GNSS_SERVICES:
            self.assertIn(svc, GNSSSpoofer.GNSS_SIGNAL_SPEC, f"{svc} spec'te yok")

    def test_glonass_is_fdma(self):
        self.assertTrue(GNSSSpoofer.GNSS_SIGNAL_SPEC["GLONASS L1"]["fdma"])
        self.assertFalse(GNSSSpoofer.GNSS_SIGNAL_SPEC["GPS L1"]["fdma"])

    def test_galileo_uses_boc(self):
        # Galileo E1 = BOC(1,1), E6 = BOC(5,5) -> bölünmüş spektrum
        self.assertEqual(GNSSSpoofer.GNSS_SIGNAL_SPEC["GALILEO E1"]["mod"], "BOC")
        self.assertEqual(GNSSSpoofer.GNSS_SIGNAL_SPEC["GALILEO E6"]["mod"], "BOC")
        self.assertGreater(GNSSSpoofer.GNSS_SIGNAL_SPEC["GALILEO E1"]["boc"], 0)

    def test_beidou_b1i_is_bpsk(self):
        # BEIDOU B1 = B1I (1561.098 MHz) -> gerçekte BPSK(2), alt-taşıyıcı YOK (BOC değil).
        self.assertEqual(GNSSSpoofer.GNSS_SIGNAL_SPEC["BEIDOU B1"]["mod"], "BPSK")
        self.assertEqual(GNSSSpoofer.GNSS_SIGNAL_SPEC["BEIDOU B1"]["boc"], 0)
        self.assertAlmostEqual(GNSSSpoofer.GNSS_SERVICES["BEIDOU B1"], 1561.098, places=3)


class TestServiceSynthesis(unittest.TestCase):
    def _synth(self, svc, n=32768):
        spec = GNSSSpoofer.GNSS_SIGNAL_SPEC[svc]
        g = GNSSSpoofer(sample_rate_hz=spec["fs"])
        return g.synthesize_service(svc, n, amplitude=0.5), spec

    def test_all_services_produce_valid_signal(self):
        for svc in GNSSSpoofer.GNSS_SERVICES:
            buf, _ = self._synth(svc, n=8192)
            self.assertEqual(buf.dtype, np.complex64, svc)
            self.assertTrue(np.all(np.isfinite(buf)), svc)
            self.assertGreater(float(np.max(np.abs(buf))), 0.0, svc)

    @staticmethod
    def _center_dip(buf, fs, boc_hz):
        """BOC bölünmüş spektrum: merkezde çukur, ±boc_hz'de yan-lob tepeleri."""
        S = np.abs(np.fft.fftshift(np.fft.fft(buf * np.hamming(len(buf))))) ** 2
        n = len(S)
        c = n // 2
        w = max(1, int(0.1e6 / fs * n))                   # ±100 kHz merkez penceresi
        off = int(boc_hz / fs * n)                        # ±boc bin ofseti
        center = float(np.mean(S[c - w:c + w]))
        sides = float(np.mean(S[c + off - w:c + off + w]) + np.mean(S[c - off - w:c - off + w]))
        return center, sides

    def test_boc_produces_split_spectrum(self):
        # Galileo E1 BOC(1,1): merkez enerji, yan-loblardan (±1.023 MHz) DÜŞÜK olmalı
        buf, spec = self._synth("GALILEO E1")
        center, sides = self._center_dip(buf, spec["fs"], spec["boc"])
        self.assertLess(center, sides, "BOC bölünmüş spektrum yok (merkez çukur değil)")

    def test_gps_is_wideband_spread_spectrum(self):
        # GPS C/A BPSK: yayılı spektrum -> enerji ~çip-oranı (±1.023 MHz) genişliğine yayılır,
        # tek dar tona toplanmaz. İşgal bandı çip-oranı mertebesinde olmalı.
        buf, spec = self._synth("GPS L1")
        S = np.abs(np.fft.fftshift(np.fft.fft(buf * np.hamming(len(buf))))) ** 2
        n = len(S)
        total = S.sum() + 1e-12
        cs = np.cumsum(S)
        lo = np.searchsorted(cs, 0.05 * total)
        hi = np.searchsorted(cs, 0.95 * total)
        occ_hz = (hi - lo) / n * spec["fs"]
        self.assertGreater(occ_hz, 0.7e6, "GPS yayılı-spektrum bandı çip-oranı mertebesinde değil")

    def test_boc_has_higher_side_energy_than_bpsk(self):
        # BOC (Galileo) ±alt-taşıyıcıda enerji yoğunlaştırır; BPSK (GPS) yoğunlaştırmaz.
        def side_ratio(svc):
            buf, spec = self._synth(svc)
            S = np.abs(np.fft.fftshift(np.fft.fft(buf * np.hamming(len(buf))))) ** 2
            n = len(S); c = n // 2
            off = int(spec.get("boc", 1.023e6) / spec["fs"] * n)
            w = max(1, int(0.15e6 / spec["fs"] * n))
            center = float(np.mean(S[c - w:c + w])) + 1e-12
            side = float(np.mean(S[c + off - w:c + off + w]) + np.mean(S[c - off - w:c - off + w]))
            return side / center
        # Galileo BOC'un yan/merkez oranı, GPS BPSK'nınkinden belirgin YÜKSEK olmalı
        self.assertGreater(side_ratio("GALILEO E1"), side_ratio("GPS L1"))

    def test_glonass_fdma_multiple_carriers(self):
        # GLONASS FDMA: enerji tek merkezde değil, birden çok frekans ofsetinde (kanallar)
        buf, spec = self._synth("GLONASS L1")
        S = np.abs(np.fft.fftshift(np.fft.fft(buf * np.hamming(len(buf))))) ** 2
        S /= S.max()
        n = len(S); c = n // 2
        chan_bins = int(spec["chan"] / spec["fs"] * n)
        # Merkez-dışı kanal konumunda (±chan) anlamlı enerji olmalı
        side_energy = np.max(S[c + chan_bins - 50:c + chan_bins + 50])
        self.assertGreater(side_energy, 0.1, "FDMA kanal enerjisi yok")


class TestDopplerAndNav(unittest.TestCase):
    def test_doppler_scales_with_service_carrier(self):
        # Doppler servisin taşıyıcısıyla ölçeklenir: L1 (1575) / L5 (1176) ≈ 1.339 (fiziksel doğruluk)
        g = GNSSSpoofer(sample_rate_hz=10e6)
        prns = [1, 2, 3, 4]
        kL1 = g._get_satellite_kinematics(39.92, 32.85, 100, prns, 0.0, 1575.42e6)
        kL5 = g._get_satellite_kinematics(39.92, 32.85, 100, prns, 0.0, 1176.45e6)
        d1 = np.mean([abs(kL1[p]["doppler_hz"]) for p in prns])
        d5 = np.mean([abs(kL5[p]["doppler_hz"]) for p in prns])
        self.assertAlmostEqual(d1 / d5, 1575.42 / 1176.45, places=2)

    def test_nav_data_seed_reproducible_and_varied(self):
        g = GNSSSpoofer(sample_rate_hz=10e6)
        a = g.generate_nav_data(num_bits=300, seed=7)
        b = g.generate_nav_data(num_bits=300, seed=7)
        c = g.generate_nav_data(num_bits=300, seed=8)
        np.testing.assert_array_equal(a, b)                    # aynı seed -> tekrarlanabilir
        self.assertFalse(np.array_equal(a, c))                # farklı seed -> farklı dolgu
        # Ephemeris dolgusu sabit-1 DEĞİL (gerçekçi değişken bit)
        self.assertGreater(len(set(a[25:].tolist())), 1)
        self.assertTrue(set(np.unique(a)).issubset({-1.0, 1.0}))


if __name__ == "__main__":
    unittest.main()
