"""spec 5.1.5 KONUM BELİRLEME — hareketli tek alıcı (GPS) ile konum + jeodezik dönüşüm + NMEA.

GPS'li platform hareket ederken biriken (konum, kerteriz) örneklerinden kaynağın konumu üçgenlenir.
Tamamen deterministik/donanımsız: gerçek NMEA cümleleri ve bilinen koordinat dönüşümleriyle test
edilir. Sentetik KONUM üretilmez — fix yoksa fix=False.
"""
import os
import sys
import unittest
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.geo import geodetic_to_enu, geodetic_to_ecef
from backend.gps_receiver import parse_gpgga, GPSReceiver
from backend.direction_finding import MovingReceiverPositioner, bearing_from_positions


def _cks(body: str) -> str:
    c = 0
    for ch in body:
        c ^= ord(ch)
    return f"{c:02X}"


def _gga(body: str) -> str:
    return f"${body}*{_cks(body)}"


class TestGeo(unittest.TestCase):
    def setUp(self):
        self.ref = (39.9200, 32.8500, 900.0)

    def test_north_offset(self):
        # ~100 m kuzey
        enu = geodetic_to_enu(39.9200 + 0.0008993, 32.8500, 900.0, *self.ref)
        self.assertLess(abs(enu[0]), 1.0)
        self.assertAlmostEqual(enu[1], 100.0, delta=2.0)
        self.assertLess(abs(enu[2]), 1.0)

    def test_east_offset(self):
        dlon = 100.0 / (111320.0 * np.cos(np.radians(39.92)))
        enu = geodetic_to_enu(39.9200, 32.8500 + dlon, 900.0, *self.ref)
        self.assertAlmostEqual(enu[0], 100.0, delta=2.0)
        self.assertLess(abs(enu[1]), 1.0)

    def test_up_offset(self):
        enu = geodetic_to_enu(39.9200, 32.8500, 950.0, *self.ref)
        self.assertAlmostEqual(enu[2], 50.0, delta=0.5)

    def test_same_point_is_origin(self):
        enu = geodetic_to_enu(*self.ref, *self.ref)
        np.testing.assert_allclose(enu, [0, 0, 0], atol=1e-6)


class TestNmeaParsing(unittest.TestCase):
    def test_valid_gpgga(self):
        line = _gga("GPGGA,123519,3955.2000,N,03251.0000,E,1,08,0.9,900.0,M,46.9,M,,")
        r = parse_gpgga(line)
        self.assertEqual(r["fix_quality"], 1)
        self.assertAlmostEqual(r["lat"], 39.92, places=3)
        self.assertAlmostEqual(r["lon"], 32.85, places=3)
        self.assertEqual(r["alt"], 900.0)

    def test_no_fix_returns_none_position(self):
        r = parse_gpgga(_gga("GPGGA,123519,,,,,0,00,,,M,,M,,"))
        self.assertEqual(r["fix_quality"], 0)
        self.assertIsNone(r["lat"])

    def test_bad_checksum_rejected(self):
        self.assertIsNone(parse_gpgga("$GPGGA,123519,3955.2000,N,03251.0000,E,1,08,0.9,900.0,M,46.9,M,,*00"))

    def test_gngga_multiconstellation(self):
        line = _gga("GNGGA,001043.00,4404.14036,N,12118.85961,W,1,12,0.98,1113.0,M,-21.3,M,,")
        r = parse_gpgga(line)
        self.assertIsNotNone(r)
        self.assertLess(r["lon"], 0)     # Batı -> negatif

    def test_non_gga_ignored(self):
        self.assertIsNone(parse_gpgga("$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"))

    def test_receiver_without_device(self):
        g = GPSReceiver()
        self.assertIsNone(g.get_position())
        self.assertFalse(g.status()["has_fix"])

    def test_receiver_ingest_updates_position(self):
        g = GPSReceiver()
        g._ingest(_gga("GPGGA,123519,3955.2000,N,03251.0000,E,1,08,0.9,900.0,M,46.9,M,,"))
        pos = g.get_position()
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0], 39.92, places=3)

    def test_receiver_ingest_no_fix_keeps_none(self):
        g = GPSReceiver()
        g._ingest(_gga("GPGGA,123519,,,,,0,00,,,M,,M,,"))
        self.assertIsNone(g.get_position())


class TestMovingReceiverPositioner(unittest.TestCase):
    @staticmethod
    def _run(src, path, minb=15.0):
        mp = MovingReceiverPositioner(min_baseline_m=minb, decay_sec=1e9)
        t = 0.0
        for rx in path:
            rx = np.array(rx, float)
            az, _, el = bearing_from_positions(rx, src)
            mp.add(rx, az, el, now=t)
            t += 1.0
        return mp, mp.estimate(now=t)

    def test_recovers_ground_source(self):
        src = np.array([400.0, 300.0, 0.0])
        path = [(x, 0, 0) for x in range(0, 310, 10)]
        mp, (pos, res, fix, cross) = self._run(src, path)
        self.assertTrue(fix)
        self.assertLess(np.linalg.norm(pos - src), 2.0)
        self.assertGreater(mp.baseline_m(), 100.0)

    def test_recovers_airborne_3d_source(self):
        # L şeklinde yol -> irtifa gözlemlenebilir olur
        src = np.array([200.0, 150.0, 120.0])
        path = [(x, 0, 0) for x in range(0, 210, 10)] + [(200, y, 0) for y in range(10, 160, 10)]
        mp, (pos, res, fix, cross) = self._run(src, path)
        self.assertTrue(fix)
        self.assertLess(np.linalg.norm(pos - src), 2.0)   # irtifa dahil geri bulunur

    def test_spatial_diversity_gate_blocks_stationary(self):
        # Sabit dururken tek örnek -> baz yok -> fix yok (sahte konum üretilmez)
        mp = MovingReceiverPositioner(min_baseline_m=15.0)
        for i in range(20):
            mp.add((0.0, 0.0, 0.0), 45.0, now=i)
        self.assertEqual(mp.sample_count(), 1)
        self.assertFalse(mp.estimate(now=20)[2])

    def test_add_requires_min_baseline(self):
        mp = MovingReceiverPositioner(min_baseline_m=15.0)
        self.assertTrue(mp.add((0, 0, 0), 10.0, now=0))       # ilk her zaman
        self.assertFalse(mp.add((5, 0, 0), 10.0, now=1))      # 5 m < 15 m -> reddedilir
        self.assertTrue(mp.add((20, 0, 0), 10.0, now=2))      # 20 m >= 15 m -> kabul

    def test_reset_clears(self):
        mp = MovingReceiverPositioner()
        mp.add((0, 0, 0), 10.0, now=0)
        mp.add((30, 0, 0), 20.0, now=1)
        mp.reset()
        self.assertEqual(mp.sample_count(), 0)


class TestWorkerMovingIntegration(unittest.TestCase):
    def test_worker_has_moving_payload_and_no_fake_without_gps(self):
        from backend.sdr_worker import SDRWorker
        w = SDRWorker()
        df = w._run_direction_finding(np.ones(2048, dtype=np.complex64) * 0.01, {"snr_db": 2.0})
        self.assertIn("moving", df)
        self.assertFalse(df["moving"]["fix"])            # GPS yok -> sahte konum yok
        self.assertEqual(df["moving"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
