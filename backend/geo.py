"""Jeodezik (WGS84 enlem/boylam/irtifa) <-> yerel ENU (Doğu-Kuzey-Yukarı, metre) dönüşümleri.

GPS alıcısı enlem/boylam/irtifa verir; yön bulma/konum motoru ise yerel ENU metre çalışır
(x=Doğu, y=Kuzey, z=Yukarı). Hareketli-alıcı konum belirlemede, alıcının her GPS konumu bir
referans orijine göre ENU metreye çevrilir; böylece platform hareket ettikçe farklı konumlardan
alınan kerterizler (LOB) üçgenlenebilir.

Saf matematik — donanımsız, deterministik, birim-test edilebilir. Standart WGS84 elipsoidi.
"""
import numpy as np

# WGS84 elipsoid parametreleri
_A = 6378137.0                      # büyük yarı eksen (m)
_F = 1.0 / 298.257223563            # basıklık
_E2 = _F * (2.0 - _F)               # birinci dışmerkezlik karesi


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Jeodezik (derece, derece, metre) -> ECEF (Earth-Centered Earth-Fixed) [X,Y,Z] metre."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    n = _A / np.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_enu(ecef: np.ndarray, ref_lat_deg: float, ref_lon_deg: float,
                ref_ecef: np.ndarray) -> np.ndarray:
    """ECEF [X,Y,Z] -> referans noktasına göre yerel ENU [Doğu, Kuzey, Yukarı] metre."""
    lat = np.radians(ref_lat_deg)
    lon = np.radians(ref_lon_deg)
    d = np.asarray(ecef, float) - np.asarray(ref_ecef, float)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    # ECEF->ENU dönüş matrisi
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return np.array([east @ d, north @ d, up @ d], dtype=float)


def geodetic_to_enu(lat_deg: float, lon_deg: float, alt_m: float,
                    ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    """Jeodezik nokta -> referans jeodezik orijine göre yerel ENU [Doğu, Kuzey, Yukarı] metre."""
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    ref_ecef = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    return ecef_to_enu(ecef, ref_lat_deg, ref_lon_deg, ref_ecef)
