"""GPS-SDR-SIM köprüsü (spec 5.2.4) — gerçek ephemeris'li GPS L1 spoofing baseband'i.

GPS-SDR-SIM (osqzss), RINEX broadcast ephemeris + sahte koordinattan GEÇERLİ nav mesajlı (subframe'li)
tam GPS L1 C/A baseband üretir. Kendi çok-sistem üretecimiz sinyal-seviyesi genişliği (GPS L1 + E1 +
B1) sağlarken, bu köprü L1 için GERÇEK-ALICI-KANDIRAN spoofing sunar (subframe'i biz yazmayız; kanıtlı
araç RINEX'i baseband'e çevirir — DSD-FME felsefesi).

Sentetik ses/veri ÜRETMEZ; kurulu değilse available()=False. Baseband dosyası önceden üretilir
(tools/gen_gps_spoof.py) ve worker onu TX'ten yayınlar.
"""
import os
import shutil
import subprocess
import numpy as np

_BINARIES = ("gps-sdr-sim",)
DEFAULT_FS_HZ = 2_600_000.0     # GPS-SDR-SIM varsayılan örnekleme (SDR bu hızda TX yapmalı)


def find_binary():
    for name in _BINARIES:
        p = shutil.which(name)
        if p:
            return p
    # ~/.local/bin PATH'te değilse elle bak
    cand = os.path.expanduser("~/.local/bin/gps-sdr-sim")
    return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None


def available() -> bool:
    return find_binary() is not None


def install_hint() -> str:
    return ("GPS-SDR-SIM kurulu değil. Gerçek-ephemeris GPS L1 spoofing için "
            "tools/install_gps_sdr_sim.sh çalıştırın (tek dosya C, sudo gerekmez).")


def generate(rinex_path: str, lat: float, lon: float, alt_m: float, out_path: str,
             fs_hz: float = DEFAULT_FS_HZ, duration_s: int = 30, iq_bits: int = 16,
             start_datetime: str = None) -> dict:
    """RINEX + sahte konumdan GPS L1 baseband (.bin) üretir. Dönüş: {'ok','error','path','fs_hz','bits'}."""
    binp = find_binary()
    if binp is None:
        return {"ok": False, "error": "gps-sdr-sim bulunamadı", "path": None}
    if not (rinex_path and os.path.isfile(rinex_path)):
        return {"ok": False, "error": f"RINEX bulunamadı: {rinex_path}", "path": None}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [binp, "-e", rinex_path,
           "-l", f"{float(lat):.6f},{float(lon):.6f},{float(alt_m):.1f}",
           "-s", str(int(fs_hz)), "-b", str(int(iq_bits)),
           "-d", str(int(duration_s)), "-o", out_path]
    if start_datetime:
        cmd += ["-t", start_datetime]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as e:
        return {"ok": False, "error": f"gps-sdr-sim çalıştırılamadı: {e}", "path": None}
    if r.returncode != 0 or not os.path.isfile(out_path):
        return {"ok": False, "error": (r.stderr or r.stdout or "üretim başarısız")[-400:], "path": None}
    return {"ok": True, "error": None, "path": out_path, "fs_hz": float(fs_hz), "bits": int(iq_bits)}


def load_baseband(path: str, iq_bits: int = 16, amplitude: float = 0.9) -> np.ndarray:
    """GPS-SDR-SIM .bin (ARAYÜZLENMİŞ I,Q) dosyasını complex64 baseband'e çevirir; [-amplitude,amplitude]
    tepe içine normalize eder (DAC-güvenli). 16-bit: int16 I,Q; 8-bit: int8 I,Q."""
    if not (path and os.path.isfile(path)):
        raise FileNotFoundError(path)
    dtype = np.int16 if int(iq_bits) == 16 else (np.int8 if int(iq_bits) == 8 else None)
    if dtype is None:
        raise ValueError("iq_bits 8 veya 16 olmalı")
    raw = np.fromfile(path, dtype=dtype)
    if raw.size < 2:
        raise ValueError("boş/bozuk baseband dosyası")
    raw = raw[: (raw.size // 2) * 2].reshape(-1, 2).astype(np.float32)
    full = 32768.0 if int(iq_bits) == 16 else 128.0
    iq = (raw[:, 0] + 1j * raw[:, 1]) / full           # [-1,1]
    pk = float(np.max(np.abs(iq))) + 1e-12
    return (amplitude * iq / pk).astype(np.complex64)
