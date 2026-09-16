"""ÖLÇÜLEN anten pattern'i ile PATTERN-EŞLEŞTİRMELİ DF (şartname 5.1.4 — Derece RMS iyileştirme).

Bu modül, tools/import_vna_pattern.py'nin VNA .s2p ölçümlerinden ürettiği data/antenna_pattern.json'u
okur ve iki iş yapar:
  1) İstenen frekans için pattern eğrisini interpole eder (AntennaPattern.pattern_at).
  2) Sahada elle döndürülen antenden toplanan (azimut, güç) örneklerini bu KALİBRE eğriyle EŞLEŞTİRİR
     (template matching) -> kerteriz (bearing) + belirsizlik (sigma). (AntennaPattern.match)

NEDEN pattern eşleştirme (sadece tepe/centroid yerine): tek tepe noktası gürültüye duyarlı; oysa
tüm güç-açı eğrisinin ŞEKLİNİ (ana hüzme + yan/arka loblar) bilinen pattern'e oturtmak, TÜM açılardaki
bilgiyi kullanır -> daha düşük varyans + ilkeli bir sigma (Fisher bilgisi). Zayıf frekanslarda
(ön/arka < eşik, pattern güvenilmez) DF motoru centroid'e döner; bu modül match içinde quality_ok=False
bildirir.

MEKANİK OFFSET: VNA ölçümünde "0° dosyası" tam boresight değil; pattern tepesi ~350°'de. Bu offset
match'te KENDİLİĞİNDEN iptal olur: eşleşme, ölçülen tepenin denk geldiği ENKODER açısını referans
tepeyle hizalar (β = θ_peak civarı), böylece sahadaki enkoder sıfırının VNA sıfırıyla aynı olması
GEREKMEZ. Pattern yalnızca eğrinin ŞEKLİNİ (hüzme genişliği, arka lob) katkılayıp tepe etrafını
rafine eder.

ASLA sentetik/uydurma çalışma-zamanı verisi üretmez: JSON yoksa/bozuksa available()=False döner ve
DF motoru mevcut centroid yöntemini kullanmaya devam eder.
"""
import os
import json
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATTERN_PATH = os.path.join(_PROJECT_ROOT, "data", "antenna_pattern.json")

# Pattern eşleştirmeye GÜVENİLECEK asgari ön/arka oranı (dB). Bunun altında anten o frekansta yeterince
# yönlü değil -> eşleştirme belirsiz -> centroid'e düş. import script'inde medyan ~16 dB, %88 frekans
# >=10 dB; zayıf noktalar (300/900 MHz civarı) bu kapıyla elenir.
MIN_FRONT_BACK_DB = 10.0
# Match için ölçülen eğride gereken asgari açı örneği (tek turda ~360°/örnekleme).
MIN_MATCH_SAMPLES = 8
# sigma tabanı/tavanı (derece): fiziksel olarak hüzme genişliğinden dar bir kerteriz iddia etmeyiz;
# tavan da güvenilmez eşleşmeyi triangulasyonda otomatik zayıf-ağırlıklı yapar.
SIGMA_FLOOR_DEG = 1.5
SIGMA_CEIL_DEG = 45.0


class AntennaPattern:
    """data/antenna_pattern.json yükleyici + frekans interpolasyonu + pattern-eşleştirmeli DF."""

    def __init__(self, path: str = _PATTERN_PATH):
        self.path = path
        self._freqs = None          # (F,) Hz
        self._angles = None         # (A,) derece (0..355, 5° grid)
        self._pattern = None        # (F, A) dB, her satır kendi tepesine normalize (tepe=0)
        self._front_back = None     # (F,) dB
        self._peak_angle = None     # (F,) derece
        self._load()

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self._freqs = np.asarray(d["freqs_hz"], dtype=float)
            self._angles = np.asarray(d["angles_deg"], dtype=float)
            self._pattern = np.asarray(d["pattern_db"], dtype=float)
            q = d.get("quality", {})
            self._front_back = np.asarray(q.get("front_back_db", []), dtype=float)
            self._peak_angle = np.asarray(q.get("peak_angle_deg", []), dtype=float)
            if self._pattern.shape != (len(self._freqs), len(self._angles)):
                raise ValueError("pattern boyutu freqs×angles ile uyumsuz")
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            self._freqs = None       # available()=False -> DF centroid'e düşer

    def available(self) -> bool:
        return self._freqs is not None and len(self._freqs) >= 2

    def _freq_index(self, freq_hz: float) -> int:
        return int(np.argmin(np.abs(self._freqs - float(freq_hz))))

    def pattern_at(self, freq_hz: float):
        """İstenen frekansa EN YAKIN ölçülen frekansın pattern eğrisini döner.
        Dönüş: (angles(A,), pattern_db(A,), front_back_db, peak_angle_deg) veya None (mevcut değil).
        Not: frekans grid'i 20 MHz; en yakın ölçüm alınır (yayın bant genişliği « 20 MHz olduğundan
        pattern şekli komşu frekanslarda pratikte aynıdır)."""
        if not self.available():
            return None
        i = self._freq_index(freq_hz)
        fb = float(self._front_back[i]) if len(self._front_back) > i else 0.0
        pk = float(self._peak_angle[i]) if len(self._peak_angle) > i else 0.0
        return self._angles.copy(), self._pattern[i].copy(), fb, pk

    def match(self, meas_az_deg, meas_power_db, freq_hz):
        """PATTERN EŞLEŞTİRME: ölçülen (azimut, güç) örneklerini frekansın kalibre pattern'ine oturtur.

        Model: kaynak β kerterizindeyken, enkoder θ'ya bakan antenin gücü ~ P_ref(β - θ). P_ref tepesi
        φ_peak'te olduğundan boresight-çerçevesi P_ref((x + φ_peak) mod 360). Her iki eğri de kendi
        tepesine normalize edilerek mutlak güç (kaynak gücü/mesafe) elenir. Maliyet:
            C(β) = Σ_i [ Pmeas_norm(θ_i) - Pref_bore(β - θ_i) ]²
        β* = argmin C. Tepe civarı parabol uydurup Fisher bilgisiyle sigma:  σ_β ≈ sqrt( (C_min/(n-1)) / a )
        (a = parabol eğrilik katsayısı, C≈C_min + a(β-β*)²).

        Girdi: meas_az_deg (list/array derece), meas_power_db (list/array dB, mutlak seviye önemsiz),
               freq_hz.
        Dönüş: dict {bearing_deg, sigma_deg, quality_ok(bool), n, front_back_db, cost_min} veya None.
        quality_ok=False ise (az örnek / düşük ön-arka / bozuk eşleşme) çağıran centroid'e düşmeli.
        """
        if not self.available():
            return None
        az = np.asarray(meas_az_deg, dtype=float) % 360.0
        pm = np.asarray(meas_power_db, dtype=float)
        n = len(az)
        if n < MIN_MATCH_SAMPLES or n != len(pm):
            return None
        ang, pref, fb, pk = self.pattern_at(freq_hz)
        pm = pm - np.max(pm)                                   # ölçüleni tepeye normalize (pref zaten normalize)
        # Boresight-çerçevesi referans: fonksiyon x(derece) -> dB, dairesel interpolasyonla.
        ang_ext = np.concatenate([ang - 360.0, ang, ang + 360.0])
        pref_ext = np.concatenate([pref, pref, pref])

        def pref_bore(x_deg):
            # x = boresight'tan açı farkı; ölçüm-çerçevesinde tepe φ_peak(pk)'te -> kaydır
            return np.interp((np.asarray(x_deg) + pk) % 360.0, ang_ext, pref_ext)

        betas = np.arange(0.0, 360.0, 1.0)                    # 1° arama
        cost = np.empty_like(betas)
        for k, b in enumerate(betas):
            resid = pm - pref_bore(b - az)
            cost[k] = float(np.dot(resid, resid))
        kmin = int(np.argmin(cost))
        beta = float(betas[kmin])
        cmin = float(cost[kmin])

        # Tepe civarı 3 nokta ile parabol -> hassas β + eğrilik a
        k0, k1, k2 = (kmin - 1) % len(betas), kmin, (kmin + 1) % len(betas)
        c0, c1, c2 = cost[k0], cost[k1], cost[k2]
        a = 0.5 * (c0 + c2 - 2.0 * c1)                        # parabol eğrilik (>0 tepe civarı)
        if a > 1e-9:
            delta = 0.5 * (c0 - c2) / (c0 - 2.0 * c1 + c2)    # [-0.5,0.5] alt-derece düzeltme
            beta = (beta + delta) % 360.0
            sigma = float(np.sqrt(max(cmin, 1e-9) / max(n - 1, 1) / a))
        else:
            sigma = SIGMA_CEIL_DEG
        sigma = float(np.clip(sigma, SIGMA_FLOOR_DEG, SIGMA_CEIL_DEG))

        quality_ok = (fb >= MIN_FRONT_BACK_DB) and (a > 1e-9) and (sigma < SIGMA_CEIL_DEG)
        return {
            "bearing_deg": round(beta, 2),
            "sigma_deg": round(sigma, 2),
            "quality_ok": bool(quality_ok),
            "n": n,
            "front_back_db": round(fb, 1),
            "cost_min": round(cmin, 4),
        }


# Modül düzeyinde tek örnek (tekrar tekrar JSON okumamak için). DF motoru bunu paylaşır.
_SHARED = None


def shared_pattern() -> AntennaPattern:
    global _SHARED
    if _SHARED is None:
        _SHARED = AntennaPattern()
    return _SHARED
