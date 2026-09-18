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
# AÇISAL KAPSAMA KAPISI (uzman P0): örnek SAYISI ≠ açısal KAPSAMA. 8 örnek 8°'lik dar bir yayda
# toplanmışken 360° template araması ill-posed'dir (yanlış kesin kerteriz). Pattern eşleştirmeye
# ancak yeterli açı gözlendiğinde GÜVEN: ön VE arka bölge görülmeli (front/back ayrımı için ~180°+).
# Eşik ampiriktir (saha verisiyle kalibre edilecek); altında centroid'e düşülür.
MIN_COVERAGE_DEG = 180.0
# AMBIGUITY (uzman P1): en iyi çözümün yanında NEREDEYSE eşit ikinci bir minimum varsa (yan lob /
# zayıf ön-arka / çok yollu), çözüm belirsizdir. GÜRÜLTÜ-FARKINDA z-skoru: ikincil minimumun birincilden
# kaç "gürültü-sigma" uzakta olduğu. C_min ≈ n·σ² (gürültünün SSE'si) olduğundan σ²≈C_min/n; maliyet
# dalgalanma std'i ≈ σ²·√(2n). z = (C_second/C_min − 1)·√(n/2) = ikincil-birincil farkının kaç-sigma'sı.
# z < eşik -> gürültü ikincili öne geçirebilir (flip riski) -> quality_ok=False -> centroid.
# (Naif (c2-c1)/(max-min) veya derinlik-oranı metrikleri yüksek yönlülükte iyi frekansları yanlış
# reddediyordu; z-skoru flip-olasılığına dayanır ve ölçek-bağımsızdır. Eşik ampiriktir, saha kalibreli.)
MIN_AMBIGUITY_Z = 3.0
AMBIGUITY_GUARD_DEG = 25.0   # birincil minimum etrafında bu pencere ikincil aramadan hariç
# sigma tabanı/tavanı (derece): fiziksel olarak hüzme genişliğinden dar bir kerteriz iddia etmeyiz;
# tavan da güvenilmez eşleşmeyi triangulasyonda otomatik zayıf-ağırlıklı yapar. NOT: 1.5° taban FİZİKSEL
# olarak KANITLANMIŞ bir alt sınır DEĞİLDİR — gerçek RF yön referanslarıyla ampirik doğrulanmalıdır.
SIGMA_FLOOR_DEG = 1.5
SIGMA_CEIL_DEG = 45.0
# Kalibrasyon frekans-grid adımının bu kadar katına kadar aralık dışına izin ver (kenar toleransı);
# ötesinde pattern UNAVAILABLE (sessizce uç-noktaya sıçrama YOK — uzman P1 aralık kapısı).
FREQ_RANGE_MARGIN_STEPS = 0.5


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

    def in_range(self, freq_hz: float) -> bool:
        """Frekans kalibrasyon aralığında mı (kenar toleransı dahil). Dışındaysa pattern kullanılmaz."""
        if not self.available():
            return False
        step = float(self._freqs[1] - self._freqs[0]) if len(self._freqs) > 1 else 20e6
        margin = FREQ_RANGE_MARGIN_STEPS * step
        return (self._freqs[0] - margin) <= float(freq_hz) <= (self._freqs[-1] + margin)

    def pattern_at(self, freq_hz: float):
        """İstenen frekansın pattern eğrisini komşu iki kalibre dilim arasında LİNEER (güç alanında)
        İNTERPOLE ederek döner. Aralık DIŞINDAysa None (uç-noktaya sessizce sıçramaz — uzman P1).
        Dönüş: (angles(A,), pattern_db(A,), front_back_db, peak_angle_deg) veya None.
        İnterpolasyon güç (doğrusal) alanında yapılıp tekrar tepeye normalize edilir; fb ve pk
        interpolasyon SONUCU eğriden yeniden hesaplanır (tutarlılık)."""
        if not self.available() or not self.in_range(freq_hz):
            return None
        f = float(freq_hz)
        fr = self._freqs
        # Bracketing iki dilim + ağırlık
        if f <= fr[0]:
            i0 = i1 = 0; w = 0.0
        elif f >= fr[-1]:
            i0 = i1 = len(fr) - 1; w = 0.0
        else:
            i1 = int(np.searchsorted(fr, f))
            i0 = i1 - 1
            w = (f - fr[i0]) / (fr[i1] - fr[i0] + 1e-12)     # 0 -> i0, 1 -> i1
        # dB pattern -> doğrusal güç -> ağırlıklı ortalama -> dB -> tepeye normalize
        p0 = np.power(10.0, self._pattern[i0] / 10.0)
        p1 = np.power(10.0, self._pattern[i1] / 10.0)
        lin = (1.0 - w) * p0 + w * p1
        db = 10.0 * np.log10(lin + 1e-12)
        db = db - db.max()
        ang = self._angles
        ipk = int(np.argmax(db))
        pk = float(ang[ipk])
        back = (pk + 180.0) % 360.0
        iback = int(np.argmin(np.abs(((ang - back + 180.0) % 360.0) - 180.0)))
        fb = float(db[ipk] - db[iback])
        return ang.copy(), db, fb, pk

    @staticmethod
    def coverage_deg(az_deg) -> float:
        """Örneklerin açısal KAPSAMASI (derece): 360 − en büyük dairesel boşluk. Tam turda ~360,
        dar yayda küçük. Pattern eşleştirmenin güvenilir olması için (ön+arka görülmeli) gerekir."""
        a = np.sort(np.asarray(az_deg, dtype=float) % 360.0)
        if len(a) < 2:
            return 0.0
        gaps = np.diff(a)
        wrap = (a[0] + 360.0) - a[-1]                          # son -> ilk (dairesel) boşluk
        largest = float(max(gaps.max(), wrap))
        return float(max(0.0, 360.0 - largest))

    def match(self, meas_az_deg, meas_power_db, freq_hz, arc_center=None, arc_half=None):
        """PATTERN EŞLEŞTİRME: ölçülen (azimut, güç) örneklerini frekansın kalibre pattern'ine oturtur.

        İLERİ-YAY KISITI (arc_center, arc_half): verilirse β araması YALNIZCA [merkez±yarı] içinde yapılır
        (arkadaki/yay-dışı çözümler elenir; dar ileri-taramada template araması iyi-tanımlı olur, arka lob
        doğal olarak dışarıda kalır -> ambiguity iyileşir). None ise tam 360° aranır (varsayılan, kapalı).

        Model: kaynak β kerterizindeyken, enkoder θ'ya bakan antenin gücü ~ P_ref(β - θ). P_ref tepesi
        φ_peak'te olduğundan boresight-çerçevesi P_ref((x + φ_peak) mod 360). Her iki eğri de kendi
        tepesine normalize edilerek mutlak güç (kaynak gücü/mesafe) elenir. Maliyet:
            C(β) = Σ_i [ Pmeas_norm(θ_i) - Pref_bore(β - θ_i) ]²
        β* = argmin C. Tepe civarı parabol uydurulup alt-derece β + belirsizlik (σ) kestirilir.

        SİGMA: σ_β ≈ sqrt( (C_min/(n-1)) / a ), a = parabol eğrilik katsayısı. Bu, maliyet eğrisinin
        yerel EĞRİLİĞİNDEN türetilen bir belirsizlik KESTİRİMİDİR (Gauss-gürültü/Fisher yorumu ancak
        gerçek gürültü modeli + yön referanslarıyla AMPİRİK doğrulanırsa savunulabilir). Şu an
        heuristic'tir; SIGMA_FLOOR_DEG fiziksel bir alt sınır garantisi DEĞİLDİR.

        KALİTE KAPILARI (quality_ok True olması için hepsi gerekir):
          - frekans kalibrasyon aralığında (pattern_at None dönmemeli),
          - açısal kapsama >= MIN_COVERAGE_DEG (dar yayda template araması ill-posed),
          - ön/arka >= MIN_FRONT_BACK_DB (anten o frekansta yeterince yönlü),
          - AMBIGUITY: ikincil minimum birincilden yeterince kötü (yan-lob/çok-yollu belirsizliği yok),
          - parabol eğriliği pozitif ve σ tavana takılı değil.
        quality_ok=False ise çağıran (DF motoru) centroid'e DÜŞER (uzman P0: gerçek fallback).

        Dönüş: dict {bearing_deg, sigma_deg, quality_ok, n, coverage_deg, front_back_db, cost_min,
        ambiguity_z} veya None (pattern yok / frekans aralık dışı / yetersiz örnek).
        """
        if not self.available():
            return None
        az = np.asarray(meas_az_deg, dtype=float) % 360.0
        pm = np.asarray(meas_power_db, dtype=float)
        n = len(az)
        if n < MIN_MATCH_SAMPLES or n != len(pm):
            return None
        got = self.pattern_at(freq_hz)
        if got is None:                                        # frekans kalibrasyon aralığı dışında
            return None
        ang, pref, fb, pk = got
        cov = self.coverage_deg(az)
        pm = pm - np.max(pm)                                   # ölçüleni tepeye normalize (pref zaten normalize)
        # Boresight-çerçevesi referans: fonksiyon x(derece) -> dB, dairesel interpolasyonla.
        ang_ext = np.concatenate([ang - 360.0, ang, ang + 360.0])
        pref_ext = np.concatenate([pref, pref, pref])

        def pref_bore(x_deg):
            # x = boresight'tan açı farkı; ölçüm-çerçevesinde tepe φ_peak(pk)'te -> kaydır
            return np.interp((np.asarray(x_deg) + pk) % 360.0, ang_ext, pref_ext)

        betas = np.arange(0.0, 360.0, 1.0)                    # 1° arama (tam tur)
        full_circle = arc_center is None
        min_cov = MIN_COVERAGE_DEG
        if not full_circle:
            # İLERİ-YAY: β adaylarını [merkez±yarı] ile sınırla; kapsama şartını yay genişliğine göre gevşet.
            d = np.abs(((betas - float(arc_center) + 180.0) % 360.0) - 180.0)
            betas = betas[d <= float(arc_half)]
            if len(betas) < 5:
                return None                                    # yay çok dar / geçersiz
            min_cov = min(MIN_COVERAGE_DEG, 0.7 * 2.0 * float(arc_half))
        cost = np.array([float(np.dot(pm - pref_bore(b - az), pm - pref_bore(b - az))) for b in betas])
        kmin = int(np.argmin(cost))
        beta = float(betas[kmin])
        cmin = float(cost[kmin])

        # AMBIGUITY: birincil minimumdan ±guard uzaktaki EN İYİ (ikincil) maliyet. Guard AÇISAL farkla
        # hesaplanır (hem tam-tur hem yay-alt-kümesi için doğru; yayda arka lob zaten dışarıda kalır).
        angd = np.abs(((betas - betas[kmin] + 180.0) % 360.0) - 180.0)
        outside = angd > AMBIGUITY_GUARD_DEG
        second = float(np.min(cost[outside])) if np.any(outside) else float(cost.max())
        # Gürültü-farkında ayrışma z-skoru (büyük=net tek çözüm, küçük=flip riski)
        ambiguity_z = (second / max(cmin, 1e-9) - 1.0) * np.sqrt(n / 2.0)

        # Tepe civarı 3 nokta ile parabol -> hassas β + eğrilik a. Tam-turda kenar SARILIR; yayda
        # kenardaki minimum için sarmadan kaçın (yanlış komşu almamak için parabolü atla).
        nb = len(betas)
        if full_circle:
            k0, k2 = (kmin - 1) % nb, (kmin + 1) % nb
        else:
            k0, k2 = kmin - 1, kmin + 1
        if 0 <= k0 and k2 < nb:
            c0, c1, c2 = cost[k0], cost[kmin], cost[k2]
            a = 0.5 * (c0 + c2 - 2.0 * c1)                    # parabol eğrilik (>0 tepe civarı)
        else:
            a = 0.0                                            # yay kenarı -> alt-derece düzeltme yok
        if a > 1e-9:
            delta = 0.5 * (c0 - c2) / (c0 - 2.0 * c1 + c2)    # [-0.5,0.5] alt-derece düzeltme
            beta = (beta + delta) % 360.0
            sigma = float(np.sqrt(max(cmin, 1e-9) / max(n - 1, 1) / a))
        else:
            sigma = SIGMA_CEIL_DEG
        sigma = float(np.clip(sigma, SIGMA_FLOOR_DEG, SIGMA_CEIL_DEG))

        quality_ok = ((fb >= MIN_FRONT_BACK_DB) and (cov >= min_cov)
                      and (ambiguity_z >= MIN_AMBIGUITY_Z) and (a > 1e-9)
                      and (sigma < SIGMA_CEIL_DEG))
        return {
            "bearing_deg": round(beta, 2),
            "sigma_deg": round(sigma, 2),
            "quality_ok": bool(quality_ok),
            "n": n,
            "coverage_deg": round(cov, 1),
            "front_back_db": round(fb, 1),
            "cost_min": round(cmin, 4),
            "ambiguity_z": round(ambiguity_z, 2),
        }


# Modül düzeyinde tek örnek (tekrar tekrar JSON okumamak için). DF motoru bunu paylaşır.
_SHARED = None


def shared_pattern() -> AntennaPattern:
    global _SHARED
    if _SHARED is None:
        _SHARED = AntennaPattern()
    return _SHARED
