"""Parametre Konsolidatörü — 5.1.2 parametrelerini ZAMAN PENCERESİNDE birleştirir (stabilizasyon).

Sorun: Sınıflandırıcı ~2 Hz çalışır; her kare bağımsız/gürültülü tahmindir -> ekranda titrer, hakem
"kesin bir çıktı" okuyamaz. Bu birim son `window_sec` saniyedeki GERÇEK ölçümleri biriktirip:
  * Kategorik alanlar (modülasyon, çoklama, EKKT, protokol, analog/sayısal):
      ÇOĞUNLUK OYU + HİSTEREZİS -> en çok desteklenen etiket, kararlı. Uyum oranı = güven.
      Yer-tutucu değerler ("Belirlenemedi/Ölçülüyor/Sinyal yok") oya SAYILMAZ: pencerede EN AZ BİR
      gerçek sınıflandırma varsa onu gösterir -> sinyal varken ASLA boş kalmaz (en olası tahmin).
  * Sayısal alanlar (taşıyıcı, bant genişliği, sembol hızı): MEDYAN -> kararlı sayı.

DÜRÜSTLÜK: Sahte üretmez. Gerçek, tekrarlı ölçümleri istatistiksel birleştirir (tek gürültülü kareden
DAHA DOĞRU). Güven düşükse "zayıf/olası" etiketiyle bildirir; tüm pencere gürültü (gerçek sınıf yok)
ise dürüstçe "ölçülüyor" der.
"""
from collections import deque, Counter
import numpy as np

# Oya sayılmayan yer-tutucu (gerçek bir sınıflandırma DEĞİL) değerler — SUBSTRING olarak aranır.
# NOT: "-" ve "" burada YOK; onlar TAM EŞLEŞME yer-tutucudur (aksi halde "Analog-Frekans"daki tire
# gerçek değeri yer-tutucu sanardı).
_PLACEHOLDER_HINTS = ("Belirlenemedi", "Ölçülüyor", "Sinyal yok", "Bekleniyor", "Durduruldu")


def _is_real(value) -> bool:
    if value is None:
        return False
    s = str(value).strip()
    if s in ("", "-", "—"):
        return False
    return not any(h in s for h in _PLACEHOLDER_HINTS)


def _is_num(v) -> bool:
    try:
        f = float(v)
        return np.isfinite(f)
    except (TypeError, ValueError):
        return False


class ParameterConsolidator:
    CATEGORICAL = ("modulation", "analog_digital", "multiplex", "ekkt", "protocol")
    NUMERIC = ("carrier_mhz", "occupied_bw_hz", "symbol_rate_hz")

    def __init__(self, window_sec: float = 3.0, min_samples: int = 2, switch_margin: float = 0.15):
        self.window_sec = float(window_sec)
        self.min_samples = int(min_samples)
        self.switch_margin = float(switch_margin)   # histerezis: yeni değer eskiyi bu farkla geçmeli
        self._hist = deque()                          # (t, dict)
        self._locked = {}                             # kategorik alan -> son gösterilen (kararlı) değer

    def reset(self):
        """Yeni kaynak (retune) veya sinyal kaybı -> geçmişi temizle (eski değer yeni kaynağa taşınmasın)."""
        self._hist.clear()
        self._locked.clear()

    def update(self, result: dict, now: float):
        """Bir sınıflandırma karesini ekle; pencere dışını at."""
        self._hist.append((float(now), dict(result or {})))
        while self._hist and (now - self._hist[0][0]) > self.window_sec:
            self._hist.popleft()

    @staticmethod
    def tier(agree: float) -> str:
        """Uyum oranından güven etiketi."""
        if agree >= 0.70:
            return "KESİN"
        if agree >= 0.40:
            return "olası"
        return "zayıf"

    def _real_values(self, field):
        return [d.get(field) for (_, d) in self._hist if _is_real(d.get(field))]

    def consolidated(self) -> dict:
        """Konsolide parametreler. Kategorik -> {value, confidence, tier}; sayısal -> {value, n}."""
        out = {"_samples": len(self._hist)}

        for f in self.CATEGORICAL:
            vals = self._real_values(f)
            if len(vals) < 1:
                continue                                  # gerçek sınıf yok -> alan atlanır (ölçülüyor)
            cnt = Counter(vals)
            top, top_n = cnt.most_common(1)[0]
            agree = top_n / len(vals)
            # HİSTEREZİS: kilitli (önceki) değer hâlâ yarışıyorsa, yeni aday onu switch_margin ile GEÇMELİ
            cur = self._locked.get(f)
            if cur is not None and cur != top and cur in cnt:
                cur_frac = cnt[cur] / len(vals)
                if (agree - cur_frac) < self.switch_margin:
                    top, agree = cur, cur_frac             # eskiyi koru (50/50 zıplamayı önle)
            self._locked[f] = top
            out[f] = {"value": top, "confidence": round(agree, 2), "tier": self.tier(agree),
                      "n": len(vals)}

        for f in self.NUMERIC:
            nums = [float(v) for (_, d) in self._hist if _is_num(d.get(f)) and float(d.get(f)) != 0.0
                    for v in [d.get(f)]]
            if nums:
                out[f] = {"value": float(np.median(nums)), "n": len(nums)}
        return out

    def label(self, field: str) -> str:
        """Bir kategorik alanın gösterime hazır etiketi: 'FM (Analog-Frekans) [KESİN %90]' gibi.
        Gerçek değer yoksa 'ölçülüyor...' (sinyal var ama henüz oturmadı / gürültü)."""
        c = self.consolidated().get(field)
        if not c or "value" not in c:
            return "ölçülüyor..."
        return f"{c['value']}  [{c['tier']} %{int(c['confidence'] * 100)}]"
