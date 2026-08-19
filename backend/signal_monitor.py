"""Sinyal İzleme/Takip (spec 5.1.3): bir sinyale 'kilitlenip' zaman içindeki davranışını izler.

İki eksen:
  * Süreklilik (continuity): sinyal var/yok geçişleri, kesinti (drop) ve yeniden yakalama
    (reacquire) olayları, toplam görülme oranı (uptime %). "Sinyal sürekli mi, kesikli mi
    (bursty), atlıyor mu?" sorusuna cevap verir.
  * Parametre geçmişi: taşıyıcı frekansı / güç / bant genişliği zaman serisi -> sapma (drift)
    ve kararlılık ölçümü.

Tamamen deterministik ve donanımsız test edilebilir. Sentetik sinyal ÜRETMEZ; yalnızca dışarıdan
verilen gerçek ölçümleri (analiz sonuçlarını) biriktirip özetler.
"""
import time
from collections import deque


class SignalMonitor:
    def __init__(self, history_seconds: float = 120.0, drop_grace_sec: float = 1.5):
        # history_seconds: parametre geçmişinin tutulacağı pencere.
        # drop_grace_sec: sinyal bu süre boyunca yoksa "kesinti (drop)" say (kısa sönümlemeler
        # her karede false-drop üretmesin).
        self.history_seconds = float(history_seconds)
        self.drop_grace_sec = float(drop_grace_sec)
        self._reset()

    def _reset(self):
        self.locked = False
        self.locked_freq_mhz = None
        self.history = deque()          # (t, present, carrier_mhz, power_dbm, bw_hz, snr_db)
        self.first_seen = None
        self.last_present_t = None      # en son "var" olduğu an
        self.last_update_t = None
        self.present = False            # anlık (grace uygulanmış) durum
        self._raw_present = False       # ham (grace öncesi) durum
        self.present_accum = 0.0        # toplam "var" süresi
        self.total_accum = 0.0          # toplam izleme süresi
        self.drop_count = 0
        self.reacquire_count = 0

    # ------------------------------------------------------------------ kontrol
    def lock(self, freq_mhz: float):
        """Belirtilen frekansa kilitlen ve izlemeyi (sıfırdan) başlat."""
        self._reset()
        self.locked = True
        self.locked_freq_mhz = float(freq_mhz)

    def unlock(self):
        self._reset()

    def is_locked(self) -> bool:
        return self.locked

    # ------------------------------------------------------------------ besleme
    def update(self, now: float, present: bool, carrier_mhz: float = None,
               power_dbm: float = None, bw_hz: float = None, snr_db: float = None):
        """Bir analiz döngüsünün sonucunu işler. Döner: olay etiketi veya None.
        Olaylar: "reacquire" (kayıp sonrası geri geldi), "drop" (grace süresini aşan kayıp)."""
        if not self.locked:
            return None
        event = None
        if self.first_seen is None:
            self.first_seen = now
            self.last_update_t = now

        # süre biriktirme
        dt = max(0.0, now - (self.last_update_t if self.last_update_t is not None else now))
        self.total_accum += dt
        if self._raw_present:
            self.present_accum += dt
        self.last_update_t = now

        was_present = self.present
        self._raw_present = bool(present)

        if present:
            self.last_present_t = now
            if not was_present:
                # kayıptan geri döndü mü? (daha önce en az bir kez görülmüşse)
                if self.drop_count > 0 or (self.last_present_t and self.present_accum > 0):
                    self.reacquire_count += 1
                    event = "reacquire"
            self.present = True
        else:
            # grace: son görülmeden bu yana geçen süre eşiği aşarsa "drop"
            if was_present and self.last_present_t is not None:
                if (now - self.last_present_t) >= self.drop_grace_sec:
                    self.present = False
                    self.drop_count += 1
                    event = "drop"

        # geçmişe kaydet
        self.history.append((now, bool(present), carrier_mhz, power_dbm, bw_hz, snr_db))
        cutoff = now - self.history_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        return event

    # ------------------------------------------------------------------ özet
    def status(self, now: float = None) -> dict:
        """İzleme durumunun özetini döner (UI/log için)."""
        if now is None:
            now = time.time()
        if not self.locked:
            return {"locked": False}

        carriers = [h[2] for h in self.history if h[1] and h[2] is not None]
        powers = [h[3] for h in self.history if h[1] and h[3] is not None]
        bws = [h[4] for h in self.history if h[1] and h[4] is not None]

        uptime = (self.present_accum / self.total_accum * 100.0) if self.total_accum > 0 else 0.0
        last_seen_ago = (now - self.last_present_t) if self.last_present_t is not None else None
        duration = (now - self.first_seen) if self.first_seen is not None else 0.0

        # süreklilik sınıfı
        if uptime >= 95.0:
            continuity = "Sürekli"
        elif uptime >= 40.0:
            continuity = "Kesikli (bursty)"
        elif uptime > 0.0:
            continuity = "Aralıklı/zayıf"
        else:
            continuity = "Yok"

        freq_drift_khz = ((max(carriers) - min(carriers)) * 1e3) if len(carriers) >= 2 else 0.0
        power_range_db = (max(powers) - min(powers)) if len(powers) >= 2 else 0.0
        bw_mean_hz = (sum(bws) / len(bws)) if bws else 0.0

        return {
            "locked": True,
            "freq_mhz": self.locked_freq_mhz,
            "present": self.present,
            "continuity": continuity,
            "uptime_pct": round(uptime, 1),
            "drops": self.drop_count,
            "reacquires": self.reacquire_count,
            "freq_drift_khz": round(freq_drift_khz, 3),
            "power_range_db": round(power_range_db, 1),
            "bw_mean_hz": round(bw_mean_hz, 1),
            "last_seen_ago_s": round(last_seen_ago, 1) if last_seen_ago is not None else None,
            "duration_s": round(duration, 1),
            "samples": len(self.history),
        }

    def summary_text(self, now: float = None) -> str:
        """Tek satır insan-okur özet (UI için)."""
        s = self.status(now)
        if not s.get("locked"):
            return "İzleme kapalı"
        state = "VAR" if s["present"] else "YOK"
        return (f"{s['freq_mhz']:.3f} MHz | {s['continuity']} "
                f"(%{s['uptime_pct']:.0f} uptime, {state}) | "
                f"kesinti:{s['drops']} geri:{s['reacquires']} | "
                f"kayma:{s['freq_drift_khz']:.1f} kHz")
