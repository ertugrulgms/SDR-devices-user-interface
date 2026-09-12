"""Gerçek zamanlı ses oynatıcı: SDR ring buffer'ından kesintisiz I/Q çeker, StreamingDemodulator
ile sese çevirir ve hoparlöre (sounddevice) verir.

Mimari — iki iş parçacığı, birbirinden bağımsız:
  * Üretici (producer) thread: engine.get_stream_new() ile YENİ I/Q örneklerini çeker (C++ tarafı
    ayrı bir okuma imleci tutar, analizle çakışmaz), demodüle eder ve ses kuyruğuna yazar.
  * sounddevice OutputStream callback'i: kuyruktan tüketip hoparlöre basar; veri yoksa sessizlik.

sounddevice bulunmayan veya ses aygıtı olmayan ortamlarda (headless test) nesne yine kurulur ama
start() False döner; UI bunu "ses aygıtı yok" olarak gösterebilir. Sentetik ses YOK — veri yalnızca
gerçek SDR akışından gelir.
"""
import threading
import numpy as np

from backend.audio_demodulator import StreamingDemodulator
from backend.digital_voice import FourFSKDetector, DSDFMEDecoder

try:
    import sounddevice as _sd
except Exception:  # pragma: no cover - ortam bağımlı
    _sd = None


class AudioPlayer:
    # Üretici döngüsünde her turda çekilecek azami I/Q örnek sayısı (~20 ms @ 2.4 MHz).
    _PULL_MAX = 1 << 16
    # Ses kuyruğunda tutulacak azami örnek (taşarsa en eskiyi at — düşük gecikme).
    _MAX_QUEUE = 48000 * 2  # ~2 sn

    def __init__(self, engine, sample_rate: float, mode: str = "FM"):
        self.engine = engine
        self.sample_rate = float(sample_rate)
        self.demod = StreamingDemodulator(sample_rate, mode=mode)
        self.out_rate = int(round(self.demod.out_rate))
        self._queue = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._muted = True          # başlangıçta sustur (kullanıcı "Dinle" deyince açılır)
        self._running = False
        self._stream = None
        self._thread = None
        self._underruns = 0
        self.last_error = None

        # Sayısal ses (spec 5.1.3): ham FM ayrımlayıcı -> 4FSK tespiti + (varsa) DSD-FME'ye köprü.
        self.digital_enabled = False
        self.digital_demod = StreamingDemodulator(sample_rate, mode="FM", digital=True,
                                                  channel_bw=12500.0)
        self.fsk = FourFSKDetector(self.out_rate)
        self.dsd = None
        self.last_fsk = {}

    # ------------------------------------------------------------- durum/kontrol
    def available(self) -> bool:
        """sounddevice ve bir çıkış aygıtı var mı?"""
        if _sd is None:
            return False
        try:
            _sd.query_devices(kind="output")
            return True
        except Exception:
            return False

    def set_mode(self, mode: str):
        with self._lock:
            self.demod.set_mode(mode)

    def set_tuning_offset(self, offset_hz: float):
        """OTOMATİK MERKEZLEME: hedef sinyalin merkez-frekanstan sapması (Hz). Operatör tam tune
        etmese bile demod sinyali DC'ye çeker -> gürültü yerine temiz ses. Worker ölçülen tepe-offseti
        ile günceller. (Aksi halde merkez-dışı sinyal kanal filtresince süzülür = gürültü.)"""
        self.demod.set_tuning_offset(offset_hz)
        self.digital_demod.set_tuning_offset(offset_hz)

    @property
    def mode(self) -> str:
        return self.demod.mode

    def set_muted(self, muted: bool):
        self._muted = bool(muted)

    def set_digital(self, enabled: bool, key: str = None) -> dict:
        """Sayısal çözmeyi aç/kapat. key verilirse (ondalık DMR Basic Privacy anahtarı) şifreli yayın
        çözülür. Döner: {'enabled', 'dsd_available', 'hint'}. DSD-FME kuruluysa çözülmüş sesi kendisi
        oynatır; kurulu değilse yalnızca 4FSK tespiti çalışır (sahte ses ÜRETİLMEZ)."""
        self.digital_enabled = bool(enabled)
        info = {"enabled": self.digital_enabled, "dsd_available": False, "hint": ""}
        if enabled:
            if self.dsd is None:
                self.dsd = DSDFMEDecoder(self.out_rate)
            info["dsd_available"] = self.dsd.available()
            if self.dsd.available():
                if not self.dsd.is_running():
                    if not self.dsd.start(bp_key=key):
                        info["hint"] = self.dsd.last_error or "DSD-FME başlatılamadı"
            else:
                info["hint"] = self.dsd.install_hint()
            self.digital_demod.reset()
        else:
            if self.dsd is not None and self.dsd.is_running():
                self.dsd.stop()
        return info

    def get_last_fsk(self) -> dict:
        return dict(self.last_fsk)

    def get_digital_data(self, n: int = 6):
        """DSD-FME'den ÇÖZÜLEN sayısal veri satırları (sync/renk kodu/çağrı/TG). Yoksa []
        (5.1.3 'ses ve/veya veriye ulaşılması' — veri kısmı)."""
        if self.dsd is not None and self.dsd.is_running():
            return self.dsd.get_recent_data(n)
        return []

    def is_muted(self) -> bool:
        return self._muted

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ yaşam döngüsü
    def start(self) -> bool:
        if self._running:
            return True
        if _sd is None:
            self.last_error = "sounddevice kurulu değil"
            return False
        if self.engine is None or not hasattr(self.engine, "get_stream_new"):
            self.last_error = "SDR akış API'si yok (get_stream_new)"
            return False
        try:
            if hasattr(self.engine, "reset_audio_cursor"):
                self.engine.reset_audio_cursor()
            self.demod.reset()
            with self._lock:
                self._queue = np.zeros(0, dtype=np.float32)
            self._stream = _sd.OutputStream(
                samplerate=self.out_rate, channels=1, dtype="float32",
                blocksize=1024, callback=self._sd_callback,
            )
            self._stream.start()
        except Exception as e:  # pragma: no cover - aygıt bağımlı
            self.last_error = f"ses aygıtı açılamadı: {e}"
            self._stream = None
            return False
        self._running = True
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None
        if self.dsd is not None and self.dsd.is_running():
            self.dsd.stop()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # --------------------------------------------------------------- iç işleyiş
    def _producer(self):
        """SDR'dan yeni I/Q çek -> demodüle et -> kuyruğa yaz. Kesintisiz döngü."""
        import time
        while self._running:
            iq = None
            try:
                iq = self.engine.get_stream_new(self._PULL_MAX)
            except Exception as e:
                self.last_error = f"akış çekme hatası: {e}"
            if iq is not None and len(iq) >= 2:
                iq = np.asarray(iq)
                try:
                    if self.digital_enabled:
                        # Ham FM ayrımlayıcı -> 4FSK tespiti + (varsa) DSD-FME'ye borula.
                        disc = self.digital_demod.process(iq)
                        if len(disc) >= 256:
                            try:
                                self.last_fsk = self.fsk.detect(disc)
                            except Exception:
                                pass
                        if self.dsd is not None and self.dsd.is_running():
                            self.dsd.feed(disc)
                        # Ham diskriminatörü hoparlöre BASMA (DSD kendi çözülmüş sesini oynatır).
                    else:
                        audio = self.demod.process(iq)
                        if len(audio):
                            with self._lock:
                                self._queue = np.concatenate([self._queue, audio])
                                if len(self._queue) > self._MAX_QUEUE:
                                    self._queue = self._queue[-self._MAX_QUEUE:]
                except Exception as e:
                    # KRİTİK: demod hatası ses THREAD'ini ÖLDÜRMESİN. Aksi halde tek bir hata sonrası
                    # ses KALICI kesilir (uygulama yeniden başlatılana dek). Logla, kareyi atla, sürdür.
                    self.last_error = f"demod hatası (kare atlandı): {e}"
                # Örnekleme hızına göre uygun bekleme (CPU'yu doldurma)
                time.sleep(max(0.005, len(iq) / self.sample_rate * 0.5))
            else:
                time.sleep(0.01)

    def _sd_callback(self, outdata, frames, time_info, status):  # pragma: no cover - RT
        if self._muted:
            outdata[:] = 0.0
            return
        with self._lock:
            n = min(frames, len(self._queue))
            if n > 0:
                outdata[:n, 0] = self._queue[:n]
                self._queue = self._queue[n:]
            if n < frames:
                outdata[n:, 0] = 0.0
                self._underruns += 1
