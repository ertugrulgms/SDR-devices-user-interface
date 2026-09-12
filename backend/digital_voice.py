"""Sayısal amatör telsiz dinleme (spec 5.1.3): "Sayısal amatör telsizin dinlenebilmesi beklenir".

İki parça:
  1) FourFSKDetector — FM ayrımlayıcı çıkışından 4 seviyeli FSK (C4FM: DMR, YSF, P25 Faz-1, NXDN'in
     temelini oluşturan modülasyon) tespiti + sembol hızı kestirimi. Tamamen yerel DSP, HARİCİ
     araç gerekmez, her zaman çalışır. Sesi çözmez ama "sayısal (C4FM/4FSK) sinyal var, ~X baud"
     bilgisini gerçek veriden üretir.

  2) DSDFMEDecoder — sesin GERÇEKTEN çözülmesi (AMBE/mbe vocoder) için HARİCİ DSD-FME aracına köprü.
     FM ayrımlayıcı sesini (48 kHz, s16le mono) DSD-FME'nin stdin'ine borular; DSD-FME çözülmüş sesi
     kendi çıkışına (hoparlör) verir. DSD-FME kurulu değilse available()=False döner ve HİÇBİR sahte
     ses üretilmez — kullanıcıya kurulum yönlendirmesi gösterilir.

Neden harici araç: AMBE/IMBE vocoder'ları patentlidir; açık uygulaması mbelib (DSD-FME) ile yapılır.
Kendi içimizde vocoder barındırmak yerine, olgun ve doğru DSD-FME'ye köprü kurmak profesyonel ve
yasal olarak doğru yoldur.
"""
import os
import time
import shutil
import socket
import threading
import subprocess
import numpy as np


# DSD-FME ikili adayları (dağıtıma göre değişebilir)
_DSD_BINARIES = ("dsd-fme", "dsd")


def find_dsd_binary():
    """Sistemde kurulu bir DSD/DSD-FME ikilisi bulur; yoksa None."""
    for name in _DSD_BINARIES:
        path = shutil.which(name)
        if path:
            return path
    return None


class FourFSKDetector:
    """FM ayrımlayıcı çıkışından 4 seviyeli FSK (C4FM) tespiti + sembol hızı kestirimi.

    C4FM: 4 simetrik sapma seviyesi (±1, ±3 birim). Ayrımlayıcı çıkışının histogramı 4 tepe
    gösterir ve seviyeler simetriktir. Bu, analog FM'den (sürekli, tek-modlu histogram) ayırt eder.
    """

    def __init__(self, audio_rate: float = 48000.0):
        self.audio_rate = float(audio_rate)

    def detect(self, disc_audio: np.ndarray) -> dict:
        """disc_audio: FM ayrımlayıcı (anlık frekans) örnekleri. Döner: tespit özeti."""
        x = np.asarray(disc_audio, dtype=np.float64)
        if len(x) < 256:
            return {"is_4fsk": False, "reason": "yetersiz örnek"}
        x = x - np.mean(x)
        s = np.std(x)
        if s < 1e-9:
            return {"is_4fsk": False, "reason": "sinyal yok"}
        xn = x / s

        # 4 seviyeli nicemleme: k-means benzeri 4 merkez (±a, ±b). Simetri varsayımıyla
        # merkezleri veriden kestir: pozitif/negatif tarafın iki kümesi.
        centers = self._estimate_4_levels(xn)
        # Her örneği en yakın merkeze ata, kuantalama hatasını ölç
        nearest = np.min(np.abs(xn[:, None] - centers[None, :]), axis=1)
        resid = float(np.mean(nearest ** 2))
        c = np.sort(centers)
        symmetric = abs(c[0] + c[3]) < 0.4 and abs(c[1] + c[2]) < 0.4
        spacing = np.diff(c)
        even_spacing = np.std(spacing) / (np.mean(spacing) + 1e-9) < 0.5

        # KRİTİK AYIRT EDİCİ — DWELL (seviyede bekleme): C4FM çıkışı basamaklıdır, örneklerin çoğu
        # 4 ayrık seviyenin ~±0.25 komşuluğunda BEKLER (yalnızca geçiş örnekleri seviye dışıdır).
        # Analog FM ise seviyeler ARASINDA sürekli gezer -> dwell oranı düşüktür. Bu, analog iki-tonlu
        # sinyalin de düşük kuantalama kalıntısı verip yanlış-pozitif üretmesini engeller.
        level_spacing = float(np.mean(spacing)) if len(spacing) else 1.0
        dwell_tol = 0.25 * level_spacing
        dwell_frac = float(np.mean(nearest < dwell_tol))

        # TONLUK (tonality): saf sinüs (analog FM'de tek ses tonu) tek FFT binine yığılır; C4FM ise
        # rastgele veri taşıdığı için enerji geniş banda yayılır. Tepe-bin oranı yüksekse (tonal)
        # bu 4FSK DEĞİLDİR -> tek-tonlu analog FM yanlış-pozitifini engeller.
        peak_bin_ratio = self._peak_bin_ratio(xn)

        is_4fsk = bool(resid < 0.15 and symmetric and even_spacing
                       and dwell_frac >= 0.6 and peak_bin_ratio < 0.05)

        result = {
            "is_4fsk": is_4fsk,
            "levels": [round(float(v), 3) for v in c],
            "quant_residual": round(float(resid), 4),
            "dwell_frac": round(dwell_frac, 3),
            "peak_bin_ratio": round(peak_bin_ratio, 4),
            "symmetric": bool(symmetric),
        }
        if is_4fsk:
            result["symbol_rate_hz"] = round(self._estimate_symbol_rate(xn), 1)
        return result

    def _peak_bin_ratio(self, xn: np.ndarray) -> float:
        """En güçlü FFT bininin toplam AC enerjisine oranı (tonluk ölçüsü). DC hariç."""
        w = np.hanning(len(xn))
        P = np.abs(np.fft.rfft((xn - np.mean(xn)) * w)) ** 2
        P = P[1:]
        tot = float(np.sum(P))
        if tot <= 0:
            return 1.0
        return float(np.max(P) / tot)

    def _estimate_4_levels(self, xn: np.ndarray, iters: int = 12) -> np.ndarray:
        """1D 4-merkezli k-means (küçük, deterministik)."""
        c = np.array([-3.0, -1.0, 1.0, 3.0]) / 2.0  # başlangıç merkezleri (normalize)
        for _ in range(iters):
            idx = np.argmin(np.abs(xn[:, None] - c[None, :]), axis=1)
            newc = c.copy()
            for k in range(4):
                m = xn[idx == k]
                if len(m) > 0:
                    newc[k] = np.mean(m)
            if np.max(np.abs(newc - c)) < 1e-4:
                c = newc
                break
            c = newc
        return c

    def _estimate_symbol_rate(self, xn: np.ndarray) -> float:
        """Sembol geçişleri (seviye değişimleri) arasından baud kestirir."""
        # Seviye indeksleri
        c = self._estimate_4_levels(xn)
        idx = np.argmin(np.abs(xn[:, None] - c[None, :]), axis=1)
        transitions = np.where(np.diff(idx) != 0)[0]
        if len(transitions) < 4:
            return 0.0
        gaps = np.diff(transitions)
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            return 0.0
        # En sık (mod benzeri) minimum geçiş aralığı ~ bir sembol süresi
        sym_samples = np.median(gaps)
        if sym_samples <= 0:
            return 0.0
        return self.audio_rate / sym_samples


def _kmeans_1d_symmetric(x: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    """1B k-ortalama (küçük, deterministik). Merkezleri yüzdeliklerle başlatır (simetrik yayılım)."""
    c = np.percentile(x, np.linspace(8, 92, k))
    for _ in range(iters):
        idx = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        newc = c.copy()
        for j in range(k):
            m = x[idx == j]
            if len(m) > 0:
                newc[j] = np.mean(m)
        if np.max(np.abs(newc - c)) < 1e-4:
            c = newc
            break
        c = newc
    return np.sort(c)


def _estimate_baud(xn: np.ndarray, centers: np.ndarray, rate: float) -> float:
    """Seviye geçişleri arasından sembol (baud) hızı kestirir."""
    idx = np.argmin(np.abs(xn[:, None] - centers[None, :]), axis=1)
    tr = np.where(np.diff(idx) != 0)[0]
    if len(tr) < 4:
        return 0.0
    gaps = np.diff(tr)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0.0
    return rate / float(np.median(gaps))


def classify_fm_or_fsk(disc_audio: np.ndarray, audio_rate: float) -> dict:
    """FM AYRIMLAYICI çıkışından ANALOG FM ile SAYISAL FSK (2 veya 4 seviye) ayrımı (5.1.2).

    Fizik: sayısal FSK'nın anlık frekansı AYRIK seviyelerde (2 veya 4) BEKLER (basamaklı);
    analog FM'inki SÜREKLİ değişir (ses). Tek spektrum bloğunda güvenilmez olan bu ayrım, tam ses
    hızındaki ayrımlayıcı üzerinde (sembol yapısı bozulmadan) GÜVENİLİR yapılır.

    Dönüş: {'is_digital', 'levels', 'symbol_rate_hz', 'confidence', 'reason'}. Kararsızsa
    is_digital=None (dürüstçe 'belirsiz' -> üst katman 'Ortak' bırakır)."""
    x = np.asarray(disc_audio, dtype=np.float64)
    if len(x) < 512:
        return {"is_digital": None, "levels": 0, "reason": "yetersiz örnek"}
    x = x - np.median(x)
    s = float(np.std(x))
    if s < 1e-9:
        return {"is_digital": None, "levels": 0, "reason": "sinyal yok"}
    xn = x / s

    # Tonluk: tek-tonlu analog FM (tek ses tonu) tek FFT binine yığılır -> ANALOG (FSK değil).
    w = np.hanning(len(xn))
    P = np.abs(np.fft.rfft((xn - np.mean(xn)) * w)) ** 2
    P = P[1:]
    peak_ratio = float(np.max(P) / (np.sum(P) + 1e-12)) if np.sum(P) > 0 else 1.0
    if peak_ratio > 0.05:
        return {"is_digital": False, "levels": 1, "confidence": 0.8,
                "reason": f"tonal (analog FM, tek ton) peak_ratio={peak_ratio:.3f}"}

    # 4 ve 2 seviyeli ayrık yapı ara (C4FM=4, 2FSK=2). Basamaklı + simetrik + eşit aralıklı + seviyede
    # bekleme (dwell) -> SAYISAL. Aksi halde sürekli -> ANALOG.
    for k in (4, 2):
        c = _kmeans_1d_symmetric(xn, k)
        nearest = np.min(np.abs(xn[:, None] - c[None, :]), axis=1)
        resid = float(np.mean(nearest ** 2))
        spacing = np.diff(c)
        if len(spacing) == 0 or np.mean(spacing) < 1e-6:
            continue
        even = bool(np.std(spacing) / (np.mean(spacing) + 1e-9) < 0.5)
        symmetric = bool(abs(c[0] + c[-1]) < 0.4)
        dwell = float(np.mean(nearest < 0.25 * float(np.mean(spacing))))
        if resid < 0.12 and dwell >= 0.6 and even and symmetric:
            baud = _estimate_baud(xn, c, audio_rate)
            return {"is_digital": True, "levels": k, "symbol_rate_hz": round(baud, 1),
                    "confidence": round(min(1.0, 0.6 + (0.6 - resid)), 2),
                    "reason": f"{k} ayrık seviye (dwell={dwell:.2f}, resid={resid:.3f})"}

    # Ayrık seviye yok -> sürekli IF -> ANALOG FM (ses)
    return {"is_digital": False, "levels": 0, "confidence": 0.7,
            "reason": "sürekli IF (ayrık seviye yok)"}


class DSDFMEDecoder:
    """Harici DSD-FME'ye TCP KÖPRÜSÜ: FM ayrımlayıcı sesini (48k s16le mono) bir TCP soketinden akıtır;
    DSD-FME bu sokete client olarak bağlanıp (SDR++ standardı, port 7355) sesi çözer ve PulseAudio'ya verir.

    NEDEN TCP (stdin değil): lwvmobile/dsd-fme sürümü stdin (`-i -`) KABUL ETMEZ; girişi pulse/wav/tcp/rtl'dir.
    `-i tcp` seçeneği tam da SDR++/GNURadio'nun 48k/1 s16le ses akışını okumak içindir -> bize birebir uyar.

    Kurulu değilse hiçbir şey yapmaz (available()=False). Sentetik ses YOK.
    """

    TCP_HOST = "127.0.0.1"
    TCP_PORT = 7355                     # DSD-FME '-i tcp' varsayılan portu (SDR++ uyumlu)

    def __init__(self, audio_rate: int = 48000):
        self.audio_rate = int(audio_rate)
        self.binary = find_dsd_binary()
        self.proc = None
        self.last_error = None
        self._srv = None               # dinleyen soket (server=biz)
        self._conn = None              # kabul edilen client (dsd-fme)
        self._accept_thread = None

    def available(self) -> bool:
        return self.binary is not None

    def install_hint(self) -> str:
        return ("DSD-FME kurulu değil. Sayısal ses (DMR/YSF/P25/NXDN) çözümü için: "
                "tools/install_dsd_fme.sh çalıştırın (mbelib + dsd-fme derler).")

    def start(self) -> bool:
        """TCP server aç (7355) -> dsd-fme'yi '-i tcp -o pulse' ile başlat (client olarak bağlanır)."""
        if not self.available():
            self.last_error = "DSD-FME bulunamadı"
            return False
        if self.proc is not None:
            return True

        # 1) TCP server'ı KUR (dsd-fme başlamadan ÖNCE dinlemede olmalı; o bize bağlanacak)
        try:
            self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv.bind((self.TCP_HOST, self.TCP_PORT))
            self._srv.listen(1)
            self._srv.settimeout(8.0)
        except OSError as e:
            self.last_error = f"TCP {self.TCP_PORT} açılamadı (başka bir DSD/SDR++ mı kullanıyor?): {e}"
            self._cleanup_sockets()
            return False

        # 2) Bağlantıyı arka planda bekle (dsd-fme birazdan client olarak bağlanır)
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        # 3) dsd-fme'yi başlat. Sürüm farkı olursa DSD_FME_CMD ile komutu değiştir
        #    (ör: export DSD_FME_CMD="dsd-fme -fa -i tcp:127.0.0.1:7355 -o pulse").
        override = os.environ.get("DSD_FME_CMD")
        cmd = override.split() if override else [self.binary, "-fa", "-i", "tcp", "-o", "pulse"]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except Exception as e:
            self.last_error = f"DSD-FME başlatılamadı: {e}"
            self._cleanup_sockets()
            self.proc = None
            return False

        # 4) HEMEN KAPANDI MI? (yanlış bayrak/sürüm) -> dürüst hata
        time.sleep(0.6)
        if self.proc.poll() is not None:
            self.last_error = ("DSD-FME hemen kapandı (komut/sürüm uyumsuz olabilir). Terminalde elle dene: "
                               + " ".join(cmd) + "   |   ya da DSD_FME_CMD ile komutu ayarla.")
            self._cleanup_sockets()
            self.proc = None
            return False
        return True

    def _accept_loop(self):
        """dsd-fme'nin TCP bağlantısını kabul et (tek client)."""
        try:
            conn, _ = self._srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._conn = conn
        except (OSError, socket.timeout):
            pass                         # bağlanamadıysa feed() sessizce atlar (dürüst)

    def feed(self, disc_audio: np.ndarray):
        """FM ayrımlayıcı sesini (float [-1,1]) 48k s16le'ye çevirip TCP client'a (dsd-fme) gönderir."""
        conn = self._conn
        if conn is None:
            return                       # dsd-fme henüz bağlanmadı (birkaç yüz ms sürebilir)
        pcm = np.clip(np.asarray(disc_audio) * 32767.0, -32768, 32767).astype("<i2")
        try:
            conn.sendall(pcm.tobytes())
        except OSError as e:
            self.last_error = f"TCP besleme hatası: {e}"
            self.stop()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _cleanup_sockets(self):
        for s in (self._conn, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self._conn = None
        self._srv = None

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        self._cleanup_sockets()
