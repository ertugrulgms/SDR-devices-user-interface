"""Yön Bulma (DF) ve Konum Belirleme motoru — GENLİK TABANLI DF + 3B LOB üçgenleme.

Mimari: 3 düğüm (2 uzak PlutoSDR + 1 ana B200mini). Her düğümde açı sensörlü, ELLE döndürülen
yönlü anten (başlangıçta Kuzey'e bakar). Genlik tabanlı DF: anten döndürülürken en yüksek sinyal
genliğinin alındığı AZİMUT = kaynağın geliş yönü (kerteriz / LOB). Uzak düğümler kendi kerterizlerini
JSON ile ana cihaza gönderir; ana düğüm kendi kerterizini B200mini + enkoderden üretir. Ana cihaz
tüm kerterizleri (LOB) birleştirip kaynağın 3B konumunu üçgenlemeyle kestirir.

Koordinat çerçevesi: yerel ENU (x=Doğu, y=Kuzey, z=Yukarı), metre. Azimut Kuzey'den (0°) saat
yönünde ölçülür (90°=Doğu). Yükseklik (elevation) ufuktan yukarı (0°=yatay).
"""
import os
import json
import time
import threading
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_NODES_PATH = os.path.join(_DATA_DIR, "df_nodes.json")

# Varsayılan düğüm yerleşimi (yerel ENU metre). Sahada ölçülen gerçek konumlarla df_nodes.json'dan
# değiştirilebilir. ~500 m tabanlı bir üçgen (kabaca 10 km² saha içinde). "self": ana (yerel) düğüm.
DEFAULT_NODES = {
    "NODE-MAIN": {"pos": [0.0, 0.0, 0.0], "self": True},
    "NODE-2":    {"pos": [500.0, 0.0, 0.0], "self": False},
    "NODE-3":    {"pos": [250.0, 433.0, 0.0], "self": False},
}


# ------------------------------------------------------------------ #
#  GEOMETRİ
# ------------------------------------------------------------------ #
def azel_to_unit(azimuth_deg: float, elevation_deg: float = 0.0) -> np.ndarray:
    """Azimut (Kuzey'den saat yönü) + yükseklik -> ENU birim yön vektörü [Doğu, Kuzey, Yukarı]."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    ce = np.cos(el)
    return np.array([ce * np.sin(az), ce * np.cos(az), np.sin(el)], dtype=float)


def bearing_from_positions(observer_xyz, target_xyz):
    """İki nokta arasındaki azimut (Kuzey'den saat yönü, derece) + yatay menzil (m) + yükseklik açısı."""
    d = np.asarray(target_xyz, float) - np.asarray(observer_xyz, float)
    az = (np.degrees(np.arctan2(d[0], d[1]))) % 360.0        # atan2(Doğu, Kuzey)
    ground = float(np.hypot(d[0], d[1]))
    el = float(np.degrees(np.arctan2(d[2], ground))) if ground > 1e-9 else 0.0
    return round(az, 2), round(ground, 2), round(el, 2)


# Konum kestirimi geometrik olarak GÜVENİLİR sayılması için gereken asgari LOB kesişim açısı.
# Bunun altında ışınlar neredeyse paralel -> küçük açı hatası devasa konum hatası verir (kötü GDOP);
# "güvenli ama yanlış" konum yerine dürüstçe "zayıf geometri (fix yok)" bildirilir.
MIN_CROSSING_ANGLE_DEG = 5.0


def lob_crossing_angle_deg(directions) -> float:
    """LOB'lar arasındaki EN İYİ (en geniş) kesişim açısı, derece. 0°=paralel (kötü), 90°=dik (ideal).
    Konum kestirim geometrisinin (GDOP) kalitesini özetler."""
    dirs = [np.asarray(d, float) / (np.linalg.norm(d) + 1e-12) for d in directions]
    best = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            c = float(np.clip(abs(np.dot(dirs[i], dirs[j])), 0.0, 1.0))
            best = max(best, float(np.degrees(np.arccos(c))))   # dar açı: 0..90
    return best


def triangulate_lob(positions, directions):
    """3B LOB (kerteriz doğrusu) EN-KÜÇÜK-KARELER üçgenlemesi + GEOMETRİ (GDOP) kapısı.

    Her düğüm i, konum p_i'den d_i yönünde bir ışın (LOB) tanımlar. Işınlar gürültü yüzünden tam
    kesişmez; kaynağın konumu, TÜM ışınlara dik uzaklıkların karelerinin toplamını EN AZ yapan
    noktadır: min_x Σ |(I - d_i d_iᵀ)(x - p_i)|². Çözüm: (Σ Aᵢ) x = Σ Aᵢ pᵢ,  Aᵢ = I - dᵢ dᵢᵀ.

    Dönüş: (konum[x,y,z], ortalama_kalıntı_m, kesişim_güvenilir_mi, kesişim_açısı_derece).
    Yükseklik bilgisi yoksa (tüm LOB'lar yatay) z ekseni belirsizdir -> 2B çözülür, z=0 (havadaki
    kaynağın YER İZDÜŞÜMÜ bulunur; irtifa için elevation ölçümü gerekir). En az 2 LOB gerekir.
    GDOP kapısı: LOB'lar neredeyse paralelse (en geniş kesişim < MIN_CROSSING_ANGLE_DEG) konum
    güvenilmezdir -> fix=False (sahte-güvenli konum üretilmez).
    """
    positions = [np.asarray(p, float) for p in positions]
    directions = [np.asarray(d, float) / (np.linalg.norm(d) + 1e-12) for d in directions]
    if len(positions) < 2:
        return np.zeros(3), 0.0, False, 0.0

    cross_deg = lob_crossing_angle_deg(directions)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in zip(positions, directions):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ p

    # Yükseklik bilgisi ANLAMLI mı? (yalnızca gürültü değil, gerçek irtifa var mı)
    # Eski eşik (herhangi bir |d_z|>1e-3 ≈ 0.06°) çok hassastı: yerdeki bir kaynakta bile
    # ±1° elevation ÖLÇÜM GÜRÜLTÜSÜ (d_z~0.017) 3B çözümü tetikleyip z'ye SAHTE irtifa (metrelerce)
    # veriyordu. MEDYAN |d_z| kullan (tek gürültülü düğüme bağışık) ve ~4° eşiği aşmasını iste:
    # gürültü (±1-2°) 2B'de (z=0, yer izdüşümü) kalır; gerçek havadaki kaynak (el≳4°) 3B çözülür.
    med_dz = float(np.median([abs(d[2]) for d in directions]))
    has_elevation = med_dz > np.sin(np.radians(4.0))
    try:
        if has_elevation and abs(np.linalg.det(A)) > 1e-9:
            x = np.linalg.solve(A, b)
        else:
            # 2B çöz (x,y); z=0 -> yer düzlemi projeksiyonu
            x2 = np.linalg.lstsq(A[:2, :2], b[:2], rcond=None)[0]
            x = np.array([x2[0], x2[1], 0.0])
    except np.linalg.LinAlgError:
        return np.zeros(3), 0.0, False, round(cross_deg, 1)

    # Ortalama dik kalıntı (kesişim kalitesi göstergesi, metre)
    res = float(np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - p))
                         for p, d in zip(positions, directions)]))
    # GDOP kapısı: geometri çok zayıfsa (ışınlar ~paralel) konum güvenilmez -> fix yok.
    fix = cross_deg >= MIN_CROSSING_ANGLE_DEG
    return x, round(res, 2), fix, round(cross_deg, 1)


# ------------------------------------------------------------------ #
#  GENLİK TABANLI KERTERİZ KESTİRİCİ (yerel düğüm)
# ------------------------------------------------------------------ #
class AmplitudeDFEstimator:
    """GENLİK TABANLI DF: anten ELLE döndürülürken (azimut, genlik) örnekleri gelir; en yüksek
    genliğin azimutu = kaynağın geliş yönü. Tepe etrafında genlik-ağırlıklı merkez (centroid) ile
    alt-derece hassasiyet sağlanır. Örnekler zamanla sönümlenir (yeniden tarama / hareketli kaynak).

    Kullanım (yerel düğüm): her karede update(enkoder_azimutu, ölçülen_genlik_dBm). bearing()
    o ana kadarki taramadan en olası kerterizi verir."""

    def __init__(self, bin_deg: float = 1.0, window_deg: float = 25.0, decay_sec: float = 8.0):
        self.bin_deg = bin_deg
        self.window_deg = window_deg          # tepe etrafı centroid penceresi
        self.decay_sec = decay_sec            # bu süreden eski örnekler unutulur
        self._bins = {}                       # az_bin(int) -> (amp_dbm, ts)

    def reset(self):
        self._bins.clear()

    def update(self, azimuth_deg: float, amp_dbm: float, now: float = None):
        now = time.time() if now is None else now
        key = int(round((azimuth_deg % 360.0) / self.bin_deg)) % int(round(360.0 / self.bin_deg))
        prev = self._bins.get(key)
        # Aynı azimutta en YÜKSEK genliği tut (tarama sırasında tepe yakalanır); eski ise güncelle
        if prev is None or amp_dbm >= prev[0] or (now - prev[1]) > self.decay_sec:
            self._bins[key] = (amp_dbm, now)

    def _prune(self, now):
        dead = [k for k, (_, ts) in self._bins.items() if (now - ts) > self.decay_sec]
        for k in dead:
            del self._bins[k]

    def bearing(self, now: float = None):
        """Dönüş: (azimut_derece | None, tepe_genlik_dBm, güven[0..1], örnek_sayısı).
        Yeterli tarama yoksa azimut None döner (dürüstçe 'ölçülemedi')."""
        now = time.time() if now is None else now
        self._prune(now)
        if len(self._bins) < 3:
            return None, -120.0, 0.0, len(self._bins)

        azs = np.array([k * self.bin_deg for k in self._bins.keys()])
        amps = np.array([v[0] for v in self._bins.values()])
        peak_i = int(np.argmax(amps))
        peak_az = float(azs[peak_i])
        peak_amp = float(amps[peak_i])
        floor = float(np.min(amps))

        # Tepe etrafında (±window) genlik-ağırlıklı dairesel centroid (alt-derece hassasiyet)
        offs = ((azs - peak_az + 180.0) % 360.0) - 180.0     # tepeye göre [-180,180]
        mask = np.abs(offs) <= self.window_deg
        w = np.power(10.0, (amps[mask] - floor) / 10.0)       # dBm -> doğrusal güç ağırlığı
        centroid_off = float(np.sum(offs[mask] * w) / (np.sum(w) + 1e-12))
        bearing_deg = (peak_az + centroid_off) % 360.0

        # Güven: tepe-taban farkı ne kadar büyükse o kadar yüksek (0..1), ~20 dB'de doyar
        conf = float(min(1.0, max(0.0, (peak_amp - floor) / 20.0)))
        return round(bearing_deg, 2), round(peak_amp, 1), round(conf, 2), len(self._bins)


# ------------------------------------------------------------------ #
#  HAREKETLİ TEK ALICI ile KONUM (spec 5.1.5)
# ------------------------------------------------------------------ #
class MovingReceiverPositioner:
    """HAREKET HALİNDEKİ TEK ALICI ile konum belirleme (spec 5.1.5).

    Platform (GPS'li) hareket ederken FARKLI konumlardan kaynağa kerteriz (LOB) alınır. Bu
    (konum, yön) örnekleri biriktirilip üçgenlenir -> kaynağın konumu. Tek bir hareketli alıcı,
    çok sayıda sabit alıcının yaptığını zamanda tarayarak yapar.

    MEKÂNSAL ÇEŞİTLİLİK ŞARTI: yeni örnek yalnızca alıcı bir öncekinden `min_baseline_m` kadar
    UZAKLAŞMIŞSA eklenir; aksi halde tüm örnekler ~aynı noktadadır ve üçgenleme tabanı (baz) oluşmaz.
    Kaynak, birikim süresince sabit varsayılır (şartname: kaynaklar sırayla yayın yapar).

    Konumlar yerel ENU metre (GPS -> geo.geodetic_to_enu ile çevrilir). Sentetik veri yok:
    örnekler yalnızca gerçek GPS konumu + gerçek kerterizden gelir.
    """

    def __init__(self, min_baseline_m: float = 15.0, max_samples: int = 40,
                 decay_sec: float = 120.0):
        self.min_baseline_m = float(min_baseline_m)
        self.max_samples = int(max_samples)
        self.decay_sec = float(decay_sec)
        self._samples = []   # list of (pos[np3], direction[np3], ts)

    def reset(self):
        self._samples.clear()

    def add(self, pos_enu, bearing_deg: float, elevation_deg: float = 0.0, now: float = None) -> bool:
        """Bir (konum, kerteriz) örneği ekle. Yalnızca yeterince yer değiştirilmişse eklenir.
        Dönüş: örnek eklendi mi (mekânsal çeşitlilik sağlandı mı)."""
        now = time.time() if now is None else now
        pos = np.asarray(pos_enu, float)
        self._prune(now)
        if self._samples:
            last_pos = self._samples[-1][0]
            if float(np.linalg.norm(pos - last_pos)) < self.min_baseline_m:
                return False    # yeterince hareket edilmedi -> baz oluşmaz, örnek ALINMAZ
        d = azel_to_unit(bearing_deg, elevation_deg)
        self._samples.append((pos, d, now))
        if len(self._samples) > self.max_samples:
            self._samples.pop(0)
        return True

    def _prune(self, now):
        self._samples = [s for s in self._samples if (now - s[2]) <= self.decay_sec]

    def sample_count(self) -> int:
        return len(self._samples)

    def baseline_m(self) -> float:
        """Biriken örneklerin kapladığı azami mesafe (üçgenleme tabanı büyüklüğü)."""
        if len(self._samples) < 2:
            return 0.0
        pts = np.array([s[0] for s in self._samples])
        d = 0.0
        for i in range(len(pts)):
            d = max(d, float(np.max(np.linalg.norm(pts - pts[i], axis=1))))
        return round(d, 1)

    def estimate(self, now: float = None):
        """Biriken (konum, kerteriz) örneklerinden kaynağı üçgenle.
        Dönüş: (konum[x,y,z], kalıntı_m, fix, kesişim_açısı_derece). <2 örnek -> fix yok."""
        now = time.time() if now is None else now
        self._prune(now)
        if len(self._samples) < 2:
            return np.zeros(3), 0.0, False, 0.0
        positions = [s[0] for s in self._samples]
        directions = [s[1] for s in self._samples]
        return triangulate_lob(positions, directions)


# ------------------------------------------------------------------ #
#  DÜĞÜM KAYIT DEFTERİ (konumlar) + FÜZYON
# ------------------------------------------------------------------ #
def load_node_registry():
    """Düğüm konumlarını data/df_nodes.json'dan yükler; yoksa DEFAULT_NODES döner."""
    try:
        with open(_NODES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data:
            return data
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return dict(DEFAULT_NODES)


def self_node_id(registry) -> str:
    for nid, cfg in registry.items():
        if cfg.get("self"):
            return nid
    return next(iter(registry), "NODE-MAIN")


class NodeBearingStore:
    """Düğümlerden gelen kerteriz (LOB) ölçümlerinin THREAD-SAFE deposu. Uzak düğümler ağ üzerinden
    (UDP/JSON) yazar; yerel (ana) düğüm kendi genlik-DF kerterizini yazar. Ana döngü okur ve
    üçgenler. Bayat (stale) ölçümler yaşlarına göre elenir (düğüm sustuysa füzyona katılmaz)."""

    def __init__(self, stale_sec: float = 5.0):
        self.stale_sec = stale_sec
        self._lock = threading.Lock()
        self._bearings = {}   # id -> {azimuth_deg, elevation_deg, amp_dbm, snr_db, freq_mhz, ts}

    def update_from_json(self, msg: dict) -> bool:
        """Ağdan gelen JSON kerteriz mesajını doğrular ve kaydeder. Beklenen şema:
        {"id","azimuth_deg",["elevation_deg"],["amp_dbm"],["snr_db"],["freq_mhz"]}.
        Otonom: geçerli mesaj gelir gelmez depoya işlenir. Dönüş: kabul edildi mi."""
        nid = msg.get("id")
        if not nid or "azimuth_deg" not in msg:
            return False
        try:
            rec = {
                "azimuth_deg": float(msg["azimuth_deg"]) % 360.0,
                "elevation_deg": float(msg.get("elevation_deg", 0.0)),
                "amp_dbm": float(msg.get("amp_dbm", -120.0)),
                "snr_db": float(msg.get("snr_db", 0.0)),
                "freq_mhz": float(msg.get("freq_mhz", 0.0)),
                "ts": time.time(),
                "source": "network",
            }
        except (TypeError, ValueError):
            return False
        with self._lock:
            self._bearings[str(nid)] = rec
        return True

    def set_self_bearing(self, node_id: str, azimuth_deg, elevation_deg=0.0,
                         amp_dbm=-120.0, snr_db=0.0, freq_mhz=0.0):
        """Yerel (ana) düğümün genlik-DF kerterizini yazar. azimuth None ise ölçüm yok sayılır."""
        if azimuth_deg is None:
            with self._lock:
                self._bearings.pop(node_id, None)
            return
        with self._lock:
            self._bearings[node_id] = {
                "azimuth_deg": float(azimuth_deg) % 360.0,
                "elevation_deg": float(elevation_deg),
                "amp_dbm": float(amp_dbm), "snr_db": float(snr_db),
                "freq_mhz": float(freq_mhz), "ts": time.time(), "source": "local",
            }

    def active_bearings(self, now: float = None):
        """Bayat olmayan (taze) kerterizler: {id: rec}. Füzyona yalnızca bunlar katılır."""
        now = time.time() if now is None else now
        with self._lock:
            return {nid: dict(r) for nid, r in self._bearings.items()
                    if (now - r["ts"]) <= self.stale_sec}

    def snapshot(self):
        """Arayüz izleme için tüm kerterizlerin kopyası (bayat olanlar dahil, yaş bilgisiyle)."""
        now = time.time()
        with self._lock:
            return {nid: {**r, "age_sec": round(now - r["ts"], 1),
                          "stale": (now - r["ts"]) > self.stale_sec}
                    for nid, r in self._bearings.items()}
