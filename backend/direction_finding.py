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
# KALINTI (residual) kapısı: kerterizler tutarsızsa (biri yanlış/yansımalı) ışınlar iyi kesişmez ->
# dik kalıntı büyür. Kesişim açısı iyi olsa bile YÜKSEK kalıntı = güvenilmez fix. Kalıntı, hedef
# MENZİLİNE göre değerlendirilir (uzak hedefte küçük açı hatası büyük mutlak kalıntı verir). GEVŞEK
# eşik: yalnızca AŞIRI tutarsız geometriyi reddeder (geçerli fix'i düşürmesin).
MAX_RESIDUAL_FRAC = 0.12         # kalıntı/menzil bunu aşarsa fix güvenilmez. Ampirik: gerçekçi ±2°
                                 # gürültü ~0.02-0.06; bir kerteriz 25-40° yanlışsa ~0.10-0.16 -> ayrışır.
MAX_RESIDUAL_FLOOR_M = 30.0      # yakın hedefte mutlak alt taban (oran çok küçük menzilde katı olmasın)

# Kerteriz belirsizliği (σ, derece) — ağırlıklı üçgenlemede 1/σ² ağırlığı verir.
# CENTROID yöntemi hüzme genişliğiyle sınırlı, kabaca ±birkaç derece: muhafazakâr sabit taban.
SIGMA_CENTROID_DEG = 8.0
from backend.antenna_pattern import SIGMA_CEIL_DEG   # σ tavanı (pattern modülüyle ortak)


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


def triangulate_lob(positions, directions, sigmas=None):
    """3B LOB (kerteriz doğrusu) EN-KÜÇÜK-KARELER üçgenlemesi + GEOMETRİ (GDOP) kapısı.

    Her düğüm i, konum p_i'den d_i yönünde bir ışın (LOB) tanımlar. Işınlar gürültü yüzünden tam
    kesişmez; kaynağın konumu, TÜM ışınlara dik uzaklıkların karelerinin toplamını EN AZ yapan
    noktadır: min_x Σ |(I - d_i d_iᵀ)(x - p_i)|². Çözüm: (Σ Aᵢ) x = Σ Aᵢ pᵢ,  Aᵢ = I - dᵢ dᵢᵀ.

    Dönüş: (konum[x,y,z], ortalama_kalıntı_m, kesişim_güvenilir_mi, kesişim_açısı_derece).
    Yükseklik bilgisi yoksa (tüm LOB'lar yatay) z ekseni belirsizdir -> 2B çözülür, z=0 (havadaki
    kaynağın YER İZDÜŞÜMÜ bulunur; irtifa için elevation ölçümü gerekir). En az 2 LOB gerekir.
    GDOP kapısı: LOB'lar neredeyse paralelse (en geniş kesişim < MIN_CROSSING_ANGLE_DEG) konum
    güvenilmezdir -> fix=False (sahte-güvenli konum üretilmez).
    İLERİ-YÖN kapısı: çözüm herhangi bir düğümün kerteriz ışınının ARKASINDAysa (hayalet hedef)
    -> fix=False. Işınlar yön taşır; matematiksel doğrular taşımaz.

    AĞIRLIK (sigmas): her düğümün kerteriz belirsizliği σ_i (derece) verilirse, o düğüm en-küçük-
    karelerde 1/σ_i² ile ağırlıklanır -> hassas (pattern-eşleşmiş, küçük σ) kerterizler baskın olur,
    belirsiz (düşük ön/arka) olanlar az etkiler. sigmas=None -> eşit ağırlık (eski davranış).
    """
    positions = [np.asarray(p, float) for p in positions]
    directions = [np.asarray(d, float) / (np.linalg.norm(d) + 1e-12) for d in directions]
    if len(positions) < 2:
        return np.zeros(3), 0.0, False, 0.0

    cross_deg = lob_crossing_angle_deg(directions)

    if sigmas is not None and len(sigmas) == len(directions):
        wts = [1.0 / max(float(s), 1e-3) ** 2 for s in sigmas]
    else:
        wts = [1.0] * len(directions)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d, wt in zip(positions, directions, wts):
        P = wt * (np.eye(3) - np.outer(d, d))
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
    # İLERİ-YÖN (RAY) KAPISI — HAYALET HEDEF önleme (KRİTİK): lstsq sonsuz DOĞRULARI kesiştirir,
    # ışınların YÖNÜNÜ (ileri/geri) yok sayar. LOB'lar ölçüm gürültüsüyle (ör. birkaç derece Kuzey
    # sapması / yansıma) ıraksarsa, en-küçük-kareler çözümü ışınları GERİYE uzatıp kesişimi antenlerin
    # ARKASINDA bulabilir -> harita hedefi gerçek yönün TAM TERSİNE fırlatır ("hayalet hedef").
    # Gerçek kaynak, HER düğümün kerteriz ışınının ÖNÜNDE olmalıdır: (x - p_i)·d_i > 0.
    # Herhangi bir düğüm hedefi arkasında "görüyorsa" geometri fiziksel olarak tutarsızdır -> fix YOK
    # (sahte-güvenli konum üretilmez; sistemin genel dürüstlük ilkesiyle uyumlu).
    forward_ok = all(float(np.dot(x - p, d)) > 0.0 for p, d in zip(positions, directions))
    # KALINTI KAPISI (uzman #16): kerterizler tutarsızsa kesişim açısı iyi olsa bile kalıntı büyür ->
    # fix güvenilmez. Menzile göre (oran) değerlendir; gevşek eşik (yalnız aşırı tutarsızı reddet).
    range_m = float(np.linalg.norm(x - np.mean(positions, axis=0)))
    residual_ok = res <= max(MAX_RESIDUAL_FLOOR_M, MAX_RESIDUAL_FRAC * range_m)
    # GDOP kapısı: geometri çok zayıfsa (ışınlar ~paralel) konum güvenilmez -> fix yok.
    fix = (cross_deg >= MIN_CROSSING_ANGLE_DEG) and forward_ok and residual_ok
    return x, round(res, 2), fix, round(cross_deg, 1)


# ------------------------------------------------------------------ #
#  GENLİK TABANLI KERTERİZ KESTİRİCİ (yerel düğüm)
# ------------------------------------------------------------------ #
class AmplitudeDFEstimator:
    """GENLİK TABANLI DF: anten ELLE döndürülürken (azimut, genlik) örnekleri gelir; en yüksek
    genliğin azimutu = kaynağın geliş yönü. Tepe etrafında genlik-ağırlıklı merkez (centroid) ile
    hassasiyet sağlanır (ANTEN HÜZME GENİŞLİĞİYLE sınırlı; LPDA'da ±birkaç derece, "alt-derece" değil).
    Örnekler zamanla sönümlenir (yeniden tarama / hareketli kaynak).

    Kullanım (yerel düğüm): her karede update(enkoder_azimutu, ölçülen_genlik_dBm). bearing()
    o ana kadarki taramadan en olası kerterizi verir."""

    def __init__(self, bin_deg: float = 1.0, window_deg: float = 25.0, decay_sec: float = 15.0,
                 freq_hz: float = None, use_pattern: bool = True):
        self.bin_deg = bin_deg
        self.window_deg = window_deg          # tepe etrafı centroid penceresi
        self.decay_sec = decay_sec            # bu süreden eski açı örnekleri unutulur. ELLE dönüşte
                                              # (~10 sn/tur) 8 sn kısaydı: tur bitmeden ilk taranan
                                              # bin'ler silinip 360° resmi eksik kalabiliyordu -> 15 sn.
        self._bins = {}                       # az_bin(int) -> (amp_dbm, ts)
        # PATTERN EŞLEŞTİRME (şartname 5.1.4 Derece RMS): ölçülen VNA pattern'i varsa kerteriz tüm
        # eğrinin ŞEKLİNE oturtularak (sadece tepe/centroid değil) çok daha hassas + belirsizlik (σ)
        # ile kestirilir. freq_hz kaynağın frekansı; None ise pattern kullanılmaz (centroid'e düşer).
        self.freq_hz = freq_hz
        self._pattern = None
        if use_pattern:
            try:
                from backend.antenna_pattern import shared_pattern
                ap = shared_pattern()
                self._pattern = ap if ap.available() else None
            except Exception:
                self._pattern = None

        self._last_sigma = SIGMA_CENTROID_DEG   # son bearing()'in belirsizliği (derece) — ağırlıklı üçgenleme için
        self._last_method = "centroid"          # "pattern" | "centroid" | ... (teşhis / arayüz)
        self._last_coverage = 0.0               # son taramanın açısal kapsaması (derece) — teşhis
        # İLERİ-YAY KAPISI (opsiyonel): alanın önünde olduğu bilindiğinde, kerteriz araması yalnızca
        # [merkez±yarı] içinde yapılır -> arkadaki/yay-dışı sahte kerterizler (şehir yansıması, arka lob)
        # elenir; dar ileri-taramada (köşe istasyon ~90°) pattern eşleştirme de açık kalır. None=kapalı.
        self.fwd_center = None                  # ileri yön merkezi (derece) — None ise kapı KAPALI
        self.fwd_half = 90.0                    # yay yarı-genişliği (derece); toplam yay = 2×bu

    def reset(self):
        self._bins.clear()

    def set_freq(self, freq_hz):
        """Kaynak frekansını ayarlar (pattern eşleştirme frekansa bağlı). None -> pattern kapalı."""
        self.freq_hz = float(freq_hz) if freq_hz else None

    def set_forward_gate(self, center_deg, half_deg=None):
        """İleri-yay kapısını aç: kerteriz yalnızca [center±half] içinde aranır. half None -> mevcut kalır."""
        self.fwd_center = float(center_deg) % 360.0
        if half_deg is not None:
            self.fwd_half = float(np.clip(half_deg, 5.0, 179.0))

    def clear_forward_gate(self):
        """İleri-yay kapısını kapat (tam 360° arama)."""
        self.fwd_center = None

    def forward_gate(self):
        """(merkez, yarı) veya (None, yarı) — arayüz/teşhis için."""
        return self.fwd_center, self.fwd_half

    def _in_arc(self, az_deg):
        """az ileri yay içinde mi (kapı kapalıysa daima True)."""
        if self.fwd_center is None:
            return True
        return abs(((float(az_deg) - self.fwd_center + 180.0) % 360.0) - 180.0) <= self.fwd_half

    def last_sigma(self) -> float:
        """Son bearing() çağrısının kerteriz belirsizliği (derece). Üçgenlemede 1/σ² ağırlık için."""
        return self._last_sigma

    def last_method(self) -> str:
        """Son kerterizin yöntemi ('pattern' / 'centroid' ...) — teşhis/arayüz için."""
        return self._last_method

    def last_coverage(self) -> float:
        """Son taramanın açısal kapsaması (derece) — teşhis/arayüz için."""
        return self._last_coverage

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
        self._last_sigma = SIGMA_CENTROID_DEG
        self._last_method = "centroid"
        if len(self._bins) < 3:
            return None, -120.0, 0.0, len(self._bins)

        azs = np.array([k * self.bin_deg for k in self._bins.keys()])
        amps = np.array([v[0] for v in self._bins.values()])
        # İLERİ-YAY KAPISI: kapalıysa (fwd_center None) hiçbir şey değişmez. Açıksa YALNIZCA yay içindeki
        # örneklerle çalış -> yay dışı (arka/kenar) tepe kerteriz üretemez. Yeterli örnek yoksa 'ölçülemedi'.
        if self.fwd_center is not None:
            in_arc = np.array([self._in_arc(a) for a in azs])
            azs, amps = azs[in_arc], amps[in_arc]
            if len(azs) < 3:
                return None, -120.0, 0.0, len(self._bins)
        peak_i = int(np.argmax(amps))
        peak_az = float(azs[peak_i])
        peak_amp = float(amps[peak_i])
        floor = float(np.min(amps))

        # Tepe etrafında (±window) genlik-ağırlıklı dairesel centroid (hassasiyet hüzme genişliğiyle sınırlı)
        offs = ((azs - peak_az + 180.0) % 360.0) - 180.0     # tepeye göre [-180,180]
        mask = np.abs(offs) <= self.window_deg
        w = np.power(10.0, (amps[mask] - floor) / 10.0)       # dBm -> doğrusal güç ağırlığı
        centroid_off = float(np.sum(offs[mask] * w) / (np.sum(w) + 1e-12))
        bearing_deg = (peak_az + centroid_off) % 360.0

        # Güven: tepe-taban farkı ne kadar büyükse o kadar yüksek (0..1), ~20 dB'de doyar
        conf = float(min(1.0, max(0.0, (peak_amp - floor) / 20.0)))

        # PATTERN EŞLEŞTİRME ile RAFİNE (şartname 5.1.4): ölçülen VNA pattern'i + frekans varsa,
        # tüm eğriyi kalibre pattern'e oturtarak çok daha hassas kerteriz + σ elde et.
        # KALİTE KAPISI (uzman P0 — GERÇEK fallback): pattern yalnızca quality_ok ise KULLANILIR
        # (frekans aralıkta + açısal kapsama yeterli + ön/arka yüksek + ambiguity düşük). Aksi halde
        # centroid kerterizi KORUNUR (pattern bearing atılır, σ şişirmekle yetinilmez). Frekans/pattern
        # yoksa da centroid.
        if self._pattern is not None and self.freq_hz:
            m = self._pattern.match(azs, amps, self.freq_hz,
                                    arc_center=self.fwd_center, arc_half=self.fwd_half)
            if m is not None:
                self._last_coverage = m.get("coverage_deg", 0.0)
                if m["quality_ok"]:
                    bearing_deg = m["bearing_deg"]
                    self._last_sigma = m["sigma_deg"]
                    self._last_method = "pattern"
                else:
                    # Pattern güvenilmez -> CENTROID kerterizi kullan (yukarıda hesaplandı). σ, pattern'in
                    # düşük güvenini yansıtacak şekilde centroid tabanının biraz üstünde tutulur.
                    self._last_sigma = min(SIGMA_CEIL_DEG, SIGMA_CENTROID_DEG * 1.5)
                    self._last_method = "centroid(pattern-düşük-kalite)"
        return round(bearing_deg, 2), round(peak_amp, 1), round(conf, 2), len(self._bins)


# NOT: "MovingReceiverPositioner" (spec 5.1.5 tek HAREKETLİ alıcı ile konum) KALDIRILDI —
# saha kurulumu 3 SABİT istasyon + dönen yönlü anten. Konum bulma 3-düğüm LOB üçgenlemesiyle
# (triangulate_lob + NodeBearingStore füzyonu) yapılıyor; hareketli-tek-alıcı yöntemi kullanılmıyor.


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


def polar_to_enu(dist_m: float, bearing_deg: float) -> list:
    """Ana cihaza göre MESAFE (m) + PUSULA AÇISI (Kuzey'den saat yönü, derece) -> ENU [Doğu, Kuzey, 0].
    Sahada yardımcı düğümü elle konumlandırmanın en kolay yolu: metreyle uzaklık + pusulayla açı.
    azel_to_unit ile aynı çerçeve (x=r·sin(az), y=r·cos(az)); yükseklik yerde 0 alınır."""
    r = float(dist_m)
    az = np.radians(float(bearing_deg))
    return [r * float(np.sin(az)), r * float(np.cos(az)), 0.0]


def enu_to_polar(pos) -> tuple:
    """ENU [Doğu, Kuzey, Yukarı] -> (mesafe_m, pusula_açısı_derece). polar_to_enu'nun tersi;
    kayıtlı konumları arayüzde mesafe+açı olarak göstermek için."""
    p = np.asarray(pos, float)
    dist = float(np.hypot(p[0], p[1]))
    bearing = float(np.degrees(np.arctan2(p[0], p[1]))) % 360.0     # atan2(Doğu, Kuzey)
    return round(dist, 1), round(bearing, 1)


def save_node_registry(registry) -> bool:
    """Düğüm kayıt defterini data/df_nodes.json'a yazar (sahada girilen konumlar kalıcı olsun).
    Başarısızsa sessizce False döner (dosya sistemi yoksa uygulama yine çalışır)."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_NODES_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


class NodeBearingStore:
    """Düğümlerden gelen kerteriz (LOB) ölçümlerinin THREAD-SAFE deposu. Uzak düğümler ağ üzerinden
    (UDP/JSON) yazar; yerel (ana) düğüm kendi genlik-DF kerterizini yazar. Ana döngü okur ve
    üçgenler. Bayat (stale) ölçümler yaşlarına göre elenir (düğüm sustuysa füzyona katılmaz)."""

    def __init__(self, stale_sec: float = 2.0):
        # stale_sec 5.0 -> 2.0: aux 10 Hz gönderdiği için taze veri hep <0.2 sn yaştadır. AUX kilitlenir/
        # ağı koparsa, ESKİ kerterizi 5 sn boyunca "taze" sayıp füzyona sokmak HAREKETLİ hedefte konumu
        # geriye çeker (bayat-veri zehirlenmesi). 2 sn'de düşür -> ölü düğüm hızla füzyondan atılır.
        self.stale_sec = stale_sec
        self._lock = threading.Lock()
        self._bearings = {}   # id -> {azimuth_deg, elevation_deg, amp_dbm, snr_db, freq_mhz, ts}
        self._live = {}       # id -> (live_angle_deg, ts)  CANLI enkoder açısı (kerterizden BAĞIMSIZ)

    def update_from_json(self, msg: dict) -> bool:
        """Ağdan gelen JSON mesajını işler. Şema:
        {"id",["azimuth_deg"],["elevation_deg"],["amp_dbm"],["snr_db"],["freq_mhz"],["live_angle_deg"]}.
        azimuth_deg (KERTERİZ) varsa füzyon deposuna, live_angle_deg (CANLI açı) varsa ayrı canlı depoya
        yazılır. Canlı açı KERTERİZ GEREKTİRMEZ -> aux, tepe bulmadan da anlık yönünü anında iletir.
        Dönüş: en az biri işlendiyse True."""
        nid = msg.get("id")
        if not nid:
            return False
        nid = str(nid)
        now = time.time()
        handled = False
        # CANLI açı (kerterizsiz de olabilir) -> ayrı depo, anlık gösterim için
        if msg.get("live_angle_deg") is not None:
            try:
                with self._lock:
                    self._live[nid] = (float(msg["live_angle_deg"]) % 360.0, now)
                handled = True
            except (TypeError, ValueError):
                pass
        # KERTERİZ (füzyon için) -> yalnızca azimuth_deg varsa
        if "azimuth_deg" in msg:
            try:
                rec = {
                    "azimuth_deg": float(msg["azimuth_deg"]) % 360.0,
                    "elevation_deg": float(msg.get("elevation_deg", 0.0)),
                    "amp_dbm": float(msg.get("amp_dbm", -120.0)),
                    "snr_db": float(msg.get("snr_db", 0.0)),
                    "freq_mhz": float(msg.get("freq_mhz", 0.0)),
                    # sigma_deg: aux'un kerteriz belirsizliği (pattern-eşleşmede küçük). Yoksa centroid tabanı.
                    "sigma_deg": float(msg.get("sigma_deg", SIGMA_CENTROID_DEG)),
                    "ts": now,
                    "source": "network",
                }
                with self._lock:
                    self._bearings[nid] = rec
                handled = True
            except (TypeError, ValueError):
                pass
        return handled

    def live_angles(self, now: float = None):
        """Taze CANLI enkoder açıları: {id: derece}. Aux antenin anlık yönü (kerterizden bağımsız)."""
        now = time.time() if now is None else now
        with self._lock:
            return {nid: ang for nid, (ang, ts) in self._live.items()
                    if (now - ts) <= self.stale_sec}

    def set_self_bearing(self, node_id: str, azimuth_deg, elevation_deg=0.0,
                         amp_dbm=-120.0, snr_db=0.0, freq_mhz=0.0, sigma_deg=SIGMA_CENTROID_DEG):
        """Yerel (ana) düğümün genlik-DF kerterizini yazar. azimuth None ise ölçüm yok sayılır.
        sigma_deg: kerteriz belirsizliği (pattern-eşleşmede küçük) -> ağırlıklı üçgenlemede kullanılır."""
        if azimuth_deg is None:
            with self._lock:
                self._bearings.pop(node_id, None)
            return
        with self._lock:
            self._bearings[node_id] = {
                "azimuth_deg": float(azimuth_deg) % 360.0,
                "elevation_deg": float(elevation_deg),
                "amp_dbm": float(amp_dbm), "snr_db": float(snr_db),
                "freq_mhz": float(freq_mhz), "sigma_deg": float(sigma_deg),
                "ts": time.time(), "source": "local",
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
