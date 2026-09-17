"""PPI (Plan Position Indicator) Radar Widget'ı — yön bulma/konum görselleştirmesi.

Merkez = ana (yerel) düğüm. Kuzey YUKARI. Eş-merkezli MENZİL HALKALARI (metre etiketli). Tespit
edilen kaynak, ana düğüme göre azimut + menzilde bir blip olarak gösterilir. Ayrıca düğüm
konumları ve her düğümün kerteriz (LOB) ışını çizilir (üçgenleme geometrisi görünür olsun).

PPI ekseni: ekran x = Doğu, y = Kuzey (yukarı). Azimut Kuzey'den saat yönü -> x=r·sin(az), y=r·cos(az).
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import numpy as np
import pyqtgraph as pg


class PPIWidget(QWidget):
    NODE_COLORS = ['#00e676', '#ffb74d', '#4fc3f7', '#ba68c8', '#ff8a65']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ring_items = []
        self._ring_labels = []
        self._node_rays = []
        self._node_markers = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_title = QLabel("RADAR")
        self.lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffffff;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_title)

        self.lbl_target = QLabel("HEDEF: KESTİRİM BEKLENİYOR...")
        self.lbl_target.setStyleSheet("font-size: 13px; font-weight: bold; color: #ff5252; "
                                      "background-color: #2a2a2a; padding: 4px; border-radius: 4px;")
        self.lbl_target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_target.setWordWrap(True)
        layout.addWidget(self.lbl_target)

        self.plot = pg.PlotWidget()
        self.plot.setBackground('#0d1b0d')             # koyu yeşil radar zemini
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMinimumSize(180, 180)             # sıkışabilir minimum; pencere ekrana sığsın
        layout.addWidget(self.plot)

        self._max_range_m = 2000.0     # varsayılan menzil: halkalar 500/1000/1500/2000 m (saha ~2 km)
        self._draw_static()

        # Hedef blip (kaynak konumu)
        self.target_scatter = pg.ScatterPlotItem(size=16, symbol='o',
                                                 pen=pg.mkPen('#ffffff', width=2),
                                                 brush=pg.mkBrush('#ff1744'))
        self.plot.addItem(self.target_scatter)

        # CANLI ANTEN YÖN ÇİZGİSİ (ham enkoder açısı) — merkezden antenin O ANKİ yönüne uzanan,
        # antenle birlikte AKICI dönen parlak süpürme çizgisi (Serial Monitor'deki açı gibi anlık).
        self.heading_line = pg.PlotDataItem([0, 0], [0, 0],
                                            pen=pg.mkPen('#00e5ff', width=3))
        self.plot.addItem(self.heading_line)

    # ------------------------------------------------------------------ #
    def _draw_static(self):
        """Menzil halkaları + eksen çizgileri + K/D/G/B etiketleri (mevcut menzile göre)."""
        for it in self._ring_items + self._ring_labels:
            self.plot.removeItem(it)
        self._ring_items.clear()
        self._ring_labels.clear()

        R = self._max_range_m
        theta = np.linspace(0, 2 * np.pi, 120)
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = R * frac
            ring = pg.PlotDataItem(r * np.cos(theta), r * np.sin(theta),
                                   pen=pg.mkPen('#1f5f1f', width=1, style=Qt.PenStyle.DashLine))
            self.plot.addItem(ring)
            self._ring_items.append(ring)
            # Menzil etiketi (metre) — kuzey ekseni üzerinde
            lbl = pg.TextItem(f"{int(r)} m", color='#4caf50', anchor=(0.5, 0.5))
            lbl.setPos(0, r)
            self.plot.addItem(lbl)
            self._ring_labels.append(lbl)

        # Eksen çizgileri (K-G, D-B)
        ax1 = pg.PlotDataItem([-R, R], [0, 0], pen=pg.mkPen('#1f5f1f', width=1))
        ax2 = pg.PlotDataItem([0, 0], [-R, R], pen=pg.mkPen('#1f5f1f', width=1))
        self.plot.addItem(ax1); self.plot.addItem(ax2)
        self._ring_items += [ax1, ax2]

        # Yön etiketleri (Kuzey yukarı)
        for text, x, y in [("K", 0, R * 1.08), ("G", 0, -R * 1.08),
                           ("D", R * 1.08, 0), ("B", -R * 1.08, 0)]:
            t = pg.TextItem(text, color='#81c784', anchor=(0.5, 0.5))
            t.setPos(x, y)
            self.plot.addItem(t)
            self._ring_labels.append(t)

        # Merkez (ana düğüm) işareti
        center = pg.ScatterPlotItem([0], [0], size=12, symbol='+',
                                    pen=pg.mkPen('#ffffff', width=2))
        self.plot.addItem(center)
        self._ring_items.append(center)

        self.plot.setXRange(-R * 1.15, R * 1.15)
        self.plot.setYRange(-R * 1.15, R * 1.15)

    @staticmethod
    def _azr_to_xy(az_deg, r):
        """Azimut (Kuzey'den saat yönü) + menzil -> PPI ekran koordinatı (x=Doğu, y=Kuzey)."""
        a = np.radians(az_deg)
        return r * np.sin(a), r * np.cos(a)

    def update_ppi(self, payload):
        """Worker payload'ından PPI'yı günceller: düğüm konumları/kerterizleri + hedef blip."""
        # CANLI ANTEN YÖN ÇİZGİSİ (ham enkoder açısı) — antenle birlikte akıcı döner. Kerterizden
        # (sinyal-tabanlı tepe) BAĞIMSIZDIR; anten fiziksel olarak nereye bakıyorsa oraya uzanır.
        enc = payload.get("encoder_angle_deg")
        if enc is not None:
            hx, hy = self._azr_to_xy(float(enc), self._max_range_m)
            self.heading_line.setData([0.0, hx], [0.0, hy])
        else:
            self.heading_line.setData([0.0, 0.0], [0.0, 0.0])

        nodes = payload.get("df_nodes", {}) or {}
        fix = payload.get("df_fix", False)
        pos = payload.get("df_position_xyz_m", [0.0, 0.0, 0.0])
        tgt_az = payload.get("df_target_bearing_deg", 0.0)
        tgt_rng = payload.get("df_target_range_m", 0.0)
        tgt_el = payload.get("df_target_elevation_deg", 0.0)

        # Ana düğüm konumu (merkez referansı) — is_self olan düğüm
        self_pos = np.array([0.0, 0.0, 0.0])
        for r in nodes.values():
            if r.get("is_self"):
                self_pos = np.array(r.get("pos", [0.0, 0.0, 0.0]), float)
                break

        # Menzil ölçeğini otomatik ayarla (hedef + düğüm uzaklıkları sığsın)
        dists = [tgt_rng]
        for r in nodes.values():
            p = np.array(r.get("pos", [0.0, 0.0, 0.0]), float) - self_pos
            dists.append(float(np.hypot(p[0], p[1])))
        need = max(dists) if dists else 0.0
        # Taban 2000 m: radar normalde hep 0/500/1000/1500/2000 gösterir; hedef/düğüm 2000 m'yi
        # aşarsa ölçek büyür (uzak hedef kırpılmaz), altında sabit 2000 m kalır.
        target_scale = max(2000.0, need * 1.2)
        # Halkaları yalnızca %25+ değişince yeniden çiz (titremeyi önle)
        if abs(target_scale - self._max_range_m) / max(self._max_range_m, 1e-9) > 0.25:
            self._max_range_m = round(target_scale, -1)
            self._draw_static()

        # Eski düğüm ışın/işaretlerini temizle
        for it in self._node_rays + self._node_markers:
            self.plot.removeItem(it)
        self._node_rays.clear()
        self._node_markers.clear()

        # Düğümleri ve kerteriz (LOB) ışınlarını çiz
        for i, (nid, r) in enumerate(nodes.items()):
            if r.get("stale"):
                continue
            p = np.array(r.get("pos", [0.0, 0.0, 0.0]), float) - self_pos
            color = self.NODE_COLORS[i % len(self.NODE_COLORS)]
            mk = pg.ScatterPlotItem([p[0]], [p[1]], size=11, symbol='t',
                                    pen=pg.mkPen('#ffffff', width=1), brush=pg.mkBrush(color))
            self.plot.addItem(mk); self._node_markers.append(mk)
            # Kerteriz ışını: düğüm konumundan azimut yönünde
            az = r.get("azimuth_deg", 0.0)
            dx, dy = self._azr_to_xy(az, self._max_range_m)
            ray = pg.PlotDataItem([p[0], p[0] + dx], [p[1], p[1] + dy],
                                  pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
            self.plot.addItem(ray); self._node_rays.append(ray)

        # Hedef blip
        if fix and tgt_rng > 0:
            tx, ty = self._azr_to_xy(tgt_az, tgt_rng)
            self.target_scatter.setData([tx], [ty])
            # Konum kalitesi (5.1.5): kesişim açısı (GDOP), kalıntı, boyut. Geometri zayıfsa uyar.
            cross = payload.get("df_fix_quality_deg", 0.0)
            resid = payload.get("df_residual_m", 0.0)
            dim = payload.get("df_dimensionality", "-")
            geo = "iyi" if cross >= 30 else ("orta" if cross >= 15 else "zayıf")
            self.lbl_target.setText(
                f"HEDEF: {tgt_az:.0f}° / {tgt_rng:.0f} m / yük {tgt_el:.0f}°  |  "
                f"X:{pos[0]:.0f} Y:{pos[1]:.0f} Z:{pos[2]:.0f} m  |  "
                f"{dim} · kesişim {cross:.0f}° ({geo}) · kalıntı {resid:.0f} m")
            self.lbl_target.setStyleSheet("font-size: 13px; font-weight: bold; color: white; "
                                          "background-color: #c62828; padding: 4px; border-radius: 4px;")
        else:
            self.target_scatter.setData([], [])
            self.lbl_target.setText("HEDEF: KESTİRİM BEKLENİYOR... (≥2 taze düğüm kerterizi gerekli)")
            self.lbl_target.setStyleSheet("font-size: 13px; font-weight: bold; color: #ff5252; "
                                          "background-color: #2a2a2a; padding: 4px; border-radius: 4px;")
