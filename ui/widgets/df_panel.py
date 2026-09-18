from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QRadioButton, QLabel, QDoubleSpinBox, QPushButton, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
import numpy as np

class DFPanel(QWidget):
    mode_changed = pyqtSignal(bool) # True if auto, False if manual
    manual_angle_changed = pyqtSignal(int, float) # device_index, angle
    node_positions_changed = pyqtSignal(list)  # [(mesafe_m, açı°), ...] yardımcı düğümler için
    calibration_toggled = pyqtSignal(bool, float) # is_checked, reference_deg
    debug_iq_clicked = pyqtSignal()
    forward_gate_set = pyqtSignal(float)   # "İleri Yönü Ayarla" -> yarı-genişlik (derece)
    forward_gate_cleared = pyqtSignal()    # ileri-yay kapısını kapat

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)

        box_style = """
            QGroupBox {
                border: 2px solid #333333;
                border-radius: 8px;
                margin-top: 8px;
                font-size: 15px;
                font-weight: bold;
                color: #00e676;
                background-color: #1e1e1e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """

        # 3 CİHAZ AÇI (BEARING)
        self.box_angle = QGroupBox("3 CİHAZ AÇI (BEARING)")
        self.box_angle.setStyleSheet(box_style)
        layout_angle = QVBoxLayout(self.box_angle)
        
        mode_layout = QHBoxLayout()
        self.rbtn_auto = QRadioButton("Otonom")
        self.rbtn_auto.setChecked(True)
        self.rbtn_auto.setStyleSheet("color: #00e676; font-weight: bold; font-size: 15px;")
        self.rbtn_manual = QRadioButton("Manuel")
        self.rbtn_manual.setStyleSheet("color: #ffb74d; font-weight: bold; font-size: 15px;")
        mode_layout.addWidget(self.rbtn_auto)
        mode_layout.addWidget(self.rbtn_manual)
        layout_angle.addLayout(mode_layout)

        self.rbtn_auto.toggled.connect(self.on_mode_changed)

        # CANLI ANTEN AÇISI (ham enkoder) — Serial Monitor'deki gibi anlık, akıcı yön göstergesi.
        # SDR-1/2/3 kerterizi (sinyal-tabanlı tepe) gösterir; bu ise antenin O ANKİ fiziksel yönü.
        self.lbl_live_angle = QLabel("CANLI ANTEN AÇISI: --°")
        self.lbl_live_angle.setStyleSheet("font-size: 17px; font-weight: bold; color: #00e5ff; "
                                          "background-color: #10222a; padding: 4px; border-radius: 4px;")
        self.lbl_live_angle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_angle.addWidget(self.lbl_live_angle)

        # İLERİ YÖN KAPISI: anteni alanın ortasına çevir -> "İleri Yönü Ayarla" o anki açıyı yay MERKEZİ
        # yapar; kerteriz yalnızca [merkez±yarı] içinde aranır (arka/yay-dışı sahte kerterizler elenir).
        # Ana istasyon: yarı ~90 (180° yay). Köşe aux'lar kendi arayüzünde ~60 (120° yay) ayarlar.
        self._fwd_active = False
        fwd_row = QHBoxLayout()
        fwd_row.setSpacing(6)
        self.btn_fwd = QPushButton("İleri Yönü Ayarla")
        self.btn_fwd.setToolTip(
            "Anteni alanın ORTASINA çevir, bas: o anki açı 'ileri' merkez olur.\n"
            "Kerteriz yalnızca bu yay içinde aranır -> arkadaki şehir sinyali/yansıması elenir.\n"
            "Tekrar basınca kapanır (360° tarama).")
        self.btn_fwd.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #00695c; "
                                   "color: white; border-radius: 4px; padding: 4px;")
        self.btn_fwd.clicked.connect(self.on_forward_toggled)
        fwd_row.addWidget(self.btn_fwd, stretch=3)
        fwd_row.addWidget(QLabel("Yarı:"))
        self.spin_fwd_half = QDoubleSpinBox()
        self.spin_fwd_half.setRange(10.0, 179.0)
        self.spin_fwd_half.setValue(90.0)          # ana istasyon varsayılanı (180° yay)
        self.spin_fwd_half.setSuffix("°")
        self.spin_fwd_half.setDecimals(0)
        self.spin_fwd_half.setStyleSheet("font-size: 13px; padding: 2px; background-color: #333; "
                                         "color: #fff; border: 1px solid #555; border-radius: 4px;")
        fwd_row.addWidget(self.spin_fwd_half, stretch=1)
        layout_angle.addLayout(fwd_row)

        self.lbl_fwd_status = QLabel("İleri yay: kapalı (360°)")
        self.lbl_fwd_status.setStyleSheet("font-size: 12px; color: #80cbc4; padding-left: 2px;")
        layout_angle.addWidget(self.lbl_fwd_status)

        self.lbl_dev1_ang = QLabel("SDR-1: --° --'")
        self.lbl_dev2_ang = QLabel("SDR-2: --° --'")
        self.lbl_dev3_ang = QLabel("SDR-3: --° --'")
        
        self.spin_dev1 = QDoubleSpinBox()
        self.spin_dev2 = QDoubleSpinBox()
        self.spin_dev3 = QDoubleSpinBox()
        
        for i, spin in enumerate([self.spin_dev1, self.spin_dev2, self.spin_dev3]):
            spin.setRange(0, 360)
            spin.setDecimals(1)
            spin.setSuffix("°")
            spin.setVisible(False)
            spin.setStyleSheet("background-color: #333333; color: white; border: 1px solid #555;")
            spin.valueChanged.connect(lambda val, idx=i: self.manual_angle_changed.emit(idx, val))

        dev1_layout = QHBoxLayout()
        dev1_layout.addWidget(self.lbl_dev1_ang)
        dev1_layout.addWidget(self.spin_dev1)
        
        # AUX CANLI anten açısı (anlık, ağdan) — SDR-2/3 satırında kerterizin YANINDA.
        self.lbl_dev2_live = QLabel("")
        self.lbl_dev3_live = QLabel("")
        for lbl in (self.lbl_dev2_live, self.lbl_dev3_live):
            lbl.setStyleSheet("font-size: 13px; color: #00e5ff;")

        dev2_layout = QHBoxLayout()
        dev2_layout.addWidget(self.lbl_dev2_ang)
        dev2_layout.addWidget(self.lbl_dev2_live)
        dev2_layout.addWidget(self.spin_dev2)

        dev3_layout = QHBoxLayout()
        dev3_layout.addWidget(self.lbl_dev3_ang)
        dev3_layout.addWidget(self.lbl_dev3_live)
        dev3_layout.addWidget(self.spin_dev3)

        for lbl in [self.lbl_dev1_ang, self.lbl_dev2_ang, self.lbl_dev3_ang]:
            lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            
        layout_angle.addLayout(dev1_layout)
        layout_angle.addLayout(dev2_layout)
        layout_angle.addLayout(dev3_layout)
        
        layout.addWidget(self.box_angle, stretch=1)

        # AĞ VE ANTEN KONTROL + YARDIMCI DÜĞÜM KONUMLARI (mesafe + pusula açısı)
        self.box_antenna_ctrl = QGroupBox("AĞ VE YARDIMCI DÜĞÜM KONUMLARI")
        self.box_antenna_ctrl.setStyleSheet(box_style)
        layout_antenna_ctrl = QVBoxLayout(self.box_antenna_ctrl)
        layout_antenna_ctrl.setSpacing(6)

        # ANT-1 = ANA CİHAZ: konumu her zaman orijin (0, 0). Diğer düğümler buna GÖRE konumlanır.
        self.lbl_ant1_status = QLabel("ANT-1 (Merkez): BAĞLANTI YOK")
        self.lbl_ant1_status.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff5252; padding: 2px;")
        layout_antenna_ctrl.addWidget(self.lbl_ant1_status)
        lbl_ref = QLabel("Ana cihaz konumu: (0, 0) — sabit referans")
        lbl_ref.setStyleSheet("font-size: 13px; color: #90caf9; padding: 0 0 4px 2px;")
        layout_antenna_ctrl.addWidget(lbl_ref)

        # Yardımcı düğüm giriş satırı üreten yardımcı: durum etiketi + Mesafe(m) + Açı(pusula°)
        def _make_node_row(default_dist, default_ang):
            status = QLabel("BAĞLANTI YOK")
            status.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff5252; padding: 2px;")
            layout_antenna_ctrl.addWidget(status)
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl_d = QLabel("Mesafe:")
            lbl_d.setStyleSheet("font-size: 13px; color: #ffffff;")
            spin_d = QDoubleSpinBox()
            spin_d.setRange(0.0, 100000.0)
            spin_d.setDecimals(0)
            spin_d.setSuffix(" m")
            spin_d.setValue(default_dist)
            lbl_a = QLabel("Açı:")
            lbl_a.setStyleSheet("font-size: 13px; color: #ffffff;")
            spin_a = QDoubleSpinBox()
            spin_a.setRange(0.0, 360.0)
            spin_a.setDecimals(1)
            spin_a.setSuffix("°")
            spin_a.setValue(default_ang)
            for sp in (spin_d, spin_a):
                sp.setStyleSheet("background-color: #333333; color: white; border: 1px solid #555;")
            row.addWidget(lbl_d); row.addWidget(spin_d)
            row.addWidget(lbl_a); row.addWidget(spin_a)
            layout_antenna_ctrl.addLayout(row)
            return status, spin_d, spin_a

        # Varsayılanlar DEFAULT_NODES ile uyumlu (NODE-2: 500m/90°=Doğu, NODE-3: 500m/30°)
        self.lbl_ant2_status, self.spin_dist2, self.spin_ang2 = _make_node_row(500.0, 90.0)
        self.lbl_ant3_status, self.spin_dist3, self.spin_ang3 = _make_node_row(500.0, 30.0)

        self.btn_apply_positions = QPushButton("Konumları Uygula")
        self.btn_apply_positions.setStyleSheet("font-size: 14px; background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
        self.btn_apply_positions.clicked.connect(self.on_apply_positions)
        layout_antenna_ctrl.addWidget(self.btn_apply_positions)

        layout.addWidget(self.box_antenna_ctrl, stretch=1)

        # YÖN BULMA DOĞRULUĞU (KALİBRASYON)
        self.box_df_accuracy = QGroupBox("YÖN BULMA DOĞRULUĞU (KALİBRASYON)")
        self.box_df_accuracy.setStyleSheet(box_style)
        layout_df = QVBoxLayout(self.box_df_accuracy)
        layout_df.setSpacing(6)

        df_ref_row = QVBoxLayout()
        self.txt_df_reference = QLineEdit("0.0")
        self.txt_df_reference.setPlaceholderText("Referans Açı (°)")
        self.txt_df_reference.setValidator(QDoubleValidator(0.0, 359.9, 1))
        self.txt_df_reference.setStyleSheet("font-size: 16px; padding: 4px; background-color: #333333; color: #ffffff; border: 1px solid #555555; border-radius: 4px;")
        
        self.btn_df_calibrate = QPushButton("Kalibrasyonu Başlat")
        self.btn_df_calibrate.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #2e7d32; color: white; border-radius: 4px; padding: 4px;")
        self.btn_df_calibrate.setCheckable(True)
        self.btn_df_calibrate.clicked.connect(self.on_calibration_toggled)
        
        self.btn_debug_iq = QPushButton("Ham IQ Kaydet (Debug)")
        self.btn_debug_iq.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #f57c00; color: white; border-radius: 4px; padding: 4px;")
        self.btn_debug_iq.clicked.connect(self.debug_iq_clicked.emit)

        df_ref_row.addWidget(self.txt_df_reference)
        df_ref_row.addWidget(self.btn_df_calibrate)
        df_ref_row.addWidget(self.btn_debug_iq)
        layout_df.addLayout(df_ref_row)

        self.lbl_df_rms = QLabel("RMS Hata: -- ° (N=0)")
        self.lbl_df_rms.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffb74d;")
        layout_df.addWidget(self.lbl_df_rms)
        layout.addWidget(self.box_df_accuracy, stretch=1)

        # NOT: Eski addStretch(3) (sağdaki boşluk) KALDIRILDI — o boşluğu artık SAHA SOHBETİ paneli
        # dolduruyor (main_window kalibrasyonun sağına ekler, geniş stretch ile sola doğru uzanır).

    def on_apply_positions(self):
        """Yardımcı düğüm konumlarını (mesafe + pusula açısı) topla ve yayınla. Ana cihaz (0,0) sabittir."""
        polar_list = [
            (self.spin_dist2.value(), self.spin_ang2.value()),
            (self.spin_dist3.value(), self.spin_ang3.value()),
        ]
        self.node_positions_changed.emit(polar_list)

    def set_node_positions(self, polar_list):
        """Kayıtlı konumlarla girişleri doldur. polar_list: [(mesafe_m, açı°), ...] (yardımcı düğümler)."""
        spins = [(self.spin_dist2, self.spin_ang2), (self.spin_dist3, self.spin_ang3)]
        for (spin_d, spin_a), item in zip(spins, polar_list):
            spin_d.setValue(float(item[1]))   # mesafe_m
            spin_a.setValue(float(item[2]))   # açı°
            # item = (id, dist, bearing) — worker.get_aux_node_positions formatı

    def on_mode_changed(self, is_auto: bool):
        for spin in [self.spin_dev1, self.spin_dev2, self.spin_dev3]:
            spin.setVisible(not is_auto)
        self.mode_changed.emit(is_auto)

    def on_calibration_toggled(self, checked: bool):
        if checked:
            try:
                reference_deg = float(self.txt_df_reference.text() or 0.0)
            except ValueError:
                reference_deg = 0.0
            self.btn_df_calibrate.setText("Kalibrasyonu Durdur")
            self.btn_df_calibrate.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #c62828; color: white; border-radius: 4px; padding: 4px;")
            self.calibration_toggled.emit(True, reference_deg)
        else:
            self.btn_df_calibrate.setText("Kalibrasyonu Başlat")
            self.btn_df_calibrate.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #2e7d32; color: white; border-radius: 4px; padding: 4px;")
            self.lbl_df_rms.setText("RMS Hata: -- ° (N=0)")
            self.calibration_toggled.emit(False, 0.0)

    def on_forward_toggled(self):
        """İleri Yönü Ayarla (toggle): açıkken kapatır, kapalıyken o anki açıyı merkez yapıp açar."""
        if self._fwd_active:
            self._fwd_active = False
            self.btn_fwd.setText("İleri Yönü Ayarla")
            self.btn_fwd.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #00695c; "
                                       "color: white; border-radius: 4px; padding: 4px;")
            self.lbl_fwd_status.setText("İleri yay: kapalı (360°)")
            self.forward_gate_cleared.emit()
        else:
            self._fwd_active = True
            self.btn_fwd.setText("İleri Yön: AKTİF (kapat)")
            self.btn_fwd.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #c62828; "
                                       "color: white; border-radius: 4px; padding: 4px;")
            self.forward_gate_set.emit(float(self.spin_fwd_half.value()))

    def update_forward_status(self, center, half):
        """Payload'dan ileri-yay durumunu göster (center None -> kapalı). Backend gerçeğiyle senkron."""
        if center is None:
            self._fwd_active = False
            self.btn_fwd.setText("İleri Yönü Ayarla")
            self.btn_fwd.setStyleSheet("font-size: 13px; font-weight: bold; background-color: #00695c; "
                                       "color: white; border-radius: 4px; padding: 4px;")
            self.lbl_fwd_status.setText("İleri yay: kapalı (360°)")
        else:
            self._fwd_active = True
            self.lbl_fwd_status.setText(f"İleri yay: {center:.0f}° ±{half:.0f}° (aktif) ✓")

    def update_angles(self, angles, dms_func):
        labels = [self.lbl_dev1_ang, self.lbl_dev2_ang, self.lbl_dev3_ang]
        for i, lbl in enumerate(labels):
            lbl.setText(f"SDR-{i+1}: {dms_func(angles[i])}")

    def update_rms(self, rms_str):
        self.lbl_df_rms.setText(rms_str)

    def update_live_angle(self, deg):
        """Canlı ham anten açısını (enkoder) göster. None ise '--'."""
        if deg is None:
            self.lbl_live_angle.setText("CANLI ANTEN AÇISI: --°  (enkoder yok)")
        else:
            self.lbl_live_angle.setText(f"CANLI ANTEN AÇISI: {deg:.1f}°")

    def update_node_live_angles(self, live_list):
        """Aux (NODE-2/3) CANLI anten açısını SDR-2/3 satırında göster (anlık, ağdan).
        live_list = [node0, node1, node2] (derece | None). node0=merkez (kendi enkoderi, üstte)."""
        live_list = list(live_list or [])
        for i, lbl in ((1, self.lbl_dev2_live), (2, self.lbl_dev3_live)):
            v = live_list[i] if i < len(live_list) else None
            lbl.setText(f"⟳ canlı {v:.0f}°" if isinstance(v, (int, float)) else "")
