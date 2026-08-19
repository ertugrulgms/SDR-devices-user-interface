from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QRadioButton, QLabel, QDoubleSpinBox, QPushButton, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
import numpy as np

class DFPanel(QWidget):
    mode_changed = pyqtSignal(bool) # True if auto, False if manual
    manual_angle_changed = pyqtSignal(int, float) # device_index, angle
    start_scan_clicked = pyqtSignal()
    calibration_toggled = pyqtSignal(bool, float) # is_checked, reference_deg
    debug_iq_clicked = pyqtSignal()

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
        
        dev2_layout = QHBoxLayout()
        dev2_layout.addWidget(self.lbl_dev2_ang)
        dev2_layout.addWidget(self.spin_dev2)
        
        dev3_layout = QHBoxLayout()
        dev3_layout.addWidget(self.lbl_dev3_ang)
        dev3_layout.addWidget(self.spin_dev3)

        for lbl in [self.lbl_dev1_ang, self.lbl_dev2_ang, self.lbl_dev3_ang]:
            lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
            
        layout_angle.addLayout(dev1_layout)
        layout_angle.addLayout(dev2_layout)
        layout_angle.addLayout(dev3_layout)
        
        layout.addWidget(self.box_angle, stretch=1)

        # AĞ VE ANTEN KONTROL (SERVO)
        self.box_antenna_ctrl = QGroupBox("AĞ VE ANTEN KONTROL (SERVO)")
        self.box_antenna_ctrl.setStyleSheet(box_style)
        layout_antenna_ctrl = QVBoxLayout(self.box_antenna_ctrl)
        layout_antenna_ctrl.setSpacing(6)
        
        self.lbl_ant1_status = QLabel("ANT-1 (Merkez): BAĞLANTI YOK")
        self.lbl_ant2_status = QLabel("ANT-2 (150m): BAĞLANTI YOK")
        self.lbl_ant3_status = QLabel("ANT-3 (150m): BAĞLANTI YOK")
        for lbl in [self.lbl_ant1_status, self.lbl_ant2_status, self.lbl_ant3_status]:
            lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff5252; padding: 2px;")
            layout_antenna_ctrl.addWidget(lbl)
            
        self.btn_start_scan = QPushButton("Otonom Anten Taraması Başlat")
        self.btn_start_scan.setStyleSheet("font-size: 14px; background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; padding: 4px;")
        self.btn_start_scan.clicked.connect(self.start_scan_clicked.emit)
        layout_antenna_ctrl.addWidget(self.btn_start_scan)
        
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
        
        # Panellerin genişliğini yarıya düşürüp sola dayamak için:
        layout.addStretch(3)

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

    def update_angles(self, angles, dms_func):
        labels = [self.lbl_dev1_ang, self.lbl_dev2_ang, self.lbl_dev3_ang]
        for i, lbl in enumerate(labels):
            lbl.setText(f"SDR-{i+1}: {dms_func(angles[i])}")

    def update_rms(self, rms_str):
        self.lbl_df_rms.setText(rms_str)
