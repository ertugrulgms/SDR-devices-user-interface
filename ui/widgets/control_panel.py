from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSlider, QDoubleSpinBox, QComboBox, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal

class ControlPanel(QWidget):
    # Signals to communicate with MainWindow/Worker
    freq_changed = pyqtSignal(float)
    gain_changed = pyqtSignal(int)
    bw_changed = pyqtSignal(str)
    antenna_changed = pyqtSignal(str)
    toggle_capture = pyqtSignal()
    open_tx_dialog = pyqtSignal()
    scan_toggled = pyqtSignal()          # BANT TARA aç/kapa (sinyal tespiti, 5.1.1)
    scan_sensitivity_changed = pyqtSignal(int)   # tarama hassasiyeti (CFAR yerel-belirginlik, dB)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. Satır: Frekans ve Bant Genişliği (Yan Yana)
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)

        # Taşıyıcı Frekansı Bloğu
        freq_layout = QVBoxLayout()
        self.lbl_freq = QLabel("Taşıyıcı Frekansı (MHz):")
        self.lbl_freq.setStyleSheet("font-size: 17px; font-weight: bold; color: #e0e0e0;")
        freq_layout.addWidget(self.lbl_freq)

        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(70.0, 6000.0)
        self.freq_input.setValue(2400.0)
        self.freq_input.setDecimals(2)
        self.freq_input.setSingleStep(1.0)
        self.freq_input.setStyleSheet("""
            QDoubleSpinBox {
                font-size: 20px; 
                font-weight: bold; 
                background-color: #333333; 
                color: #00e676; 
                padding: 6px; 
                border-radius: 4px;
            }
        """)
        self.freq_input.valueChanged.connect(self.freq_changed.emit)
        freq_layout.addWidget(self.freq_input)

        # Bant Genişliği Bloğu
        bw_layout = QVBoxLayout()
        self.lbl_bw = QLabel("Bant Genişliği (MHz):")
        self.lbl_bw.setStyleSheet("font-size: 17px; font-weight: bold; color: #e0e0e0;")
        bw_layout.addWidget(self.lbl_bw)

        self.combo_bw = QComboBox()
        self.combo_bw.addItems(["1.0", "2.0", "2.4", "4.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0", "56.0", "61.44"])
        self.combo_bw.setCurrentText("2.4")
        self.combo_bw.setStyleSheet("font-size: 16px; padding: 4px; background-color: #333333; color: white;")
        self.combo_bw.currentTextChanged.connect(self.bw_changed.emit)
        bw_layout.addWidget(self.combo_bw)

        row1_layout.addLayout(freq_layout)
        row1_layout.addLayout(bw_layout)
        
        layout.addLayout(row1_layout)

        # 2. Satır: Anten Portu ve Gain (Yan Yana)
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)

        # Port Bloğu (Sola dayalı, dar alan)
        port_layout = QVBoxLayout()
        self.lbl_antenna = QLabel("Port:")
        self.lbl_antenna.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
        port_layout.addWidget(self.lbl_antenna)

        self.combo_antenna = QComboBox()
        self.combo_antenna.addItems(["TX/RX", "RX2"])
        self.combo_antenna.setCurrentText("TX/RX")
        self.combo_antenna.setStyleSheet("font-size: 14px; padding: 4px; background-color: #333333; color: white;")
        self.combo_antenna.setMaximumWidth(80)  # Çok kısa tut
        self.combo_antenna.currentTextChanged.connect(self.antenna_changed.emit)
        port_layout.addWidget(self.combo_antenna)
        
        row2_layout.addLayout(port_layout, stretch=0)

        # Güç Seviyesi (Gain) Bloğu
        gain_layout = QVBoxLayout()
        self.lbl_gain = QLabel("Güç Seviyesi / Gain (dB):")
        self.lbl_gain.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
        gain_layout.addWidget(self.lbl_gain)

        gain_sub_layout = QHBoxLayout()
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 76)
        self.gain_slider.setValue(40)
        self.gain_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #444444; height: 10px; background: #2a2a2a; border-radius: 5px; }
            QSlider::handle:horizontal { background: #e0e0e0; border: 2px solid #555555; width: 22px; height: 22px; margin: -6px 0; border-radius: 11px; }
            QSlider::handle:horizontal:hover { background: #ffffff; border: 2px solid #2e7d32; }
        """)
        
        self.gain_label = QLabel("40 dB")
        self.gain_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        
        gain_sub_layout.addWidget(self.gain_slider, stretch=5)
        gain_sub_layout.addWidget(self.gain_label, stretch=1)
        gain_layout.addLayout(gain_sub_layout)

        row2_layout.addLayout(gain_layout, stretch=1)
        
        layout.addLayout(row2_layout)

        # Üst kısımla arayı biraz açalım
        layout.addSpacing(10)

        # Ana Aksiyon Butonları (Yan Yana)
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        # Başlat/Durdur Butonu
        self.toggle_btn = QPushButton("Sinyal Alımını Başlat")
        self.toggle_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")
        self.toggle_btn.clicked.connect(self.toggle_capture.emit)
        action_row.addWidget(self.toggle_btn)

        # TX Butonu
        self.tx_btn = QPushButton("TX / KARIŞTIRMA")
        self.tx_btn.setStyleSheet("background-color: #00bcd4; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")
        self.tx_btn.clicked.connect(self.open_tx_dialog.emit)
        action_row.addWidget(self.tx_btn)

        layout.addLayout(action_row)

        # Butonlarla alt kısım arasını biraz açalım
        layout.addSpacing(5)

        layout.addSpacing(5)

        # --- BANT TARAMA / SİNYAL TESPİTİ (şartname 5.1.1) ---
        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)

        self.scan_btn = QPushButton("BANT TARA (Sinyal Bul)")
        self.scan_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 13px; height: 40px; border-radius: 6px;")
        self.scan_btn.clicked.connect(self.scan_toggled.emit)
        scan_row.addWidget(self.scan_btn, stretch=3)

        self.scan_start = QDoubleSpinBox()
        self.scan_start.setRange(70.0, 6000.0); self.scan_start.setValue(400.0); self.scan_start.setDecimals(1)
        self.scan_stop = QDoubleSpinBox()
        self.scan_stop.setRange(70.0, 6000.0); self.scan_stop.setValue(2500.0); self.scan_stop.setDecimals(1)
        for sp in (self.scan_start, self.scan_stop):
            sp.setStyleSheet("font-size: 14px; padding: 4px; background-color: #333333; color: white; border-radius: 4px;")
        
        _dash = QLabel("-")
        _dash.setStyleSheet("font-size: 16px; color: #888;")
        
        scan_row.addWidget(self.scan_start, stretch=2)
        scan_row.addWidget(_dash)
        scan_row.addWidget(self.scan_stop, stretch=2)
        
        layout.addLayout(scan_row)

        # Tarama HASSASİYETİ (CFAR yerel-belirginlik eşiği). Güçlü bir sinyalin yükselttiği düz gürültü
        # tabanı yalancı tespit üretir; bu sürgü "gerçek bir sinyal, çevresini kaç dB aşmalı" der.
        # SOL=Hassas (zayıf sinyali de bulur, gürültü artabilir) — SAĞ=Seçici (sadece net sinyaller).
        sens_row = QHBoxLayout()
        sens_row.setSpacing(8)
        self.lbl_sens_cap = QLabel("Tarama Hassasiyeti:")
        self.lbl_sens_cap.setStyleSheet("font-size: 13px; font-weight: bold; color: #ce93d8;")
        sens_row.addWidget(self.lbl_sens_cap)
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(6, 22)      # dB; düşük=hassas, yüksek=seçici
        self.sens_slider.setValue(12)         # varsayılan = SCAN_PROMINENCE_DB (dengeli, temiz gelir)
        self.sens_slider.setToolTip(
            "Bir tespit, ÇEVRESİNDEKİ gürültü tabanını en az bu kadar (dB) aşmalı.\n"
            "Sol = Hassas (zayıf sinyalleri de bulur, gürültü artabilir)\n"
            "Sağ = Seçici (sadece net/güçlü sinyaller — kalabalık ortamda temiz liste)")
        self.sens_slider.valueChanged.connect(self._on_sens_changed)
        sens_row.addWidget(self.sens_slider, stretch=3)
        self.lbl_sens_val = QLabel("Dengeli (12 dB)")
        self.lbl_sens_val.setStyleSheet("font-size: 12px; color: #b0b0b0; min-width: 110px;")
        sens_row.addWidget(self.lbl_sens_val)
        layout.addLayout(sens_row)

        # NOT: "Tespit Edilen Sinyaller" listesi ALT SATIRDAKİ DetectionPanel'e taşındı
        # (ui/widgets/detection_panel.py) — orada daha geniş, ilk-görülme saatli, çift-tıkla-tune'lu
        # ve CSV'ye aktarılabilir. Burada yalnızca tarama BUTONU ve aralık kutuları kalır.

        layout.addStretch(1)

    def on_gain_changed(self, value):
        self.gain_label.setText(f"{value} dB")
        self.gain_changed.emit(value)

    def _on_sens_changed(self, value):
        tier = "Seçici" if value >= 15 else ("Hassas" if value <= 9 else "Dengeli")
        self.lbl_sens_val.setText(f"{tier} ({value} dB)")
        self.scan_sensitivity_changed.emit(value)

    def set_capturing_state(self, is_capturing: bool):
        if is_capturing:
            self.combo_bw.setEnabled(False)
            self.toggle_btn.setText("Durdur")
            self.toggle_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")
        else:
            self.combo_bw.setEnabled(True)
            self.toggle_btn.setText("Sinyal Alımını Başlat")
            self.toggle_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")

    def set_scanning_state(self, scanning: bool):
        if scanning:
            self.scan_btn.setText("TARAMAYI DURDUR")
            self.scan_btn.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; font-size: 16px; height: 40px; border-radius: 6px;")
        else:
            self.scan_btn.setText("BANT TARA (Sinyalleri Bul)")
            self.scan_btn.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; font-size: 16px; height: 40px; border-radius: 6px;")

