from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                             QComboBox, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph as pg

class AnalysisField:
    __slots__ = ("widget", "title", "value")

    def __init__(self, widget: QLabel, title: str, value: str = "Bekleniyor..."):
        self.widget = widget
        self.title = title
        self.value = value

class AnalysisPanel(QWidget):
    # main_window bunlara bağlanır (gerçek ses oynatma kontrolü)
    audio_mode_changed = pyqtSignal(str)      # "FM" / "AM" / "USB" / "LSB"
    audio_listen_toggled = pyqtSignal(bool)   # True=Dinle, False=Sustur
    audio_digital_toggled = pyqtSignal(bool)  # True=Sayısal Çöz (DSD-FME/4FSK), False=kapat

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Başlık
        self.lbl_title = QLabel("SİNYAL ANALİZ SONUÇLARI")
        self.lbl_title.setStyleSheet("font-size: 21px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.lbl_title)
        
        self.carrier_label = QLabel()
        self.bandwidth_label = QLabel()
        self.power_label = QLabel()
        self.signal_type_label = QLabel()
        self.modulation_label = QLabel()
        self.protocol_label = QLabel()
        self.multiplex_label = QLabel()
        self.ekkt_label = QLabel()
        self.digital_label = QLabel()

        for _lbl in (self.carrier_label, self.bandwidth_label, self.power_label, self.signal_type_label,
                     self.modulation_label, self.protocol_label, self.multiplex_label,
                     self.ekkt_label, self.digital_label):
            _lbl.setWordWrap(True)
            _lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #cccccc;")
            layout.addWidget(_lbl)

        self.analysis_data = {
            "carrier": AnalysisField(self.carrier_label, "Taşıyıcı Frekansı: "),
            "bandwidth": AnalysisField(self.bandwidth_label, "Bant Genişliği: "),
            "power": AnalysisField(self.power_label, "Güç Seviyesi (Tepe): "),
            "signal_type": AnalysisField(self.signal_type_label, "Analog/Sayısal Ayrımı: "),
            "modulation": AnalysisField(self.modulation_label, "Modülasyon Türü: "),
            "protocol": AnalysisField(self.protocol_label, "Protokol Türü: "),
            "multiplex": AnalysisField(self.multiplex_label, "Çoklama Türü: "),
            "ekkt": AnalysisField(self.ekkt_label, "EKKT Tedbiri (FHSS/DSSS): "),
            "digital": AnalysisField(self.digital_label, "Diğer Sayısal Özellikler: "),
        }
        
        # İlk değerleri yazdır
        for field in self.analysis_data.values():
            field.widget.setText(f"{field.title} {field.value}")

        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setStyleSheet("background-color: #444444; height: 1px;")
        layout.addWidget(self.line)

        # DEMODÜLE VERİ AKIŞI
        self.lbl_title_demod = QLabel("DEMODÜLE VERİ / SES ÇIKTISI")
        self.lbl_title_demod.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.lbl_title_demod)

        self.lbl_audio_status = QLabel("Demodüle Ses/Veri: Sinyal Bekleniyor...")
        self.lbl_audio_status.setStyleSheet("font-size: 17px; font-weight: bold; color: #ffb74d;")
        self.lbl_audio_status.setWordWrap(True)
        layout.addWidget(self.lbl_audio_status)

        # Ses kontrol satırı: demod modu seçici + Dinle/Sustur
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        lbl_mode = QLabel("Demod:")
        lbl_mode.setStyleSheet("font-size: 15px; color: #cccccc;")
        ctrl_row.addWidget(lbl_mode)

        self.audio_mode_combo = QComboBox()
        self.audio_mode_combo.addItems(["FM", "AM", "USB", "LSB"])
        self.audio_mode_combo.setStyleSheet("font-size: 15px; padding: 3px;")
        self.audio_mode_combo.currentTextChanged.connect(self.audio_mode_changed.emit)
        ctrl_row.addWidget(self.audio_mode_combo)

        self.btn_listen = QPushButton("🔊 Dinle")
        self.btn_listen.setCheckable(True)
        self.btn_listen.setStyleSheet(
            "QPushButton{font-size:15px;font-weight:bold;padding:5px 12px;background:#2e7d32;color:#fff;border-radius:4px;}"
            "QPushButton:checked{background:#c62828;}")
        self.btn_listen.toggled.connect(self._on_listen_toggled)
        ctrl_row.addWidget(self.btn_listen)

        self.btn_digital = QPushButton("📻 Sayısal Çöz")
        self.btn_digital.setCheckable(True)
        self.btn_digital.setStyleSheet(
            "QPushButton{font-size:15px;font-weight:bold;padding:5px 12px;background:#455a64;color:#fff;border-radius:4px;}"
            "QPushButton:checked{background:#6a1b9a;}")
        self.btn_digital.toggled.connect(self._on_digital_toggled)
        ctrl_row.addWidget(self.btn_digital)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setBackground('#121212')
        self.audio_plot.setFixedHeight(50)
        self.audio_plot.hideAxis('bottom')
        self.audio_plot.hideAxis('left')
        self.audio_line = self.audio_plot.plot(pen=pg.mkPen(color='#00e676', width=1.5))
        layout.addWidget(self.audio_plot)

        # SİNYAL İZLEME/TAKİP (spec 5.1.3): süreklilik + parametre geçmişi
        self.lbl_monitor = QLabel("İzleme: —")
        self.lbl_monitor.setStyleSheet("font-size: 15px; font-weight: bold; color: #4fc3f7;")
        self.lbl_monitor.setWordWrap(True)
        layout.addWidget(self.lbl_monitor)

        layout.addStretch(1)

    def update_field(self, field_id: str, value: str):
        if field_id in self.analysis_data:
            field = self.analysis_data[field_id]
            field.value = value
            field.widget.setText(f"{field.title} {field.value}")

    def _on_listen_toggled(self, checked: bool):
        # checked=True -> Dinle aktif; buton metnini güncelle ve sinyali yay
        self.btn_listen.setText("🔇 Sustur" if checked else "🔊 Dinle")
        self.audio_listen_toggled.emit(checked)

    def _on_digital_toggled(self, checked: bool):
        self.btn_digital.setText("📻 Sayısal Dur" if checked else "📻 Sayısal Çöz")
        self.audio_digital_toggled.emit(checked)

    def update_digital_voice(self, dv: dict):
        """4FSK/C4FM tespit özetini 'Diğer Sayısal Özellikler' alanında gösterir."""
        if not dv:
            return
        if dv.get("is_4fsk"):
            baud = dv.get("symbol_rate_hz", 0.0)
            self.update_field("digital", f"C4FM/4FSK sayısal telsiz (≈{baud:.0f} baud) — DSD-FME'ye yönlendiriliyor")
        else:
            self.update_field("digital", "Sayısal çözme aktif — 4FSK/C4FM tespit edilmedi (analog olabilir)")

    def set_audio_device_available(self, available: bool):
        """Ses aygıtı yoksa Dinle butonunu devre dışı bırak (headless/aygıtsız)."""
        self.btn_listen.setEnabled(available)
        if not available:
            self.btn_listen.setToolTip("Ses çıkış aygıtı bulunamadı")

    def update_audio_status(self, status: str):
        self.lbl_audio_status.setText(f"Demodüle Ses/Veri: {status}")

    def update_monitor(self, mon: dict):
        """Sinyal izleme özetini gösterir (worker payload'ından 'monitor')."""
        if not mon or not mon.get("locked"):
            self.lbl_monitor.setText("İzleme: —")
            return
        state = "VAR" if mon.get("present") else "YOK"
        txt = (f"İzleme {mon['freq_mhz']:.3f} MHz: {mon.get('continuity','-')} "
               f"(%{mon.get('uptime_pct',0):.0f} uptime, {state}) | "
               f"kesinti:{mon.get('drops',0)} geri:{mon.get('reacquires',0)} | "
               f"kayma:{mon.get('freq_drift_khz',0):.1f} kHz | "
               f"güç aralığı:{mon.get('power_range_db',0):.1f} dB")
        self.lbl_monitor.setText(txt)

    def update_audio_plot(self, data):
        # data tuple ise x ve y olarak ayır, yoksa direkt bas
        if isinstance(data, tuple) and len(data) == 2:
            self.audio_line.setData(data[0], data[1])
        else:
            self.audio_line.setData(data)
