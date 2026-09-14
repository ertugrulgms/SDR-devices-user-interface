"""SAHA SOHBETİ paneli — aux istasyonları (NODE-2/3) ile merkez arasında UDP tabanlı yazışma.

Telsizsiz koordinasyon için: aux operatörü "446'ya geçtim, tarıyorum" yazar; merkez "kuzeye 30°"
der. Mesajlar zaten var olan 192.168.1.x ağı üzerinden (UDP 5006) gider; merkez hub'dır (backend
sdr_worker _chat_listener_loop). Bu widget yalnızca gösterim + giriş; ağ işi backend'de.
"""
import time

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit,
                             QLineEdit, QPushButton, QSizePolicy)
from PyQt6.QtCore import pyqtSignal


class ChatPanel(QWidget):
    send_requested = pyqtSignal(str)     # kullanıcı mesaj yazıp gönderdi

    def __init__(self, parent=None):
        super().__init__(parent)
        # Yatayda YAYIL (soldaki boşluğu doldur) + makul minimum genişlik.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(340)
        self.init_ui()

    def init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("SAHA SOHBETİ (Aux ↔ Merkez)")
        box.setStyleSheet(
            "QGroupBox{border:2px solid #333;border-radius:8px;margin-top:8px;font-size:14px;"
            "font-weight:bold;color:#00e5ff;background-color:#1e1e1e;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top center;padding:0 5px;}")
        v = QVBoxLayout(box)
        v.setSpacing(4)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            "background-color:#141414;color:#e0e0e0;font-family:monospace;font-size:12px;"
            "border:1px solid #333;border-radius:4px;")
        v.addWidget(self.log, stretch=1)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Mesaj yaz… (Enter ile gönder)")
        self.inp.setStyleSheet("background:#2a2a2a;color:#fff;padding:6px;border:1px solid #555;"
                               "border-radius:4px;font-size:13px;")
        self.inp.returnPressed.connect(self._on_send)
        row.addWidget(self.inp, stretch=1)
        self.btn = QPushButton("Gönder")
        self.btn.setStyleSheet("background:#00838f;color:#fff;font-weight:bold;padding:6px 12px;"
                               "border-radius:4px;font-size:13px;")
        self.btn.clicked.connect(self._on_send)
        row.addWidget(self.btn)
        v.addLayout(row)

        outer.addWidget(box)

    def _on_send(self):
        text = self.inp.text().strip()
        if text:
            self.send_requested.emit(text)
            self.inp.clear()

    def add_message(self, sender: str, text: str, ts: float = None):
        """Gelen/giden mesajı listeye ekle. MERKEZ mesajları farklı renkte (kendi sesin)."""
        tstr = time.strftime("%H:%M:%S", time.localtime(ts or time.time()))
        color = "#00e5ff" if sender == "MERKEZ" else "#00e676"
        safe = (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.log.append(f'<span style="color:#777">[{tstr}]</span> '
                        f'<b style="color:{color}">{sender}:</b> <span style="color:#e0e0e0">{safe}</span>')
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())
