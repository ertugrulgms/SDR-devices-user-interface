"""TESPİT EDİLEN SİNYALLER paneli (şartname 5.1.1 — Sinyal Tespiti).

Bant tarama sırasında gürültü tabanını aşan yayınlar OTOMATİK tespit edilir ve burada listelenir.
Liste KALICIDIR (Max-Hold): sahadaki kaynaklar SIRAYLA yayın yaptığı için, biri susup diğeri
başlasa bile önceki frekans listede kalır -> tur sonunda "şu frekansları bulduk" denebilir.

Özellikler:
  * İLK GÖRÜLME saati  -> kaynakların hangi sırayla açıldığı görülür (sıralı yayın senaryosu).
  * ÇİFT TIKLA -> TUNE -> listedeki frekansa anında geçiş (elle frekans yazma/yazım hatası yok).
  * DIŞA AKTAR (CSV)   -> hakeme/rapora temiz kanıt dosyası.
"""
import os
import csv
import time

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXPORT_DIR = os.path.join(_PROJECT_ROOT, "data")


class DetectionPanel(QWidget):
    tune_requested = pyqtSignal(float)    # çift tıklanan tespitin frekansı (MHz)
    exported = pyqtSignal(str)            # dışa aktarılan dosyanın yolu (log için)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._detections = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # --- Başlık satırı: sayaç + dışa aktar ---
        head = QHBoxLayout()
        head.setSpacing(6)
        self.lbl_title = QLabel("TESPİT EDİLEN SİNYALLER: 0")
        self.lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ce93d8;")
        head.addWidget(self.lbl_title, stretch=1)

        self.btn_export = QPushButton("Dışa Aktar")
        self.btn_export.setToolTip("Tespit listesini CSV olarak data/ klasörüne kaydet")
        self.btn_export.setStyleSheet(
            "QPushButton{font-size:12px;font-weight:bold;padding:3px 10px;background:#6a1b9a;"
            "color:#fff;border-radius:4px;}"
            "QPushButton:disabled{background:#3a3a3a;color:#888;}")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_export.setEnabled(False)
        head.addWidget(self.btn_export, stretch=0)
        layout.addLayout(head)

        # --- Sütun başlıkları (hizalı okuma) ---
        self.lbl_header = QLabel("  İLK GÖR.    FREKANS      GÜÇ    SNR   BANT GEN.")
        self.lbl_header.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #9a80aa; padding-left: 2px;")
        layout.addWidget(self.lbl_header)

        # --- Liste (çift tıkla -> tune) ---
        self.detection_list = QListWidget()
        self.detection_list.setStyleSheet(
            "font-family: monospace; font-size: 13px; background-color: #1a1420; color: #e0d0f0;"
            "border: 1px solid #4a2a5a; border-radius: 4px;")
        self.detection_list.setToolTip("Bir sinyale ÇİFT TIKLA -> cihaz o frekansa tune olur")
        self.detection_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.detection_list, stretch=1)

        self.lbl_hint = QLabel("💡 Bir sinyale çift tıkla → o frekansa tune olur")
        self.lbl_hint.setStyleSheet("font-size: 11px; color: #8a7a95;")
        layout.addWidget(self.lbl_hint)

    # ------------------------------------------------------------------ #
    def update_detections(self, detections: list):
        """Tespit listesini güncelle. detections: [{freq_mhz, power_dbfs, snr_db, bw_mhz, first_ts}]"""
        self._detections = list(detections or [])
        self.lbl_title.setText(f"TESPİT EDİLEN SİNYALLER: {len(self._detections)}")
        self.btn_export.setEnabled(bool(self._detections))

        # Seçili satırı koru (kullanıcı listeyi incelerken kaymasın)
        prev_row = self.detection_list.currentRow()
        self.detection_list.clear()
        for d in self._detections:
            bw = d.get("bw_mhz", 0.0) or 0.0
            bw_str = f"{bw*1000:>5.0f} kHz" if 0 < bw < 1.0 else (f"{bw:>5.1f} MHz" if bw > 0 else "   —   ")
            ts = d.get("first_ts", 0.0) or 0.0
            t_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts > 0 else "  --:--  "
            text = (f"  {t_str}  {d['freq_mhz']:>9.3f} MHz  "
                    f"{d['power_dbfs']:>5.0f} dBFS  {d['snr_db']:>3.0f}  {bw_str}")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, float(d["freq_mhz"]))   # çift tık için frekans
            self.detection_list.addItem(item)
        if 0 <= prev_row < self.detection_list.count():
            self.detection_list.setCurrentRow(prev_row)

    def clear_detections(self):
        self._detections = []
        self.detection_list.clear()
        self.lbl_title.setText("TESPİT EDİLEN SİNYALLER: 0")
        self.btn_export.setEnabled(False)

    # ------------------------------------------------------------------ #
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Çift tıklanan tespitin frekansına tune iste."""
        freq = item.data(Qt.ItemDataRole.UserRole)
        if freq is not None:
            self.tune_requested.emit(float(freq))

    def _on_export(self):
        """Tespit listesini CSV olarak data/ klasörüne kaydeder (hakeme/rapora kanıt)."""
        if not self._detections:
            return
        try:
            os.makedirs(_EXPORT_DIR, exist_ok=True)
            path = os.path.join(_EXPORT_DIR, time.strftime("tespitler_%Y%m%d_%H%M%S.csv"))
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ilk_gorulme", "frekans_mhz", "guc_dbfs", "snr_db", "bant_genisligi_mhz"])
                for d in self._detections:
                    ts = d.get("first_ts", 0.0) or 0.0
                    w.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts > 0 else "",
                        f"{d['freq_mhz']:.3f}", f"{d['power_dbfs']:.1f}",
                        f"{d['snr_db']:.1f}", f"{d.get('bw_mhz', 0.0):.4f}"])
            self.exported.emit(path)
        except OSError as e:
            self.exported.emit(f"HATA: dışa aktarılamadı ({e})")
