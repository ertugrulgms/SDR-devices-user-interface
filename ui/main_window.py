import time
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QTextEdit, QDialog, QScrollArea)
from PyQt6.QtCore import Qt, QDateTime
import pyqtgraph as pg

from backend.sdr_worker import SDRWorker
from ui.widgets import (ControlPanel, AnalysisPanel, SpectrumWidget,
                        DFPanel, PPIWidget, TxDialog, DetectionPanel, ChatPanel)

FFT_POINTS = 2048
WATERFALL_HISTORY = 100

def _to_dms(deg_val: float) -> str:
    d = int(deg_val)
    m = int((deg_val - d) * 60)
    return f"{d}° {abs(m):02d}'"

class SDRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR EH Kontrol Paneli")
        self.is_capturing = False
        self.start_time = time.time()

        self.waterfall_data = np.full((WATERFALL_HISTORY, FFT_POINTS), -100.0)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) 
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # ----------------------------------------------------
        # SOL SÜTUN
        # ----------------------------------------------------
        self.control_panel = ControlPanel()
        self.analysis_panel = AnalysisPanel()

        # YALNIZCA control_panel kaydırılabilir sarmalayıcıya alınır (uzun içerik ekrana sığsın).
        # analysis_panel (SİNYAL ANALİZİ + DEMODÜLE SES/VERİ ÇIKTISI dahil) scroll DIŞINDA kalır ->
        # HER ZAMAN görünür (kaydırma gerektirmez). Önceki sürümde tüm sol sütun scroll içindeydi ve
        # en alttaki ses çıktısı kaydırma çizgisinin altında kalıp görünmüyordu.
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidget(self.control_panel)
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.control_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.control_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self.left_column = QWidget()
        left_layout = QVBoxLayout(self.left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(self.control_scroll, stretch=6)
        left_layout.addWidget(self.analysis_panel, stretch=3)   # scroll DIŞI -> ses çıktısı hep görünür

        # ----------------------------------------------------
        # SAĞ SÜTUN
        # ----------------------------------------------------
        self.right_column = QWidget()
        right_layout = QVBoxLayout(self.right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.spectrum_widget = SpectrumWidget()
        self.df_panel = DFPanel()
        # SAHA SOHBETİ: DF panelinin (kalibrasyon kutusunun) SAĞINA, radarın üstüne yerleşir.
        # Geniş stretch (3) ile eski boşluğu doldurup sola doğru uzanır.
        self.chat_panel = ChatPanel()
        self.df_panel.layout().addWidget(self.chat_panel, stretch=3)

        right_layout.addWidget(self.spectrum_widget, stretch=8)
        right_layout.addWidget(self.df_panel, stretch=2)

        # ----------------------------------------------------
        # ANA EKRAN DÜZEN BİRLEŞTİRME
        # ----------------------------------------------------
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.left_column, stretch=30)
        top_layout.addWidget(self.right_column, stretch=70)
        
        # ----------------------------------------------------
        # ALT SÜTUN (Log ve Radar)
        # ----------------------------------------------------
        bottom_layout = QHBoxLayout()
        
        # Log Ekranı
        self.log_container = QWidget()
        log_layout = QVBoxLayout(self.log_container)
        # Alt satır artık ÜÇ bölmeli: [loglar | tespit edilen sinyaller | radar]. Loglar yatayda
        # yarıya indi (stretch=1 vs tespit paneli stretch=1) -> eski 113px sağ boşluğa gerek kalmadı.
        # ESKİ DEĞER (geri dönüş için): setContentsMargins(0, 5, 113, 0)
        log_layout.setContentsMargins(0, 5, 6, 0)
        
        from PyQt6.QtWidgets import QLabel
        self.lbl_log_title = QLabel("SİSTEM LOGLARI VE UYARILAR")
        self.lbl_log_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        log_layout.addWidget(self.lbl_log_title)
        
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Sistem logları ve uyarılar burada görünecek...")
        self.log_panel.setStyleSheet("font-size: 14px; font-family: monospace; font-weight: bold; background-color: #1e1e1e; color: #e0e0e0; border: 1px solid #333333;")
        log_layout.addWidget(self.log_panel)
        
        self.ppi_widget = PPIWidget()
        # Radar 400->360: kare radar alt satır yüksekliğini belirliyor; küçültünce alt satır düşer,
        # üst satır genişler, sol-üst kontrol paneli scroll'u kaybolur. (Log değil, RADAR sebepti.)
        # ESKİ: setFixedWidth(400) ; setMaximumHeight(400)
        self.ppi_widget.setFixedWidth(360)
        self.ppi_widget.setMaximumHeight(360)

        # TESPİT EDİLEN SİNYALLER (5.1.1) — alt satırın ORTA bölmesi. Kontrol panelinden buraya taşındı:
        # tarama sonuçları geniş/okunaklı görünür, çift tıkla o frekansa tune olunur, CSV'ye aktarılır.
        self.detection_panel = DetectionPanel()

        # Alt satır: [SİSTEM LOGLARI | TESPİT EDİLEN SİNYALLER | RADAR]
        bottom_layout.addWidget(self.log_container, stretch=1)
        bottom_layout.addWidget(self.detection_panel, stretch=1)
        bottom_layout.addWidget(self.ppi_widget, stretch=0)
        bottom_layout.addSpacing(75)   # radar ~2cm SOLA kaydırıldı (sağına boşluk)

        main_layout.addLayout(top_layout, stretch=8)
        main_layout.addLayout(bottom_layout, stretch=2)   # spektrum/paneller baskın; PPI makul

        # Backend Worker Thread
        self.worker = SDRWorker()
        self.worker.data_ready.connect(self.update_gui_from_worker)
        self.worker.log_signal.connect(self.add_log)
        # SAHA SOHBETİ: giden mesaj -> worker (UDP dağıt); gelen mesaj -> panele ekle
        self.chat_panel.send_requested.connect(self.worker.send_chat)
        self.worker.chat_signal.connect(self._on_chat_message)

        self.setup_connections()
        self.add_log("Arayüz başarıyla başlatıldı. Cihaz bağlantısı bekleniyor...")

    def _on_chat_message(self, msg):
        """Worker'dan gelen sohbet mesajını panele ekle (aux veya merkez)."""
        self.chat_panel.add_message(msg.get("from", "?"), msg.get("text", ""), msg.get("ts"))

    def closeEvent(self, event):
        if getattr(self.worker, '_is_running', False):
            self.add_log("Uygulama kapatılıyor: SDR Worker Thread güvenli şekilde durduruluyor...")
            self.worker.stop()
        event.accept()

    def setup_connections(self):
        # Control Panel Sinyalleri
        self.control_panel.freq_changed.connect(self.log_freq_change)
        self.control_panel.gain_changed.connect(self.log_gain_change)
        self.control_panel.bw_changed.connect(self.log_bw_change)
        self.control_panel.antenna_changed.connect(self.log_antenna_change)
        self.control_panel.toggle_capture.connect(self.handle_capture_toggle)
        self.control_panel.open_tx_dialog.connect(self.open_tx_dialog)
        self.control_panel.scan_toggled.connect(self.handle_scan_toggle)
        self.control_panel.scan_sensitivity_changed.connect(self.worker.set_scan_sensitivity)
        # Tespit paneli: çift tıkla -> o frekansa tune;  dışa aktarma -> log
        self.detection_panel.tune_requested.connect(self.tune_to_detection)
        self.detection_panel.exported.connect(
            lambda p: self.add_log(f"📄 Tespit listesi dışa aktarıldı: {p}"))

        # DF Panel Sinyalleri
        self.df_panel.mode_changed.connect(self.toggle_df_mode)
        self.df_panel.manual_angle_changed.connect(self.manual_angle_changed)
        self.df_panel.node_positions_changed.connect(self.apply_node_positions)
        self.df_panel.calibration_toggled.connect(self.handle_df_calibration_toggle)
        self.df_panel.debug_iq_clicked.connect(self.handle_debug_iq_clicked)
        # Kayıtlı yardımcı düğüm konumlarını (df_nodes.json) panele önceden yükle
        self.df_panel.set_node_positions(self.worker.get_aux_node_positions())

        # Ses (spec 5.1.3): demod modu + Dinle/Sustur + Sayısal Çöz
        self.analysis_panel.audio_mode_changed.connect(self.worker.set_audio_mode)
        self.analysis_panel.audio_listen_toggled.connect(self.handle_audio_listen)
        self.analysis_panel.audio_digital_toggled.connect(self.handle_audio_digital)

    def handle_audio_digital(self, enable: bool):
        """Sayısal Çöz butonu. DSD-FME yoksa da 4FSK tespiti çalışır; durumu bildir.
        Şifre anahtarı alanı doluysa (ŞİFRELİ yayın) o anahtarla çözer."""
        key = self.analysis_panel.get_decrypt_key() if enable else None
        info = self.worker.set_digital_decode(enable, key=key or None)
        if enable and not info.get("enabled"):
            self.analysis_panel.btn_digital.blockSignals(True)
            self.analysis_panel.btn_digital.setChecked(False)
            self.analysis_panel.btn_digital.setText("📻 Sayısal Çöz")
            self.analysis_panel.btn_digital.blockSignals(False)
            self.analysis_panel.update_audio_status("Sayısal çözme başlatılamadı (donanım yok)")
        elif enable and not info.get("dsd_available"):
            self.analysis_panel.update_field("digital", "Sayısal çözme: DSD-FME kurulu değil — 4FSK/C4FM tespiti aktif")

    def handle_audio_listen(self, listen: bool):
        """Dinle/Sustur butonu. Başlatma başarısızsa butonu geri al ve durumu bildir."""
        ok = self.worker.set_audio_listen(listen)
        if listen and not ok:
            # Başlatılamadı (donanım yok / ses aygıtı yok) -> butonu geri çevir
            self.analysis_panel.btn_listen.blockSignals(True)
            self.analysis_panel.btn_listen.setChecked(False)
            self.analysis_panel.btn_listen.setText("🔊 Dinle")
            self.analysis_panel.btn_listen.blockSignals(False)
            self.analysis_panel.update_audio_status("Dinlenemedi (donanım/ses aygıtı yok)")
        elif listen:
            self.analysis_panel.update_audio_status(f"{self.worker.audio_mode} dinleniyor")
        else:
            self.analysis_panel.update_audio_status("Susturuldu")

    def open_tx_dialog(self):
        if getattr(self.worker, 'tx_active', False):
            self.handle_tx_stop_clicked()
            return
            
        dialog = TxDialog(self)
        dialog.tx_signal_started.connect(self.handle_tx_started)
        # CTCSS OTOMATİK-TAŞIMA (5.2.3): dinlerken tespit edilen hedef CTCSS tonunu aldatma sekmesinde
        # otomatik seç -> operatör tahmin etmez, hedefin ton-squelch'i garanti açılır.
        ct = float(getattr(self, "_last_ctcss_hz", 0.0) or 0.0)
        if ct > 0 and hasattr(dialog, "cmb_ctcss"):
            for i in range(dialog.cmb_ctcss.count()):
                txt = dialog.cmb_ctcss.itemText(i)
                if txt and txt[0].isdigit() and abs(float(txt.split()[0]) - ct) < 0.3:
                    dialog.cmb_ctcss.setCurrentIndex(i)
                    break
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if hasattr(dialog, 'tx_payload'):
                payload = dialog.tx_payload
                # Karıştırma frekansı TX ekranından geldiyse ana panelin frekans göstergesini
                # eşitle (kullanıcı orada da görsün); asıl geçişi worker.trigger_tx yapar.
                tx_freq = payload.get("tx_freq_mhz")
                if tx_freq:
                    self.control_panel.freq_input.blockSignals(True)
                    self.control_panel.freq_input.setValue(float(tx_freq))
                    self.control_panel.freq_input.blockSignals(False)

                # Karıştırma BANT GENİŞLİĞİ TX ekranından geldiyse: örnekleme hızını değiştirir
                # (donanım yeni hızda başlatılır). Ana panel combo'sunu eşitle + start ÖNCESİ uygula
                # ki start yeni hızda init etsin (yeniden-başlatma yarışı olmasın).
                tx_bw = payload.get("tx_bw_mhz")
                tx_bw_str = payload.get("tx_bw_str")
                if tx_bw:
                    if tx_bw_str:
                        self.control_panel.combo_bw.blockSignals(True)
                        self.control_panel.combo_bw.setCurrentText(tx_bw_str)
                        self.control_panel.combo_bw.blockSignals(False)
                    self.worker.set_bandwidth(float(tx_bw))

                if not self.is_capturing:
                    self.worker.set_frequency(float(tx_freq) if tx_freq
                                              else self.control_panel.freq_input.value())
                    self.worker.set_gain(self.control_panel.gain_slider.value())
                    self.worker.start()

                self.worker.trigger_tx(payload)

    def handle_tx_started(self, tx_info_msg):
        self.add_log(f"⚠️ [TX AKTİF] {tx_info_msg}")
        self.control_panel.tx_btn.setText("TX DURDUR (KARIŞTIRMA AKTİF)")
        self.control_panel.tx_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")

    def handle_tx_stop_clicked(self):
        self.worker.stop_tx()
        self.add_log("⏹ [TX DURDU] Operatör TX yayınını sonlandırdı.")
        self.control_panel.tx_btn.setText("TX / KARIŞTIRMA & ALDATMA MODÜLÜ")
        self.control_panel.tx_btn.setStyleSheet("background-color: #00bcd4; color: white; font-weight: bold; font-size: 14px; height: 40px; border-radius: 6px;")
        
        if not self.is_capturing:
            self.worker.stop()
            self.spectrum_widget.signal_line.setData([], [])
            self.waterfall_data.fill(-100.0)
            self.spectrum_widget.update_waterfall(self.waterfall_data.T)

    def handle_df_calibration_toggle(self, is_checked: bool, reference_deg: float):
        if is_checked:
            self.worker.start_df_calibration(reference_deg)
        else:
            self.worker.stop_df_calibration()

    def handle_debug_iq_clicked(self):
        if hasattr(self, 'worker'):
            self.worker.dump_raw_iq()

    def tune_to_detection(self, freq_mhz: float):
        """Tespit listesinde ÇİFT TIKLANAN sinyalin frekansına tune ol (5.1.1 -> 5.1.2 geçişi).
        Frekans kutusunu ayarlamak yeterli: valueChanged -> freq_changed -> worker.set_frequency zinciri
        zaten bağlı (elle yazma/yazım hatası olmadan hızlı geçiş)."""
        self.control_panel.freq_input.setValue(float(freq_mhz))
        self.add_log(f"🎯 Tespit edilen sinyale tune olundu: {freq_mhz:.3f} MHz")

    def apply_node_positions(self, polar_list):
        """DF panelinden gelen yardımcı düğüm konumlarını (mesafe m + pusula açısı°) worker'a uygula.
        Ana cihaz (0,0) sabittir; konum belirleme (5.1.5) bu konumlara göre üçgenler."""
        if hasattr(self, "worker"):
            res = self.worker.set_aux_node_positions(polar_list)
            self.add_log("Yardımcı düğüm konumları uygulandı: " + ", ".join(res.get("applied", [])))

    def toggle_df_mode(self, is_auto: bool):
        if self.is_capturing:
            self.worker.set_df_mode(auto=is_auto)
            
        if not is_auto:
            self.add_log("Yön Bulma modu MANUEL olarak değiştirildi.")
            angles = (self.df_panel.spin_dev1.value(), self.df_panel.spin_dev2.value(), self.df_panel.spin_dev3.value())
            self.worker.set_manual_angles(angles)
        else:
            self.add_log("Yön Bulma modu OTONOM olarak değiştirildi.")

    def manual_angle_changed(self, idx: int, val: float):
        if not self.df_panel.rbtn_auto.isChecked() and self.is_capturing:
            angles = (self.df_panel.spin_dev1.value(), self.df_panel.spin_dev2.value(), self.df_panel.spin_dev3.value())
            self.worker.set_manual_angles(angles)

    def add_log(self, text):
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss") 
        self.log_panel.append(f"<div style='margin-bottom: 6px;'>[{current_time}] {text}</div>")

    def log_gain_change(self, value):
        self.add_log(f"Güç Seviyesi (Gain) değiştirildi: {value} dB")
        if hasattr(self, 'worker'):
            self.worker.set_gain(value)

    def log_bw_change(self, text):
        self.add_log(f"Bant Genişliği seçildi: {text} MHz")
        if hasattr(self, 'worker'):
            self.worker.set_bandwidth(float(text))

    def log_antenna_change(self, text):
        self.add_log(f"RX Anten Portu seçildi: {text}")
        if hasattr(self, 'worker'):
            self.worker.set_antenna(text)

    def log_freq_change(self, value):
        self.add_log(f"Taşıyıcı Frekansı değiştirildi: {value} MHz")
        if self.is_capturing:
            self.worker.set_frequency(value)
        else:
            self.spectrum_widget.plot_widget.setXRange(value - 5, value + 5, padding=0)

    def handle_scan_toggle(self):
        """BANT TARA aç/kapa (sinyal tespiti, 5.1.1). Alım kapalıysa önce başlatır."""
        if getattr(self.worker, "scan_active", False):
            self.worker.stop_scan_rf()
            self.control_panel.set_scanning_state(False)
            return
        # Tarama gerçek donanım gerektirir; alım kapalıysa başlat
        if not self.is_capturing:
            self.worker.set_gain(self.control_panel.gain_slider.value())
            self.worker.set_bandwidth(float(self.control_panel.combo_bw.currentText()))
            self.worker.set_antenna(self.control_panel.combo_antenna.currentText())
            self.worker.start()
            self.is_capturing = True
            self.control_panel.set_capturing_state(True)
        self.worker.start_scan_rf(self.control_panel.scan_start.value(),
                                  self.control_panel.scan_stop.value())
        self.control_panel.set_scanning_state(True)

    def handle_capture_toggle(self):
        if not self.is_capturing:
            self.worker.set_frequency(self.control_panel.freq_input.value())
            self.worker.set_gain(self.control_panel.gain_slider.value())
            self.worker.set_bandwidth(float(self.control_panel.combo_bw.currentText()))
            self.worker.set_antenna(self.control_panel.combo_antenna.currentText())

            is_auto = self.df_panel.rbtn_auto.isChecked()
            self.worker.set_df_mode(auto=is_auto)
            if not is_auto:
                self.worker.set_manual_angles(
                    (self.df_panel.spin_dev1.value(), self.df_panel.spin_dev2.value(), self.df_panel.spin_dev3.value()))

            self.worker.start()
            self.is_capturing = True
            self.control_panel.set_capturing_state(True)
            
            for field_id in self.analysis_panel.analysis_data:
                self.analysis_panel.update_field(field_id, "Ölçülüyor...")
            self.analysis_panel.update_audio_status("Sinyal işleniyor...")
        else:
            self.worker.stop()
            self.is_capturing = False
            self.control_panel.set_capturing_state(False)

            # ET DURUMU'nu ve TX butonunu sıfırla: worker durunca payload gelmez, aksi halde
            # "ET DURUMU: AKTİF" ekranda donar ve tekrar başlatınca durum karışır.
            # (control_panel'de ET status widget'i yoksa güvenle atla -> AttributeError/çökme yok.)
            if hasattr(self.control_panel, "lbl_et_status"):
                self.control_panel.lbl_et_status.setText("ET DURUMU: PASİF")
            self.control_panel.tx_btn.setText("TX / KARIŞTIRMA & ALDATMA MODÜLÜ")
            self.control_panel.tx_btn.setStyleSheet(
                "background-color: #00bcd4; color: white; font-weight: bold; font-size: 17px; height: 45px; border-radius: 6px;")

            for field_id in self.analysis_panel.analysis_data:
                self.analysis_panel.update_field(field_id, "Durduruldu")
            self.analysis_panel.update_audio_status("Sinyal Bekleniyor...")
            
            self.df_panel.update_angles([0, 0, 0], lambda x: "--° --'")
            self.ppi_widget.update_ppi({})   # PPI'yı boş duruma al
            
            self.analysis_panel.update_audio_plot(([], []))
            self.waterfall_data.fill(-100.0)
            self.spectrum_widget.update_waterfall(self.waterfall_data.T)

    def update_gui_from_worker(self, payload: dict):
        try:
            x_freqs = payload["x_freqs"]
            fft_dbm = payload["fft_dbm"]
            start_freq, end_freq = payload["freq_bounds"]

            self.spectrum_widget.update_spectrum(x_freqs, fft_dbm)

            # BANT TARAMA / SİNYAL TESPİTİ (5.1.1): tespit listesini + durumu güncelle
            self.detection_panel.update_detections(payload.get("scan_detections", []))
            self.control_panel.set_scanning_state(payload.get("scan_active", False))

            peak_pwr = np.max(fft_dbm)
            current_noise = np.median(fft_dbm)
            if not hasattr(self, 'noise_floor_avg'):
                self.noise_floor_avg = current_noise
            else:
                self.noise_floor_avg = 0.9 * self.noise_floor_avg + 0.1 * current_noise

            self.spectrum_widget.plot_widget.setYRange(self.noise_floor_avg - 10, max(self.noise_floor_avg + 15, peak_pwr + 8))

            # Waterfall zaman-frekans gösterir; her satır anlık kalmalı (max-hold değil, yoksa
            # yatay çizgilere dönüşür ve zaman bilgisi kaybolur).
            self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
            self.waterfall_data[-1, :] = fft_dbm
        
            self.spectrum_widget.waterfall_image.setRect(pg.QtCore.QRectF(start_freq, 0, payload["bandwidth_mhz"], WATERFALL_HISTORY))
            self.spectrum_widget.waterfall_widget.setXRange(start_freq, end_freq, padding=0)
            self.spectrum_widget.waterfall_image.setLevels([self.noise_floor_avg - 2, self.noise_floor_avg + 25])
            self.spectrum_widget.update_waterfall(self.waterfall_data.T)

            self.analysis_panel.update_audio_plot((np.linspace(0, 1, 100), payload["audio_y"]))

            # SİNYAL İZLEME/TAKİP (5.1.3): süreklilik + parametre geçmişi
            self.analysis_panel.update_monitor(payload.get("monitor"))

            # SAYISAL SES (5.1.3): 4FSK/C4FM tespit özeti
            self.analysis_panel.update_digital_voice(payload.get("digital_voice"))
            # SAYISAL VERİ (5.1.3): DSD-FME'den çözülen çağrı/sync/renk-kodu/TG satırları
            self.analysis_panel.update_digital_data(payload.get("digital_data"))

            if self.is_capturing:
                # KONSOLİDE PARAMETRELER (5.1.2): titrek ham değer yerine 3 sn çoğunluk/medyan +
                # güven etiketi. Kategorik: "değer [KESİN/olası/zayıf %XX]". Sayısal: medyan.
                pc = payload.get("params_consolidated", {}) or {}
                def _cat(field, fallback):
                    c = pc.get(field)
                    if c and "value" in c:
                        return f"{c['value']}  [{c['tier']} %{int(c['confidence'] * 100)}]"
                    return fallback
                def _num(field):
                    c = pc.get(field)
                    return c["value"] if (c and "value" in c) else None

                occ_hz = _num("occupied_bw_hz") or payload.get("occupied_bw_hz", 0.0)
                if occ_hz and occ_hz > 0:
                    occ_str = f"{occ_hz/1e6:.3f} MHz" if occ_hz >= 1e5 else f"{occ_hz/1e3:.1f} kHz"
                    self.analysis_panel.update_field("bandwidth", f"{occ_str} (işgal) / {payload['bandwidth_mhz']:.1f} MHz aralık")
                else:
                    self.analysis_panel.update_field("bandwidth", f"{payload['bandwidth_mhz']:.1f} MHz aralık (sinyal yok)")
                # Güç birimi kalibrasyon durumuna göre DÜRÜST etiketlenir: tek-nokta kalibrasyon
                # yapıldıysa değer mutlak dBm'e yakınsar ("dBm kal."), aksi halde bağıl dBFS'tir.
                _cal = payload.get("power_cal_offset_db", 0.0) or 0.0
                _unit = "dBm (kal.)" if abs(_cal) > 1e-6 else "dBFS (bağıl)"
                self.analysis_panel.update_field("power", f"{np.max(fft_dbm):.1f} {_unit}")

                sig_class = payload.get("signal_class", "Ölçülüyor...")
                flat = payload.get("spectral_flatness")
                snr = payload.get("snr_db")
                if flat is not None and snr is not None:
                    self.analysis_panel.update_field("signal_type", f"{sig_class}  (düzlük={flat}, SNR≈{snr} dB)")
                else:
                    self.analysis_panel.update_field("signal_type", sig_class)

                # CTCSS (5.2.3): tespit edilen alt-ses ton-squelch'i -> protokol alanında göster +
                # sakla (TX aldatma sekmesine otomatik taşınır). Analog voice telsizin "protokolü".
                self._last_ctcss_hz = float(payload.get("ctcss_hz", 0.0) or 0.0)

                # Modülasyon (5.1.2): KONSOLİDE (çoğunluk oyu + güven etiketi) -> titremez, kararlı.
                self.analysis_panel.update_field(
                    "modulation", _cat("modulation", payload.get("modulation", "Ölçülüyor...")))

                # Protokol: konsolide + (analog FM'de) tespit edilen squelch (CTCSS tonu / DCS kodu)
                proto = _cat("protocol", payload.get("protocol", "Ölçülüyor..."))
                if self._last_ctcss_hz > 0:
                    proto = f"{proto}  |  🔉 CTCSS {self._last_ctcss_hz:.1f} Hz"
                elif payload.get("squelch_type") == "DCS":
                    proto = f"{proto}  |  🔉 DCS (kod yakalandı → aldatmaya hazır)"

                # Çoklama Türü (5.1.2): OFDM / FDMA / DSSS-CDMA / TDMA / Tek Taşıyıcı
                self.analysis_panel.update_field(
                    "multiplex", _cat("multiplex", payload.get("multiplex", "Ölçülüyor...")))

                # EKKT Tedbiri (5.1.2): FHSS / DSSS / Yok
                self.analysis_panel.update_field(
                    "ekkt", _cat("ekkt", payload.get("ekkt", "Ölçülüyor...")))

                # Protokol Türü (5.1.2): bant planı + olası protokol + (analog FM'de) CTCSS tonu (5.2.3)
                self.analysis_panel.update_field("protocol", proto)

                # Taşıyıcı Frekansı (5.1.2): KONSOLİDE medyan (kararlı), yoksa ham tepe
                carrier = _num("carrier_mhz")
                if carrier is None:
                    carrier = payload.get("carrier_mhz")
                self.analysis_panel.update_field("carrier",
                    f"{carrier:.4f} MHz" if carrier is not None else "Sinyal yok / ölçülemedi")

                # Diğer Sayısal Özellikler (5.1.2): sembol/baud hızı (dijitalde) — C4FM/4FSK özeti
                # ayrıca update_digital_voice ile "digital" alanına yazılır.
                baud = _num("symbol_rate_hz") or payload.get("symbol_rate_hz", 0.0) or 0.0
                if baud > 0:
                    self.analysis_panel.update_field("digital", f"Sembol hızı ≈ {baud/1e3:.1f} kBd")

            df_rms_deg = payload.get("df_rms_deg")
            if df_rms_deg is not None:
                self.df_panel.update_rms(f"RMS Hata: {df_rms_deg}° (N={payload.get('df_sample_count', 0)})")

            # ET DURUMU göstergesi (varsa güncelle; control_panel'de yoksa sessizce atla -> çökme yok).
            # (Önceki sürüm olmayan set_et_status()/lbl_et_status'a erişip her karede AttributeError
            #  fırlatıp arayüzü dolduruyor/donuyordu.)
            if hasattr(self.control_panel, "lbl_et_status"):
                if payload.get("tx_active"):
                    mode_name = payload.get("tx_mode", "AKTİF")
                    self.control_panel.lbl_et_status.setText(f"ET DURUMU: AKTİF — {mode_name}")
                    self.control_panel.lbl_et_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff; background-color: #c62828; padding: 6px; border-radius: 4px; border: 1px solid #ff5252;")
                else:
                    self.control_panel.lbl_et_status.setText("ET DURUMU: PASİF")
                    self.control_panel.lbl_et_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #9e9e9e; padding: 6px; border-radius: 4px; border: 1px solid #444;")
            
            if payload.get("ant_status"):
                self.df_panel.lbl_ant1_status.setText("ANT-1 (Merkez): ESP32 BAĞLI (USB)")
                self.df_panel.lbl_ant1_status.setStyleSheet("font-size: 15px; font-weight: bold; color: #00e676; padding: 2px;")
            else:
                self.df_panel.lbl_ant1_status.setText("ANT-1 (Merkez): BAĞLANTI YOK")
                self.df_panel.lbl_ant1_status.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff5252; padding: 2px;")

            # ANT-2 / ANT-3: uzak ağ düğümlerinin CANLI durumu (bağlı=yeşil, değil=kırmızı).
            # df_remote_nodes registry sırasına göre self-olmayan düğümler; ANT-2->ilk, ANT-3->ikinci.
            _green = "font-size: 15px; font-weight: bold; color: #00e676; padding: 2px;"
            _red = "font-size: 15px; font-weight: bold; color: #ff5252; padding: 2px;"
            remote = payload.get("df_remote_nodes", []) or []
            for lbl, rn in zip((self.df_panel.lbl_ant2_status, self.df_panel.lbl_ant3_status), remote):
                nid = rn.get("id", "NODE")
                if rn.get("connected"):
                    age = rn.get("age_sec")
                    age_str = f" ({age:.1f} sn önce)" if isinstance(age, (int, float)) else ""
                    lbl.setText(f"{nid}: BAĞLI{age_str}")
                    lbl.setStyleSheet(_green)
                else:
                    lbl.setText(f"{nid}: BAĞLANTI YOK")
                    lbl.setStyleSheet(_red)

            angles = payload["angles"]
            self.df_panel.update_angles(angles, _to_dms)
            # CANLI ham anten açısı (enkoder) — DF panelinde akıcı göster (Serial Monitor gibi)
            self.df_panel.update_live_angle(payload.get("encoder_angle_deg"))
            # AUX (NODE-2/3) canlı anten açısı — SDR-2/3 satırında anlık (kerterizin 15 sn gecikmesini beklemez)
            self.df_panel.update_node_live_angles(payload.get("node_live_angles"))

            # PPI radar: gerçek DF payload'ından (düğüm kerterizleri + üçgenleme fix'i) güncelle
            # (payload["encoder_angle_deg"] radarda canlı yön çizgisi olarak çizilir)
            self.ppi_widget.update_ppi(payload)
        except Exception as _slot_exc:
            # GUI slot'unda beklenmedik hata olursa PyQt uygulamayı çökertebilir; yut+logla.
            import traceback
            traceback.print_exc()
        finally:
            # BACKPRESSURE: bu kare işlendi -> worker yeni kare gönderebilir. finally'de olması
            # ŞART: slot ortada hata atsa bile bayrak bırakılmazsa worker bir daha emit etmez
            # ve arayüz kalıcı donar. Böylece hata olsa da akış sürer.
            if getattr(self, 'worker', None) is not None:
                self.worker._gui_ready = True
