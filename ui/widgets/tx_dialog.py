from PyQt6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QPushButton, QTabWidget, QFileDialog)
from PyQt6.QtCore import Qt, QRegularExpression, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QRegularExpressionValidator

class TxDialog(QDialog):
    """TX Karıştırma ve Aldatma Modülü Pop-Up Penceresi"""
    tx_signal_started = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TX / KARIŞTIRMA & ALDATMA MODÜLÜ")
        self.resize(800, 520)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e; 
                color: #ffffff; 
                font-size: 15px;
            }
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: #e0e0e0;
            }
            QComboBox, QLineEdit {
                font-size: 15px; 
                padding: 8px; 
                background-color: #333333; 
                color: #ffffff; 
                border: 1px solid #555555; 
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #252525; }
            QTabBar::tab { 
                background: #333; 
                color: #ccc; 
                padding: 10px 18px; 
                font-size: 15px; 
                font-weight: bold; 
            }
            QTabBar::tab:selected { background: #2e7d32; color: #fff; }
        """)

        double_val = QDoubleValidator(0.0, 1000.0, 2)
        percent_val = QDoubleValidator(0.0, 100.0, 1)

        # Sekme 1: Sürekli Karıştırma
        tab_continuous = QWidget()
        layout_cont = QVBoxLayout(tab_continuous)
        layout_cont.setSpacing(14)
        
        layout_cont.addWidget(QLabel("Karıştırma Tipi:"))
        self.cmb_jam_type = QComboBox()
        self.cmb_jam_type.addItems(["Tekli (Spot Gürültü)", "Çoklu (Çok-Bant Gürültü)",
                                    "Baraj (Barrage Noise)", "Süpürmeli (Frekans Tarama)"])
        layout_cont.addWidget(self.cmb_jam_type)

        # KARIŞTIRMA FREKANSI: operatör doğrudan buradan girer (ana panele gitmeye gerek yok).
        layout_cont.addWidget(QLabel("Karıştırma Frekansı (MHz, 70–6000):"))
        self.txt_jam_freq = QLineEdit("2437")
        self.txt_jam_freq.setValidator(QDoubleValidator(70.0, 6000.0, 3))
        layout_cont.addWidget(self.txt_jam_freq)

        # KARIŞTIRMA BANT GENİŞLİĞİ: baraj/süpürme bandını belirler (ana panelle aynı seçenekler).
        # Dinleme için 2.4; geniş baraj/Wi-Fi için 20–56 MHz. (Yüksek değerde RX overflow olabilir.)
        layout_cont.addWidget(QLabel("Karıştırma Bant Genişliği (MHz):"))
        self.cmb_jam_bw = QComboBox()
        self.cmb_jam_bw.addItems(["1.0", "2.0", "2.4", "4.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0", "56.0", "61.44"])
        self.cmb_jam_bw.setCurrentText("20.0")
        layout_cont.addWidget(self.cmb_jam_bw)

        # TEK GÜÇ ALANI (TX RF kazancı): gerçek radyasyon gücünü belirler. B200mini üst sınırı ~89.75 dB.
        # (JSR kaldırıldı: dijital sürüş tam-skalaya sabittir; güç yalnızca bu alanla ayarlanır.)
        layout_cont.addWidget(QLabel("TX Gücü (RF Kazanç, dB, 0–89.75):"))
        self.txt_tx_gain = QLineEdit("80")
        self.txt_tx_gain.setValidator(QDoubleValidator(0.0, 89.75, 2))
        layout_cont.addWidget(self.txt_tx_gain)
        layout_cont.addStretch(1)

        # Sekme 2: Aralıklı Karıştırma (Look-Through)
        tab_intermittent = QWidget()
        layout_inter = QVBoxLayout(tab_intermittent)
        layout_inter.setSpacing(14)
        
        from PyQt6.QtWidgets import QCheckBox
        sec_val = QDoubleValidator(0.5, 600.0, 1)      # süreler SANİYE cinsinden (ms değil)

        layout_inter.addWidget(QLabel("Aralıklı (Look-Through) Karıştırma Parametreleri:"))

        # KARIŞTIRMA FREKANSI — TX bu sekmede BAĞIMSIZ girilir (ana panel "Sinyal Al" frekansından
        # etkilenmez). Oto-tarama açıksa hedef susunca [tarama başlangıç–bitiş] bandı taranır; bu alan
        # başlangıç karıştırma frekansıdır. Sürekli/analog sekmeleriyle tutarlı.
        layout_inter.addWidget(QLabel("Karıştırma Frekansı (MHz, 70–6000):"))
        self.txt_lt_freq = QLineEdit("433.500")
        self.txt_lt_freq.setValidator(QDoubleValidator(70.0, 6000.0, 3))
        layout_inter.addWidget(self.txt_lt_freq)

        # BANT GENİŞLİĞİ (MHz) — TX ekranından bağımsız (ana panelden değil). Jam + dinleme + tarama
        # bu örnekleme hızında yapılır. Dar hedef için 2.4; geniş tarama için daha fazla.
        layout_inter.addWidget(QLabel("Karıştırma/Tarama Bant Genişliği (MHz):"))
        self.cmb_lt_bw = QComboBox()
        self.cmb_lt_bw.addItems(["1.0", "2.0", "2.4", "4.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0", "56.0", "61.44"])
        self.cmb_lt_bw.setCurrentText("2.4")
        layout_inter.addWidget(self.cmb_lt_bw)

        # KARIŞTIRMA + DİNLEME süreleri — SANİYE cinsinden (5 sn jam / 2 sn dinle gibi)
        row_t = QHBoxLayout()
        col_jam = QVBoxLayout()
        col_jam.addWidget(QLabel("Karıştırma Süresi (sn):"))
        self.txt_jam_sec = QLineEdit("5")
        self.txt_jam_sec.setValidator(sec_val)
        col_jam.addWidget(self.txt_jam_sec)
        row_t.addLayout(col_jam)
        col_listen = QVBoxLayout()
        col_listen.addWidget(QLabel("Dinleme Süresi (sn):"))
        self.txt_listen_sec = QLineEdit("2")
        self.txt_listen_sec.setValidator(sec_val)
        col_listen.addWidget(self.txt_listen_sec)
        row_t.addLayout(col_listen)
        layout_inter.addLayout(row_t)

        # OTOMATİK HEDEF TARAMA: hedef susunca band tarayıp yeni aktif frekans(lar) bul, sırayla ez
        self.chk_lt_autoscan = QCheckBox("Hedef susunca bandı tara ve yeni hedefleri sırayla karıştır")
        self.chk_lt_autoscan.setChecked(True)
        self.chk_lt_autoscan.setStyleSheet("font-size:14px; color:#e0e0e0;")
        layout_inter.addWidget(self.chk_lt_autoscan)

        row_b = QHBoxLayout()
        col_s = QVBoxLayout()
        col_s.addWidget(QLabel("Jam Tarama Başlangıç (MHz):"))
        self.txt_lt_scan_start = QLineEdit("430.0")
        self.txt_lt_scan_start.setValidator(QDoubleValidator(70.0, 6000.0, 3))
        col_s.addWidget(self.txt_lt_scan_start)
        row_b.addLayout(col_s)
        col_e = QVBoxLayout()
        col_e.addWidget(QLabel("Jam Tarama Bitiş (MHz):"))
        self.txt_lt_scan_stop = QLineEdit("440.0")
        self.txt_lt_scan_stop.setValidator(QDoubleValidator(70.0, 6000.0, 3))
        col_e.addWidget(self.txt_lt_scan_stop)
        row_b.addLayout(col_e)
        layout_inter.addLayout(row_b)

        _hint = QLabel("Akış: KARIŞTIR (sn) → DİNLE (sn). Dinlemede hedef hâlâ yayındaysa ez; SUSTUYSA "
                       "band taranır, bulunan aktif frekanslar SIRAYLA ezilir. Öneri: 5 sn jam / 2 sn dinle.\n"
                       "Donanım koruması için süreler en az 0.5 sn uygulanır. Otomatik tarama kapalıysa "
                       "yalnızca o anki frekans ezilir (klasik arabakış).")
        _hint.setStyleSheet("color:#9aa4b0; font-size:11px;")
        _hint.setWordWrap(True)
        layout_inter.addWidget(_hint)
        layout_inter.addStretch(1)

        # Sekme 3: Analog Telsiz Aldatma
        tab_analog_spoof = QWidget()
        layout_analog = QVBoxLayout(tab_analog_spoof)
        layout_analog.setSpacing(14)
        
        # HEDEF TELSİZ FREKANSI: aldatma bu frekanstan yayınlanır. Bu alan olmadan yayın, panelin
        # o anki frekansında kalıyordu -> hedef telsiz DUYMUYORDU (operatör frekansı yanlışlıkla
        # 'NBFM Sapma' alanına giriyordu). PMR446 kanal 1 varsayılan (446.00625 MHz).
        layout_analog.addWidget(QLabel("Hedef Telsiz Frekansı (MHz, 70–6000):"))
        self.txt_analog_freq = QLineEdit("446.00625")
        self.txt_analog_freq.setValidator(QDoubleValidator(70.0, 6000.0, 5))
        layout_analog.addWidget(self.txt_analog_freq)

        # BANT GENİŞLİĞİ (MHz) — TX ekranından bağımsız. NBFM ses telsizi DAR-BANTTIR (~12.5 kHz kanal);
        # düşük örnekleme (1-2 MHz) yeterli ve temizdir. Ana panelden bağımsız girilir.
        layout_analog.addWidget(QLabel("Bant Genişliği (MHz):"))
        self.cmb_analog_bw = QComboBox()
        self.cmb_analog_bw.addItems(["1.0", "2.0", "2.4", "4.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0", "56.0", "61.44"])
        self.cmb_analog_bw.setCurrentText("2.0")
        layout_analog.addWidget(self.cmb_analog_bw)

        layout_analog.addWidget(QLabel("Aldatma Sinyali Dalga Şekli:"))
        self.cmb_wave = QComboBox()
        self.cmb_wave.addItems(["Ses Dosyası (WAV Mesaj)", "Sinüs Dalga (Tone)", "Ses/Audio Sahte Ses",
                                "Gürültü Modüleli AM", "Gürültü Modüleli FM"])
        layout_analog.addWidget(self.cmb_wave)

        # GERÇEK SES MESAJI (WAV) — hedef analog telsize anlaşılır sahte yayın (5.2.3)
        layout_analog.addWidget(QLabel("Ses Mesajı Dosyası (WAV) — hedef 'yanlış duysun':"))
        file_row = QHBoxLayout()
        self.txt_decept_file = QLineEdit("")
        self.txt_decept_file.setPlaceholderText("örn. /home/.../sahte_anons.wav")
        file_row.addWidget(self.txt_decept_file)
        self.btn_browse_wav = QPushButton("Gözat…")
        self.btn_browse_wav.clicked.connect(self._browse_wav)
        file_row.addWidget(self.btn_browse_wav)
        layout_analog.addLayout(file_row)

        layout_analog.addWidget(QLabel("Aldatma Modülasyonu (hedefe uydur):"))
        self.cmb_decept_mod = QComboBox()
        self.cmb_decept_mod.addItems(["NBFM (dar-bant FM ses telsizi)", "AM (genlik modülasyonu)"])
        layout_analog.addWidget(self.cmb_decept_mod)

        layout_analog.addWidget(QLabel("NBFM Tepe Sapması (Hz, hedef kanalına göre):"))
        self.txt_decept_dev = QLineEdit("2000")
        self.txt_decept_dev.setValidator(QDoubleValidator(500.0, 75000.0, 1))
        layout_analog.addWidget(self.txt_decept_dev)

        # CTCSS alt-ses tonu (ton-squelch): hedef bu ton olmadan hoparlörü AÇMAZ (5.2.3 kritik)
        layout_analog.addWidget(QLabel("CTCSS Alt-Ses Tonu (hedefin ton-squelch'ini açar):"))
        self.cmb_ctcss = QComboBox()
        self.cmb_ctcss.addItem("Yok (carrier squelch)")
        for tone in [67.0, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4, 100.0,
                     103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3, 131.8, 136.5, 141.3, 146.2,
                     151.4, 156.7, 162.2, 167.9, 173.8, 179.9, 186.2, 192.8, 203.5, 210.7, 218.1,
                     225.7, 233.6, 241.8, 250.3]:
            self.cmb_ctcss.addItem(f"{tone:.1f} Hz")
        layout_analog.addWidget(self.cmb_ctcss)

        # NOT: "Zamanlama Offseti (ms)" alanı kaldırıldı — süre DEĞİLdi (12 ms), yalnızca sentetik
        # dalgalara fark edilmez bir faz kaydırması ekliyordu ve WAV mesaj modunda hiç kullanılmıyordu.
        # Yayın, DURDURULANA kadar süreklidir (WAV mesajı loop'ta tekrarlanır). Operatörü yanıltmasın.
        layout_analog.addStretch(1)

        # Sekme 4: GNSS Aldatma
        tab_gnss_spoof = QWidget()
        layout_gnss = QVBoxLayout(tab_gnss_spoof)
        layout_gnss.setSpacing(14)
        
        layout_gnss.addWidget(QLabel("GNSS Servisi (frekans otomatik ayarlanır):"))
        self.cmb_gnss_code = QComboBox()
        # ŞİMDİLİK yalnızca ÜST L-BANDI servisleri (tek GNSS anteniyle kapsanır):
        #   GPS L1 = 1575.42, GALILEO E1 = 1575.42 (aynı!), BEIDOU B1 = 1561.098 (~14 MHz yakın).
        # Diğer servisler farklı frekanslarda (ayrı anten ister) -> ileride kullanmak için YORUMDA.
        self.cmb_gnss_code.addItems([
            "GPS L1", "GALILEO E1", "BEIDOU B1",
            # --- İLERİDE (ayrı anten/frekans gerektirir) — yorumdan çıkarınca geri gelir ---
            # "GPS L2", "GPS L5",
            # "GLONASS L1", "GLONASS L2", "GLONASS L3",
            # "GALILEO E5a", "GALILEO E5b", "GALILEO E6",
            # "BEIDOU B2", "BEIDOU B3",
        ])
        layout_gnss.addWidget(self.cmb_gnss_code)

        # BANT GENİŞLİĞİ (MHz) — TX ekranından bağımsız. Kendi üretecimizde servise göre önerilen:
        # GPS L1 ~16, GALILEO E1 / BEIDOU B1 ~6 MHz. (GPS-SDR-SIM modunda otomatik 2.6 MHz'e zorlanır.)
        layout_gnss.addWidget(QLabel("GNSS Bant Genişliği (MHz):"))
        self.cmb_gnss_bw = QComboBox()
        # Ana panel değerleri + GNSS'e özel 2.6 (GPS-SDR-SIM için gerekli)
        self.cmb_gnss_bw.addItems(["2.6", "1.0", "2.0", "2.4", "4.0", "5.0", "10.0", "20.0", "30.0", "40.0", "50.0", "56.0", "61.44"])
        self.cmb_gnss_bw.setCurrentText("2.6")
        layout_gnss.addWidget(self.cmb_gnss_bw)

        layout_gnss.addWidget(QLabel("Spoofing Modu (Otonom veya Sabit):"))
        self.cmb_spoof_mode = QComboBox()
        self.cmb_spoof_mode.addItems([
            "MANUAL (Sabit Koordinata Kitle)",
            "AUTONOMOUS_RANDOM (Okyanusa Işınla)",
            "DYNAMIC_DRIFT (Kuzeye Sürekli Kaydır)",
            # Gerçek ephemeris'li GPS L1 (gerçek alıcıyı kandırır) — önce tools/gen_gps_spoof.py ile
            # data/gpssim_l1.bin üret; SDR 2.6 MHz olmalı. Yalnızca GPS L1 servisinde geçerli.
            "GPS-SDR-SIM L1 (gerçek ephemeris)",
        ])
        layout_gnss.addWidget(self.cmb_spoof_mode)

        layout_gnss.addWidget(QLabel("Manuel/Referans Koordinat (Enlem, Boylam):"))
        self.txt_coords = QLineEdit("39.9207, 32.8541")
        coords_regex = QRegularExpression(r"^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$")
        self.txt_coords.setValidator(QRegularExpressionValidator(coords_regex))
        layout_gnss.addWidget(self.txt_coords)
        
        # Seçilen moda göre koordinat girişini etkinleştir/devre dışı bırak
        self.cmb_spoof_mode.currentTextChanged.connect(
            lambda text: self.txt_coords.setEnabled("MANUAL" in text or "DYNAMIC" in text)
        )
        
        layout_gnss.addStretch(1)

        # Sekme 5: Wi-Fi Engelleme (GERÇEK baraj yayını — bulgu #9)
        tab_wifi_jam = QWidget()
        layout_wifi = QVBoxLayout(tab_wifi_jam)
        layout_wifi.setSpacing(14)

        layout_wifi.addWidget(QLabel("Hedef Wi-Fi Bandı (merkez frekansına otomatik geçilir):"))
        self.cmb_wifi_band = QComboBox()
        self.cmb_wifi_band.addItems(["2.4 GHz (802.11 b/g/n)", "5.8 GHz (802.11 a/n/ac)"])
        layout_wifi.addWidget(self.cmb_wifi_band)

        # NOT: "Deauthentication" kaldırıldı — bu teknik SDR baseband ile değil, monitör-modlu
        # bir Wi-Fi kartıyla 802.11 yönetim çerçevesi enjeksiyonu gerektirir. Sahte seçenek
        # bırakmak yerine dürüstçe yalnızca fiziksel olarak yapılabileni (geniş-bant baraj) sunuyoruz.
        layout_wifi.addWidget(QLabel("Karıştırma Yöntemi:"))
        self.cmb_wifi_type = QComboBox()
        self.cmb_wifi_type.addItems(["Sürekli Gürültü (Barrage)"])
        layout_wifi.addWidget(self.cmb_wifi_type)
        layout_wifi.addStretch(1)

        self.tabs.addTab(tab_continuous, "Sürekli Karıştırma")
        self.tabs.addTab(tab_intermittent, "Aralıklı Karıştırma")
        self.tabs.addTab(tab_analog_spoof, "Analog Aldatma")
        self.tabs.addTab(tab_gnss_spoof, "GNSS Aldatma")
        self.tabs.addTab(tab_wifi_jam, "Wi-Fi Engelleme")
        layout.addWidget(self.tabs)

        btn_layout = QHBoxLayout()
        self.btn_start_tx = QPushButton("YAYINI / DÜZENEĞİ BAŞLAT")
        self.btn_start_tx.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; padding: 12px; font-size: 16px; border-radius: 4px;")
        self.btn_start_tx.clicked.connect(self.on_start_tx_clicked)

        self.btn_close = QPushButton("Kapat")
        self.btn_close.setStyleSheet("background-color: #555555; color: white; padding: 12px; font-size: 16px; border-radius: 4px;")
        self.btn_close.clicked.connect(self.close)

        btn_layout.addWidget(self.btn_start_tx)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def _browse_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "Aldatma Ses Mesajı (WAV) Seç", "",
                                              "WAV ses dosyaları (*.wav)")
        if path:
            self.txt_decept_file.setText(path)

    def on_start_tx_clicked(self):
        current_tab_idx = self.tabs.currentIndex()
        
        tx_data = {
            "tab_index": current_tab_idx,
            # JSR kaldırıldı: dijital sürüş tam-skalaya sabit (20 dB = tavan); güç yalnızca TX kazancı.
            "jsr_db": 20.0,
            "tx_gain_db": float(self.txt_tx_gain.text() or 80.0),
            # ARABAKIŞ süreleri — SANİYE cinsinden (ms değil). jam_sec: karıştırma, listen_sec: dinleme.
            "jam_sec": float(self.txt_jam_sec.text() or 5.0),
            "listen_sec": float(self.txt_listen_sec.text() or 2.0),
            # Otomatik hedef tarama (hedef susunca bandı tara, yeni aktif frekansları sırayla ez)
            "lt_auto_scan": bool(self.chk_lt_autoscan.isChecked()),
            "lt_scan_start_mhz": float(self.txt_lt_scan_start.text() or 430.0),
            "lt_scan_stop_mhz": float(self.txt_lt_scan_stop.text() or 440.0),
            "wave_type": self.cmb_wave.currentText(),
            "offset_ms": 12.0,   # sentetik dalgalar için sabit; WAV modunda kullanılmaz (UI alanı kaldırıldı)
            # Gerçek ses aldatma (5.2.3): WAV mesaj + hedefe uygun modülasyon/sapma + CTCSS tonu
            "decept_audio_file": self.txt_decept_file.text().strip(),
            "decept_mod": "AM" if self.cmb_decept_mod.currentText().startswith("AM") else "NBFM",
            "decept_deviation_hz": float(self.txt_decept_dev.text() or 2000.0),
            "decept_ctcss_hz": (0.0 if self.cmb_ctcss.currentIndex() == 0
                                else float(self.cmb_ctcss.currentText().split()[0])),
            "gnss_code": self.cmb_gnss_code.currentText(),
            "coords": self.txt_coords.text(),
            # Hedef Sinyal Tipi kaldırıldı: baraj bandı ana paneldeki Bant Genişliği'nden gelir (varsayılan profil).
            "target_signal": "",
            "wifi_band": self.cmb_wifi_band.currentText(),
            "wifi_method": self.cmb_wifi_type.currentText(),
        }

        if current_tab_idx == 0:
            jam = self.cmb_jam_type.currentText()
            if "Baraj" in jam:
                tx_data["mode"] = "BARRAGE"
            elif "Çoklu" in jam:
                tx_data["mode"] = "MULTI_TONE"      # önceden SINGLE_TONE'a düşüyordu (bug)
            elif "Süpürmeli" in jam:
                tx_data["mode"] = "SWEEP"           # frekans-çevik hedeflere karşı chirp (retune yok)
            else:
                tx_data["mode"] = "SPOT"            # tekli = hedef kanalına dar-bant spot GÜRÜLTÜ
            # Karıştırma frekansı + bant genişliği (operatör TX ekranından girer)
            tx_data["tx_freq_mhz"] = float(self.txt_jam_freq.text() or 2437.0)
            tx_data["tx_bw_str"] = self.cmb_jam_bw.currentText()          # ana panel combo'suyla eşleşir
            tx_data["tx_bw_mhz"] = float(self.cmb_jam_bw.currentText())
            msg = (f"SÜREKLİ KARIŞTIRMA BAŞLATILDI -> Tip: {jam} | "
                   f"Frekans: {tx_data['tx_freq_mhz']:.3f} MHz | BW: {tx_data['tx_bw_mhz']:.1f} MHz | "
                   f"TX Güç: {tx_data['tx_gain_db']:.0f} dB")
        elif current_tab_idx == 1:
            tx_data["mode"] = "LOOK_THROUGH"
            tx_data["tx_bw_mhz"] = float(self.cmb_lt_bw.currentText() or 2.4)   # TX ekranından bağımsız BW
            # BAĞIMSIZ karıştırma frekansı (ana RX frekansından bağımsız). Worker bu frekansa geçer,
            # arabakış ilk hedefi burası olur; autoscan açıksa hedef susunca tarama bandına geçilir.
            tx_data["tx_freq_mhz"] = float(self.txt_lt_freq.text() or 433.5)
            if tx_data["lt_auto_scan"]:
                msg = (f"ARABAKIŞLI KARIŞTIRMA (OTO-TARAMA) -> Jam {tx_data['jam_sec']:.0f} sn / "
                       f"Dinle {tx_data['listen_sec']:.0f} sn | Tarama: "
                       f"{tx_data['lt_scan_start_mhz']:.1f}-{tx_data['lt_scan_stop_mhz']:.1f} MHz")
            else:
                msg = (f"ARABAKIŞLI KARIŞTIRMA -> Jam {tx_data['jam_sec']:.0f} sn / "
                       f"Dinle {tx_data['listen_sec']:.0f} sn (tek frekans)")
        elif current_tab_idx == 2:
            tx_data["mode"] = "ANALOG_SPOOF"
            # Hedef telsiz frekansına geç (worker set_frequency uygular). Bu olmadan hedef duymaz.
            tx_data["tx_freq_mhz"] = float(self.txt_analog_freq.text() or 446.00625)
            tx_data["tx_bw_mhz"] = float(self.cmb_analog_bw.currentText() or 2.0)   # TX ekranından bağımsız BW
            ctcss = "Yok" if self.cmb_ctcss.currentIndex() == 0 else f"{tx_data['decept_ctcss_hz']:.1f} Hz"
            msg = (f"ANALOG ALDATMA BAŞLATILDI -> Frekans: {tx_data['tx_freq_mhz']:.5f} MHz | "
                   f"Dalga: {tx_data['wave_type']} | Mod: {tx_data['decept_mod']} | "
                   f"Sapma: {tx_data['decept_deviation_hz']:.0f} Hz | CTCSS: {ctcss}")
        elif current_tab_idx == 3:
            tx_data["mode"] = "GNSS_SPOOF"
            spoof_mode_text = self.cmb_spoof_mode.currentText()
            # BANT GENİŞLİĞİ TX ekranından (bağımsız). GPS-SDR-SIM'de 2.6 MHz'e ZORLANIR (dosya 2.6 Msps).
            tx_data["tx_bw_mhz"] = float(self.cmb_gnss_bw.currentText() or 6.0)
            if "GPS-SDR-SIM" in spoof_mode_text:
                tx_data["spoof_mode"] = "GPSSDRSIM_L1"   # gerçek ephemeris baseband dosyası (GPS L1)
                tx_data["gnss_code"] = "GPS L1"          # yalnızca GPS L1'de geçerli
                tx_data["tx_bw_mhz"] = 2.6               # GPS-SDR-SIM 2.6 Msps -> BW zorla 2.6
            elif "RANDOM" in spoof_mode_text:
                tx_data["spoof_mode"] = "AUTONOMOUS_RANDOM"
            elif "DYNAMIC" in spoof_mode_text:
                tx_data["spoof_mode"] = "DYNAMIC_DRIFT"
            else:
                tx_data["spoof_mode"] = "MANUAL"
            
            msg = f"GNSS ALDATMA ({tx_data['spoof_mode']}) BAŞLATILDI -> Kod: {tx_data['gnss_code']} | Coords: [{tx_data['coords']}]"
        else:
            tx_data["mode"] = "WIFI_JAMMING"          # gerçek baraj yayını (bulgu #9)
            wifi_band = self.cmb_wifi_band.currentText()
            wifi_type = self.cmb_wifi_type.currentText()
            msg = f"WI-FI ENGELLEME (BARAJ) BAŞLATILDI -> Bant: {wifi_band} | Yöntem: {wifi_type}"

        self.tx_payload = tx_data
        self.tx_signal_started.emit(msg)
        self.accept()
