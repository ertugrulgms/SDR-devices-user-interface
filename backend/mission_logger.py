import sqlite3
import csv
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Proje kök dizinine göre sabit bir veri klasörü (çalıştırma dizinine bağlı değil)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


class MissionLogger:
    """
    SDR Elektronik Harp Operasyonları için Veritabanı (SQLite) ve CSV Kayıt Motoru.
    Tespit edilen hedeflerin koordinatlarını, frekanslarını, ET taarruz durumlarını
    ve Yön Bulma (DF) doğruluk ölçümlerini loglar.
    """
    def __init__(self, db_name="mission_logs.db"):
        os.makedirs(_DATA_DIR, exist_ok=True)
        self.db_path = os.path.join(_DATA_DIR, db_name)
        self.conn = None
        self.cursor = None
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_tables()
        except sqlite3.Error as exc:
            logger.error("MissionLogger: veritabanı açılamadı (%s): %s", self.db_path, exc)

    def _create_tables(self):
        """Log tablolarını yoksa oluşturur."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS target_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frequency_mhz REAL,
                target_x_km REAL,
                target_y_km REAL,
                tx_mode_active TEXT,
                notes TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS df_accuracy_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                reference_deg REAL,
                measured_deg REAL,
                rms_error_deg REAL,
                sample_count INTEGER
            )
        """)
        # SİNYAL İSTİHBARAT LOGU: tespit edilen sinyallerin (modülasyon/çoklama/EKKT/protokol/BW/SNR)
        # zaman-frekans kaydı. EH görevinde "saat X'te şu frekansta FHSS/OFDM tespit edildi" kanıtı.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frequency_mhz REAL,
                modulation TEXT,
                multiplex TEXT,
                ekkt TEXT,
                protocol TEXT,
                occupied_bw_hz REAL,
                snr_db REAL,
                confidence REAL
            )
        """)
        # YÖN BULMA / KONUM LOGU: her üçgenleme fix'i — kaynağın 3B konumu, kesişim kalitesi ve
        # füzyona giren düğüm kerterizleri (JSON). Arayüzde gösterilmese de raporda tam kayıt kalır;
        # ayrıca operatör bu açı bilgilerini elle girip konumu yeniden üretebilir.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS df_fix_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frequency_mhz REAL,
                pos_x_m REAL,
                pos_y_m REAL,
                pos_z_m REAL,
                residual_m REAL,
                node_count INTEGER,
                target_bearing_deg REAL,
                target_range_m REAL,
                target_elevation_deg REAL,
                nodes_json TEXT
            )
        """)
        # SİNYAL TESPİT LOGU (şartname 5.1.1): bant taramada gürültü üstü tespit edilen her yeni
        # sinyalin frekansı + gücü + SNR'ı. "Frekanslar söylenmeden bulundu" kanıtı.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frequency_mhz REAL,
                power_dbfs REAL,
                snr_db REAL
            )
        """)
        self.conn.commit()

    def log_detection(self, freq_mhz: float, power_dbfs: float, snr_db: float):
        """Bant taramada tespit edilen bir sinyali kaydeder (5.1.1)."""
        if self.cursor is None:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            self.cursor.execute(
                "INSERT INTO detection_logs (timestamp, frequency_mhz, power_dbfs, snr_db) VALUES (?, ?, ?, ?)",
                (now_str, freq_mhz, power_dbfs, snr_db))
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.error("MissionLogger.log_detection hata: %s", exc)

    def log_target(self, freq_mhz: float, x_km: float, y_km: float, tx_mode: str, notes: str = ""):
        """Anlık hedef tespitini ve durumunu veritabanına kaydeder."""
        if self.cursor is None:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            self.cursor.execute("""
                INSERT INTO target_logs (timestamp, frequency_mhz, target_x_km, target_y_km, tx_mode_active, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now_str, freq_mhz, x_km, y_km, tx_mode, notes))
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.error("MissionLogger.log_target hata: %s", exc)

    def log_df_accuracy(self, reference_deg: float, measured_deg: float, rms_error_deg: float, sample_count: int):
        """Yön Bulma kalibrasyon ölçümünü ve RMS doğruluğunu kaydeder (KTR için kanıt)."""
        if self.cursor is None:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            self.cursor.execute("""
                INSERT INTO df_accuracy_logs (timestamp, reference_deg, measured_deg, rms_error_deg, sample_count)
                VALUES (?, ?, ?, ?, ?)
            """, (now_str, reference_deg, measured_deg, rms_error_deg, sample_count))
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.error("MissionLogger.log_df_accuracy hata: %s", exc)

    def log_signal(self, freq_mhz: float, modulation: str, multiplex: str, ekkt: str,
                   protocol: str, occupied_bw_hz: float, snr_db: float, confidence: float):
        """Tespit edilen bir sinyalin sınıflandırma sonucunu kaydeder (sinyal istihbaratı)."""
        if self.cursor is None:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            self.cursor.execute("""
                INSERT INTO signal_logs (timestamp, frequency_mhz, modulation, multiplex, ekkt,
                                         protocol, occupied_bw_hz, snr_db, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_str, freq_mhz, modulation, multiplex, ekkt, protocol,
                  occupied_bw_hz, snr_db, confidence))
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.error("MissionLogger.log_signal hata: %s", exc)

    def log_df_fix(self, freq_mhz, pos_xyz, residual_m, node_count,
                   target_bearing_deg, target_range_m, target_elevation_deg, nodes: dict):
        """Bir üçgenleme fix'ini (kaynak konumu + düğüm kerterizleri) kaydeder. nodes: düğüm
        kerteriz sözlüğü (id -> {azimuth_deg, elevation_deg, amp_dbm, ...}); JSON olarak saklanır."""
        if self.cursor is None:
            return
        import json as _json
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        try:
            self.cursor.execute("""
                INSERT INTO df_fix_logs (timestamp, frequency_mhz, pos_x_m, pos_y_m, pos_z_m,
                    residual_m, node_count, target_bearing_deg, target_range_m,
                    target_elevation_deg, nodes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_str, freq_mhz, float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]),
                  residual_m, node_count, target_bearing_deg, target_range_m,
                  target_elevation_deg, _json.dumps(nodes, ensure_ascii=False)))
            self.conn.commit()
        except (sqlite3.Error, TypeError, ValueError) as exc:
            logger.error("MissionLogger.log_df_fix hata: %s", exc)

    def export_to_csv(self, csv_filename="mission_report.csv"):
        """Tüm logları jüri/raporlama için Excel uyumlu CSV formatına dönüştürür."""
        csv_path = os.path.join(_DATA_DIR, csv_filename)
        if self.cursor is None:
            return csv_path
        try:
            self.cursor.execute("SELECT * FROM target_logs")
            rows = self.cursor.fetchall()

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Tarih/Saat", "Frekans (MHz)", "Hedef X (km)", "Hedef Y (km)", "TX Karıştırma Modu", "Notlar"])
                writer.writerows(rows)
        except (sqlite3.Error, OSError) as exc:
            logger.error("MissionLogger.export_to_csv hata: %s", exc)

        return csv_path

    def close(self):
        if self.conn is not None:
            self.conn.close()