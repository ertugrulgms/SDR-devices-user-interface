import time
import numpy as np
import socket
import threading
import json
from PyQt6.QtCore import QThread, pyqtSignal

from backend.dsp_processor import DSPProcessor, DFAccuracyTracker, NoiseFloorTracker
from backend.jamming_generator import JammingGenerator
from backend.gnss_spoofer import GNSSSpoofer
from backend.mission_logger import MissionLogger  # <-- LOGLAYICI EKLENDİ
from backend.hardware_controller import HardwareController
from backend.audio_demodulator import AudioDemodulator, StreamingDemodulator
from backend.digital_voice import classify_fm_or_fsk
from backend.signal_monitor import SignalMonitor
from backend.signal_classifier import SignalClassifier, HoppingHistoryTracker
from backend.param_consolidator import ParameterConsolidator
from backend.tx_engine import TxWaveformBuilder, WIFI_BAND_CENTERS, WAV_DECEPTION_WAVE
from backend.audio_deception import load_wav_mono, detect_ctcss, detect_dcs, extract_subaudio
from backend.direction_finding import (NodeBearingStore, AmplitudeDFEstimator, azel_to_unit,
                                       triangulate_lob, bearing_from_positions,
                                       load_node_registry, self_node_id,
                                       polar_to_enu, enu_to_polar, save_node_registry)

# --- SOAPYSDR DONANIM KÜTÜPHANESİ KONTROLÜ ---
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    import sdr_core
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False

# --- SABİTLER (bulgu #13: sihirli sayılar isimlendirildi) ---
FFT_POINTS = 2048
UPDATE_INTERVAL_SEC = 0.033       # ~30 FPS (33ms)
TX_GAIN_DEFAULT_DB = 80.0         # varsayılan TX RF kazancı (jamming için yüksek)
TX_GAIN_MAX_DB = 89.75            # B200mini TX kazanç üst sınırı
CLASSIFY_PERIOD_SEC = 0.5         # olay tetiklendiğinde ağır AMC en fazla bu sıklıkta çalışır
CLASSIFY_TRIGGER_DB = 10.0        # OLAY EŞİĞİ: tepe-taban farkı bunu aşınca sinyal "var" sayılır
                                  # -> ağır AMC yalnızca o zaman çalışır (event-based, CPU korunur)
AUDIO_SQUELCH_SNR_DB = 8.0        # SES SQUELCH: kanal SNR'ı bunun altındaysa ses susar (sürekli
                                  # AGC-yükseltilmiş cızırtı yerine sessizlik). set_squelch_snr_db ile ayarlanır
AUDIO_CENTER_MAX_HZ = 200_000.0   # ses oto-merkezleme: yalnızca ±bu kadar kaymayı düzelt (tüm bandı
                                  # tarayıp uzak spur'a kilitlenme). Dinleme her BW'de merkezde çalışır.
CLASSIFY_SNAPSHOT_N = 32768       # ring buffer'dan alınacak kayıpsız sınıflandırma kaydı (örnek)
LOOK_THROUGH_SIGNAL_TH_DB = 10.0  # arabakış: tepe-taban farkı bu eşiğin üstündeyse "kanalda sinyal var"
# Aç/kapa (T/R) döngü periyodu ALT SINIRI — hem DONANIM KORUMASI hem KARIŞTIRMA GÜCÜ.
# T/R geçişinde bir "ölü zaman" var (~30-80 ms: TX akışı deaktive + tampon boşalt/doldur + oturma).
# Periyot bu ölü zamana yakınsa cihaz ömrünün BÜYÜK KISMINI aç/kapa'yla harcar -> havaya basılan
# RMS jam gücü DİBE ÇAKILIR ve LED saniyede defalarca yanıp söner (T/R'yi parçalar). EW kuralı:
# "SDR'ı olabildiğince UZUN TX'te tut." Bu yüzden taban 250->500 ms: ölü zaman toplam periyodun
# <%16'sı kalır (>%84 gerçek jam), LED en fazla ~2 Hz. Varsayılan periyot 1000 ms + duty %90 ile
# jam ~900 ms KESİNTİSİZ tam güç, peek ~100 ms. Operatör daha da büyük periyot seçebilir; küçük
# seçse de buraya klipslenir (kendini sabote edip gücü öldüremez).
LOOK_THROUGH_MIN_PERIOD_SEC = 0.5
LOOK_THROUGH_ENDED_WINDOWS = 2    # arabakış: bu kadar ardışık BOŞ dinleme penceresi -> "yayın sonlandı"
LOOK_THROUGH_REFOCUS_HZ = 60e3   # arabakış: hedef offseti bu kadar kayarsa gücü yeniden odakla (rebuild)
LOOK_THROUGH_SCAN_SETTLE_SEC = 0.12  # arabakış oto-tarama: her retune sonrası RX oturma süresi
HEALTH_POLL_SEC = 2.0             # C++ akış-sağlığı sayaçlarını raporlama periyodu
# --- YÖN BULMA (şartname 5.1.4) ---
DF_SIGNAL_PRESENT_DB = 6.0        # genlik-DF örneği YALNIZCA sinyal bu SNR'yi aşınca beslenir; aksi
                                  # halde (kaynak sustuğunda) gürültü tepesi azimut-genlik haritasını
                                  # kirletir ve sahte kerteriz üretirdi (sıralı yayın senaryosu).
DF_PEAK_WINDOW_BINS = 2          # genlik ölçümünde tepe etrafı ±bin entegrasyonu (tek-bin gürültüsü)
DF_FUSION_FREQ_TOL_MHZ = 0.1     # füzyon: düğüm kerterizi ancak hedef frekansına ±bu kadar yakınsa
                                 # katılır (farklı frekanstaki aux başka hedefi ölçüyordur -> dışla)

# --- RF BANT TARAMA / SİNYAL TESPİTİ (şartname 5.1.1) ---
SCAN_DETECT_DB = 10.0             # sinyal, TARİHSEL-MİN gürültü tabanını bu kadar aşarsa tespit
SCAN_SETTLE_SEC = 0.04           # retune sonrası oturma; kısa tutuldu (FHSS/burst POI için, uzman #3)
SCAN_MERGE_MHZ = 0.0125          # bitişik tespitleri birleştirme aralığı. PMR el telsizi KANAL ARALIĞI
                                 # 12.5 kHz'dir; 0.05 (50 kHz) bitişik iki kanalı TEK sinyalde eziyordu.
                                 # 12.5 kHz -> yan yana kanallar ayrı hedefler olarak görünür.
SCAN_MAX_DETECTIONS = 400        # tespit listesi üst sınırı
SCAN_CONFIRM_HITS = 1            # PTT (bas-konuş) telsizi/drone kumandası ANLIK yayın yapar; kaynak
                                 # susunca 2. onay gelmez -> eskiden (2) bu sinyalleri KAÇIRIYORDU.
                                 # 1 = ilk görüşte listeye al. Hayalet-selini CFAR+eşik+DC-guard eler
                                 # (temporal onaya gerek kalmadı). Sahada çok yalancı-pozitif olursa 2 yap.
SCAN_CANDIDATE_TTL_SEC = 8.0     # onaylanmamış aday bu süre yeniden görülmezse düşürülür (CONFIRM=1'de kullanılmaz)
SCAN_DC_GUARD_BINS = 2           # merkezdeki (n/2) DAR DC/LO dikenini ele. 4 idi: merkeze denk gelen
                                 # gerçek dar telsizi de siliyordu (kör nokta). 2 -> yalnız ~1-2 bin diken
                                 # silinir; ~8 kHz'lik gerçek telsiz merkezde bile korunur.
SCAN_CLOSE_GAP_HZ = 4_000        # bu kadar kısa eşik-altı boşluklar kapatılır (dalgalı sinyal parçalanmasın).
                                 # 15 kHz idi: 12.5 kHz aralıklı bitişik PMR kanallarının arasını da
                                 # köprüleyip tek sinyalde birleştiriyordu. 4 kHz -> kanallar ayrı kalır,
                                 # ama tek sinyalin kendi içindeki küçük çentikler yine kapatılır.
SCAN_SHOULDER_DB = 22.0          # bir tespit, YAKIN + çok daha güçlü bir sinyalin "omuz/etek" gölgesinde
                                 # (bu kadar dB zayıf) ise AYRI sinyal sayılmaz (baskın taşıyıcı eteği)
SCAN_SHOULDER_SPAN_MULT = 1.5    # gölge yarıçapı = güçlü sinyalin bant genişliği × bu
SCAN_SHOULDER_MIN_MHZ = 0.10     # minimum gölge yarıçapı (dar taşıyıcının yakın etekleri için)
SCAN_PROMINENCE_DB = 12.0        # CFAR: tespit, YEREL çevre tabanını bu kadar aşmalı (varsayılan
                                 # hassasiyet). Güçlü taşıyıcının yükselttiği DÜZ gürültü tabanını eler;
                                 # gerçek TEPE'yi tutar. Sürgüyle ayarlanabilir (set_scan_sensitivity).
# --- OTOMATİK KAZANÇ KONTROLÜ (AGC) ---
RX_GAIN_MAX_DB = 76.0             # B200mini RX kazanç üst sınırı
AGC_INTERVAL_SEC = 0.3            # AGC en fazla bu sıklıkta kazanç değiştirir (donanım otursun)
AGC_CLIP_PEAK = 0.85             # tepe genlik bunu aşarsa ACİL kazanç düşür (ADC doygunluğu yakın)
AGC_TARGET_HI = 0.70             # hedef tepe üst sınırı (üstünde yavaş düşür)
AGC_TARGET_LO = 0.12             # hedef tepe alt sınırı (altında yükselt, sinyal zayıf)
AGC_STEP_DOWN_DB = 3.0           # doygunluğa yakınken hızlı düşüş
AGC_STEP_UP_DB = 2.0             # zayıf sinyalde yavaş yükseliş
# AGC YÜKSELİŞ TAVANI: sessizlikte (sinyal yokken) kazancı donanım tavanına (76 dB) kadar
# tırmandırmak, PTT'ye basıldığında güçlü sinyalin ADC'yi sert kırpmasına yol açar (kırpma ->
# IQ dengesini bozar -> ayna görüntüsü + harmonik 'çim'). AGC KENDİLİĞİNDEN en fazla buraya kadar
# yükselir; operatör elle daha yükseğe alabilir. 58 dB hâlâ yüksek hassasiyet sağlar.
AGC_HUNT_CEILING_DB = 58.0


class SDRWorker(QThread):
    data_ready = pyqtSignal(dict)
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(bool)
    chat_signal = pyqtSignal(dict)     # aux<->merkez sohbet mesajı {from, text, ts}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False

        # --- YÖN BULMA (DF) MODLARI ---
        self.is_auto_df = True
        self.manual_angles = (0.0, 0.0, 0.0)

        # --- GERÇEK DF/KONUM ALTYAPISI (genlik-tabanlı + 3B LOB üçgenleme) ---
        # Düğüm konumları (yerel ENU metre) config'ten; hangisi bizim ana (self) düğüm.
        self.df_registry = load_node_registry()
        self.self_id = self_node_id(self.df_registry)
        # Uzak düğümlerden ağ (UDP/JSON) ile gelen + yerel üretilen kerterizlerin thread-safe deposu.
        self.node_store = NodeBearingStore(stale_sec=2.0)   # 5.0 idi: ölü/donuk aux'un ESKİ kerterizi
        # 5 sn füzyona girip hareketli hedefte konumu geriye çekiyordu. Aux 10 Hz gönderir -> 2 sn yeter.
        # Yerel (ana) düğümün genlik-tabanlı kerteriz kestiricisi (enkoder azimutu + ölçülen genlik).
        # freq_hz pattern eşleştirme için gerekir; center_freq_mhz aşağıda ayarlanınca set_freq ile verilir.
        self.self_amp_df = AmplitudeDFEstimator()

        # ENKODER (ESP32/AS5600) — GPS'TEN ÖNCE bağlanır. ESP32-S3 native USB /dev/ttyACM0'da görünür;
        # GPS de aynı varsayılana sahip. GPS önce açarsa ESP32'nin portunu kapıp tutuyordu -> enkoder
        # bağlanamıyordu. Bu yüzden aday portları (ttyACM0/ttyUSB0/glob) önce ENKODER dener; ilk AÇILAN
        # porta bağlanır. Aldığı port GPS'e yasaklanır (aynı porta iki nesne açılamaz).
        self.hw_ctrl = None
        import glob as _glob
        _enc_cands = ['/dev/ttyACM0', '/dev/ttyUSB0'] + sorted(_glob.glob('/dev/ttyACM*') + _glob.glob('/dev/ttyUSB*'))
        _seen = set()
        for _p in _enc_cands:
            if _p in _seen:
                continue
            _seen.add(_p)
            _hc = HardwareController(port=_p)
            # ANGLE DOĞRULAMASI: yalnızca gerçekten "ANGLE:" verisi GÖNDEREN porta bağlan. ESP32-S3
            # iki CDC arayüzü yaratır; yanlış (sessiz) porta bağlanınca "bağlı" görünüp açı GELMİYORDU.
            if _hc.connect(verify_angle=True, verify_timeout=1.0):
                self.hw_ctrl = _hc
                break
        if self.hw_ctrl is None:                       # ANGLE gören port yoksa yine de bir nesne dursun
            self.hw_ctrl = HardwareController(port='/dev/ttyACM0')   # is_connected=False (dürüst: açı yok)
        _enc_port = self.hw_ctrl.port if self.hw_ctrl.is_connected else None

        # NOT: HAREKETLİ TEK ALICI (spec 5.1.5, GPS'li platform) + GPSReceiver KALDIRILDI —
        # saha kurulumu 3 SABİT istasyon. Konum bulma 3-düğüm LOB üçgenlemesiyle yapılıyor.
        self.udp_thread = None
        self.udp_socket = None
        # SOHBET (aux<->merkez, UDP 5006). Merkez=hub: gelen mesajı diğer düğümlere dağıtır.
        self.chat_thread = None
        self.chat_socket = None
        self._chat_peers = {}          # node_id -> ip (kerteriz/canlı paketlerin adresinden öğrenilir)
        self.CHAT_PORT = 5006

        self.center_freq_mhz = 2400.0
        self.self_amp_df.set_freq(self.center_freq_mhz * 1e6)   # pattern eşleştirme frekansı
        self.gain_db = 40.0
        self.tx_gain_db = TX_GAIN_DEFAULT_DB   # TX RF kazancı (dB); artık arayüzden ayarlanır (bulgu #2)
        # Bant genişliği (spektrumda gösterilen frekans aralığının genişliği) her zaman
        # gerçek örnekleme hızına (sample_rate) eşit olmalıdır; aksi halde FFT ekseni yanlış
        # ölçekte çizilir. İkisi tek noktadan (sample_rate) türetilir.
        self.sample_rate = 2.4e6
        self.bandwidth_mhz = self.sample_rate / 1e6
        self.antenna_port = "TX/RX"

        # Gerçek zamanlı ses dinleme (spec 5.1.3): ring buffer -> demod -> hoparlör.
        # Tembel (lazy) kurulur: yalnızca donanım çalışırken ve kullanıcı "Dinle" dediğinde.
        self.audio_player = None
        self.audio_mode = "FM"
        self.audio_squelch_snr_db = AUDIO_SQUELCH_SNR_DB   # ses squelch eşiği (ayarlanabilir)

        # Sinyal izleme/takip (spec 5.1.3): ayarlı frekansa kilitlen, sürekliliği + parametre
        # geçmişini (frekans/güç/BW sapması) izle. Tarama/TX sırasında askıya alınır.
        self.monitor = SignalMonitor()

        self.start_time = 0.0
        
        # Algoritma Motorları
        self.dsp = DSPProcessor(fft_size=FFT_POINTS, antenna_spacing_m=0.0625)
        # Ayarlı (tuned) görünüm için tarihsel-min gürültü tabanı — düşük-SNR/geniş-bant sağlamlığı.
        self.tuned_nf = NoiseFloorTracker()
        # ZAMAN-ORTALAMALI tespit spektrumu (doğrusal güç EMA'sı): gürültü varyansını düşürür ->
        # zayıf ama KALICI sinyaller ortaya çıkar (şelalenin gözle yaptığı temporal integrasyon).
        self._det_spectrum = None
        self.jam_gen = JammingGenerator(sample_rate_hz=self.sample_rate)
        self.gnss_gen = GNSSSpoofer(sample_rate_hz=self.sample_rate)
        # Otomatik Modülasyon Sınıflandırıcı (Faz 1). Pahalı olduğu için her frame değil,
        # ~0.5 sn'de bir çağrılır (bkz. run döngüsü). Örnekleme hızına bağlı -> set_bandwidth'te yenilenir.
        self.classifier = SignalClassifier(sample_rate_hz=self.sample_rate)
        self._clf_result = {"modulation": "Ölçülüyor...", "confidence": 0.0}
        self._last_classify_time = 0.0
        self._detected_ctcss_hz = 0.0     # analog FM CTCSS alt-ses tonu (aldatma için, 5.2.3)
        self._squelch_type = None         # "CTCSS" / "DCS" / None (tespit edilen squelch türü)
        self._squelch_subaudio = None     # DCS: yakalanan alt-ses kodu (aldatmada geri-oynatılır)
        self._squelch_rate = 0.0
        # PARAMETRE KONSOLİDATÖRÜ (5.1.2 stabilizasyon): ~2 Hz gürültülü sınıflandırmaları 3 sn
        # penceresinde çoğunluk oyu + medyan ile KARARLI, güven-etiketli tek çıktıya indirger.
        self.param_consol = ParameterConsolidator(window_sec=3.0)
        # FHSS zaman-geçmişi izleyici: her kare tepe-frekansı biriktirip zaman içinde atlama
        # tespit eder (tek-blok tespitinin fiziksel imkansızlığını çözer — uzman eleştirisi #1).
        self.hop_tracker = HoppingHistoryTracker(window_sec=2.0, min_snr_db=CLASSIFY_TRIGGER_DB)

        # TX dalga-şekli üreticisi (God Object'ten ayrıldı, bulgu #4). Buffer üretimi + hedef
        # profili + DAC güvenliği burada; SDRWorker yalnızca donanımı sürer.
        self.tx_builder = TxWaveformBuilder(self.jam_gen, self.gnss_gen, self.sample_rate)

        # Yön Bulma (DF) Doğruluk Takipçisi (5.1.4 - "Derece RMS" metriği için, ED/pasif)
        self.df_tracker = DFAccuracyTracker(max_samples=200)
        self._last_df_log_time = 0.0
        
        # Görev Kayıt (Logger) Motoru
        self.logger = MissionLogger(db_name="sdr_mission_logs.db")
        self.last_log_time = 0.0
        
        # (Enkoder/ESP32 bağlantısı yukarıda, GPS'ten ÖNCE yapıldı — port çakışması için.)

        # DSP (Gerçek Ses Demodülatörü)
        self.audio_demod = AudioDemodulator(sample_rate=self.sample_rate, audio_rate=48000)

        # Otomatik Kazanç Kontrolü (AGC): tepe genliği doygunluk-altı ideal bantta tutar.
        self.agc_enabled = True
        self._last_agc_time = 0.0
        
        # Sinyal Sınıflandırma (Hold-Time) için zaman tutucu
        self._last_valid_sig_time = 0.0

        # RF BANT TARAMA / SİNYAL TESPİTİ (şartname 5.1.1): merkez frekansı bir aralıkta süpürüp
        # gürültü üstü sinyalleri frekans+güçle yakalar. Frekanslar bilinmiyorken kaynakları bulmak için.
        self.scan_active = False
        self.scan_start_mhz = 400.0
        self.scan_stop_mhz = 2500.0
        self.scan_step_mhz = 2.0
        self.scan_cursor_mhz = 0.0
        self._scan_settle_until = 0.0
        self.scan_detections = {}     # key -> {freq_mhz, power_dbfs, snr_db, bw_mhz, ts, count} (Max-Hold)
        self._scan_candidates = {}    # key -> aday (onay için); SCAN_CONFIRM_HITS turda görülünce PROMOTE
        self.scan_prominence_db = SCAN_PROMINENCE_DB   # CFAR yerel-belirginlik eşiği (sürgü ayarlar)
        self._nf_trackers = {}        # merkez-frekans -> NoiseFloorTracker (tarihsel-min gürültü tabanı)

        # Kapalı-çevrim akıllı karıştırma: son ölçülen hedef sinyalin bandı/offseti (RX'ten).
        # Jamming başlarken bu banda odaklanılır (gücü tüm banda değil hedefe topla).
        self._last_measured_bw_hz = 0.0
        self._last_peak_offset_hz = 0.0

        # TX Durum Değişkenleri
        self.tx_active = False
        self.tx_params = {
            "mode": "NONE",
            "jsr_db": 15.0,
            # ARABAKIŞ (saniye cinsinden) + otomatik hedef tarama varsayılanları.
            # NOT: eski "duty_percent" / "look_time_ms" KALDIRILDI — arabakış artık yüzde/ms yerine
            # doğrudan KARIŞTIRMA (jam_sec) ve DİNLEME (listen_sec) SANİYELERİ ile çalışır.
            "jam_sec": 5.0,
            "listen_sec": 2.0,
            "lt_auto_scan": True,
            "lt_scan_start_mhz": 430.0,
            "lt_scan_stop_mhz": 440.0,
            "wave_type": "Sinüs Dalga (Tone)",
            "offset_ms": 12.0
        }

        # Donanım (USRP) durum bayrağı. Gerçek RX/TX akışları, stream nesneleri ve
        # I/O thread'leri C++ tarafında (sdr_core.SDREngine) yönetilir; Python yalnızca
        # motoru sürer. (Önceki sürümdeki rx_stream/tx_stream/rx_thread/io_chunk_size vb.
        # alanlar RX C++'a taşındıktan sonra ölü kalmıştı; kaldırıldı.)
        self.use_hardware = False

        # dump_raw_iq() için en son IQ anlık görüntüsü (thread'ler arası güvenli erişim).
        self.iq_lock = threading.Lock()
        self.latest_iq = np.zeros(FFT_POINTS, dtype=np.complex64)

    def _init_hardware(self):
        """C++ sdr_core kütüphanesi üzerinden donanımı başlatır."""
        if not SOAPY_AVAILABLE:
            self.log_signal.emit("Backend: sdr_core modülü bulunamadı! Simülasyon moda geçiliyor.")
            return False

        try:
            # Örnekleme hızı (Master Clock) değişmişse donanımı kökten yeniden başlatmak ZORUNLUDUR.
            # Aksi halde UHD "unexpected sid" vererek çöker.
            if getattr(self, 'hardware_initialized', False) and hasattr(self, 'engine'):
                if getattr(self, 'last_hw_sample_rate', 0) == self.sample_rate:
                    self.engine.set_frequency(self.center_freq_mhz * 1e6)
                    self.engine.set_gain(self.gain_db)
                    self.engine.set_antenna(self.antenna_port)
                    self.use_hardware = True
                    return True
                else:
                    self.log_signal.emit("Bant Genişliği Değişimi Algılandı. Donanım yeniden başlatılıyor...")
                    # Eski motoru DETERMİNİSTİK kapat: C++ SDREngine yıkıcısı cihazı (unmake)
                    # serbest bırakır. Yalnızca None atamak Python GC'ye bağlıdır; cihaz kilidi
                    # yeni motor yaratılana kadar açık kalıp "device busy" hatası verebilir.
                    # Önce stream'leri durdur, sonra referansı düşür (del ile refcount=0 -> yıkıcı).
                    try:
                        self.engine.stop()
                    except Exception as exc:
                        self.log_signal.emit(f"Backend: Eski motor durdurulurken uyarı: {exc}")
                    del self.engine
                    self.hardware_initialized = False

            self.engine = sdr_core.SDREngine()
            self.use_hardware = self.engine.init_hardware(self.sample_rate, self.center_freq_mhz * 1e6, self.gain_db)
            if not self.use_hardware:
                self.log_signal.emit("Backend: C++ Donanım Motoru başlatılamadı!")
                return False
                
            self.engine.set_antenna(self.antenna_port)
            
            self.hardware_initialized = True
            self.last_hw_sample_rate = self.sample_rate
            self.log_signal.emit("Backend: C++ Motoru (sdr_core) üzerinden USRP BAŞARIYLA BAĞLANDI!")
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_signal.emit(f"Backend: Donanım başlatılamadı ({str(e)}). Lütfen terminaldeki detaya (traceback) bakın.")
            return False

    def _build_tx_buffer(self) -> np.ndarray:
        """Seçili TX moduna göre donanımın looplayacağı DAC-güvenli baseband tamponunu üretir.
        Üretim mantığı TxWaveformBuilder'a taşındı (God Object'ten ayrıştırma, bulgu #4); burada
        yalnızca üreticiyi çağırıp moda özgü ek log (ör. GNSS Doppler) veriyoruz."""
        
        # Dinamik modlar (GNSS DYNAMIC_DRIFT vb.) için sürekli değişen zamana (time_sec) ihtiyaç var.
        # Motor ne kadar süredir çalışıyor:
        self.tx_params["time_sec"] = time.time() - getattr(self, "start_time", time.time())

        # GPS-SDR-SIM (gerçek ephemeris) GPS L1: önceden üretilen baseband dosyasını (tools/gen_gps_spoof.py)
        # DOĞRUDAN yayınla -> gerçek alıcı sahte konuma kilitlenir. Kendi sentetik üretecimiz atlanır.
        if (self.tx_params.get("mode") == "GNSS_SPOOF"
                and self.tx_params.get("spoof_mode") == "GPSSDRSIM_L1"):
            try:
                from backend import gps_sdr_sim as _gss
                _bin = self.tx_params.get("gpssim_bin") or os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gpssim_l1.bin")
                buf = _gss.load_baseband(_bin, iq_bits=16, amplitude=0.9)
                self.log_signal.emit(f"Backend: GPS-SDR-SIM baseband yüklendi -> {_bin} "
                                     f"({len(buf)} örnek, GERÇEK ephemeris, GPS L1)")
                return buf
            except Exception as exc:
                self.log_signal.emit(f"⚠️ GPS-SDR-SIM baseband yüklenemedi ({exc}). "
                                     f"Önce: python tools/gen_gps_spoof.py --rinex ... --lat ... --lon ...")
                # düş: kendi sentetik GPS L1 üretecine geri dön

        buf, meta = self.tx_builder.build(self.tx_params)
        
        # NOT (KRİTİK DÜZELTME): C++ "otonom GNSS motoru" (update_gnss_sats -> gnss_active) DEVRE DIŞI.
        # O yol donanımda YALNIZCA GPS C/A üretir (GLONASS FDMA / Galileo·Beidou BOC YOK) ve aktifken
        # tx_buffer'ı (yani burada üretilen ZENGİN çok-sistem Python baseband'ini) TAMAMEN yok sayardı.
        # Sonuç: GALILEO/GLONASS/BEIDOU seçilse bile o frekansta GPS C/A yayınlanır (yanlış sistem) VE
        # gnss_active bir daha false'a dönmediği için sonraki TÜM modlar (baraj/spot/analog aldatma)
        # da bozulup GNSS yayınlar kalırdı (yeniden başlatana dek). Bu yüzden çok-sistem Python
        # baseband'i (buf) doğrudan set_tx_buffer ile gönderilir; C++ GNSS yolu kullanılmaz.

        if "gnss_service" in meta:
            spec = self.gnss_gen.GNSS_SIGNAL_SPEC.get(meta.get("gnss_service", ""), {})
            mod = spec.get("mod", "BPSK"); fam = spec.get("fam", "?")
            fdma = " + FDMA" if spec.get("fdma") else ""
            coords = meta.get('target_coords', '?')
            s_mode = meta.get('spoof_mode', 'MANUAL')
            
            self.log_signal.emit(
                f"Backend: GNSS baseband üretildi -> {meta.get('gnss_service')} "
                f"({fam} {mod}{fdma}, ≥4 uydu + nav data) | Mod: {s_mode} | Koordinat: [{coords}]")
        return buf

    def trigger_tx(self, params: dict):
        self.tx_params.update(params)

        # TX RF KAZANCI (bulgu #2): operatör arayüzden verir; koda gömülü sabit yerine kullanılır.
        # B200mini üst sınırına (TX_GAIN_MAX_DB) ve 0'a güvenli şekilde kırpılır.
        if "tx_gain_db" in params:
            try:
                self.tx_gain_db = float(np.clip(float(params["tx_gain_db"]), 0.0, TX_GAIN_MAX_DB))
            except (TypeError, ValueError):
                pass

        # KARIŞTIRMA FREKANSI (sürekli karıştırma): operatör TX ekranından girdiği frekansa geç.
        # (GNSS/Wi-Fi modları aşağıda kendi bandına zaten otomatik geçer; onlarda bu atlanır.)
        if params.get("tx_freq_mhz") and self.tx_params.get("mode") not in ("GNSS_SPOOF", "WIFI_JAMMING"):
            try:
                self.set_frequency(float(params["tx_freq_mhz"]))
                self.log_signal.emit(f"Backend: Karıştırma frekansı -> {float(params['tx_freq_mhz']):.3f} MHz")
            except (TypeError, ValueError):
                pass

        # GNSS aldatma: seçilen servisin taşıyıcı frekansına OTOMATİK geç (GPS L1/L2/L5, GLONASS,
        # Galileo, Beidou...). Aksi halde yanlış frekansta "GNSS" yayını yapılır.
        if self.tx_params.get("mode") == "GNSS_SPOOF":
            service = self.tx_params.get("gnss_code", "GPS L1")
            freq_mhz = self.gnss_gen.GNSS_SERVICES.get(service)
            if freq_mhz is None:                          # bilinmeyen/eski etiket -> GPS L1
                service, freq_mhz = "GPS L1", 1575.42
            if abs(self.center_freq_mhz - freq_mhz) > 1e-3:
                self.set_frequency(freq_mhz)
            self.log_signal.emit(f"Backend: GNSS aldatma servisi -> {service} ({freq_mhz:.3f} MHz)")
            # GPS-SDR-SIM (gerçek ephemeris): baseband 2.6 Msps üretilir -> SDR de 2.6 MHz olmalı.
            if self.tx_params.get("spoof_mode") == "GPSSDRSIM_L1" and abs(self.sample_rate - 2.6e6) > 0.15e6:
                self.log_signal.emit(f"⚠️ GPS-SDR-SIM: baseband 2.6 Msps'tir; Bant Genişliğini "
                                     f"2.6 MHz yapıp yeniden başlatın (şu an {self.sample_rate/1e6:.2f} MHz).")
            # Servisin kod-oranı/BOC/FDMA'sının TEMİZ üretimi için önerilen örnekleme hızı. Çalışan
            # akışta örnekleme hızını değiştirmek UHD'yi çökertebildiği için OTOMATİK değiştirmiyoruz;
            # yetersizse operatörü uyarıyoruz (sinyal yine üretilir, ama düşük hızda aliaslanabilir).
            rec_fs = self.gnss_gen.recommended_fs_hz(service)
            if self.sample_rate < rec_fs * 0.95:
                self.log_signal.emit(
                    f"⚠️ GNSS: {service} için önerilen örnekleme ≥{rec_fs/1e6:.1f} MHz "
                    f"(şu an {self.sample_rate/1e6:.1f} MHz). Temiz kod-oranı/BOC/FDMA için "
                    f"Bant Genişliğini {rec_fs/1e6:.1f} MHz yapıp yayını yeniden başlatın.")

        # ANALOG ALDATMA — GERÇEK SES MESAJI (5.2.3): WAV dalga-şekli seçildiyse dosyayı YÜKLE ve
        # örneklerini tx_params'a koy. Hedef analog telsiz gerçek-dışı ama ANLAŞILIR sahte yayını
        # çalar ('yanlış duyar'). Dosya yoksa/bozuksa dürüstçe uyar, sentetik ses üretme.
        if (self.tx_params.get("mode") == "ANALOG_SPOOF"
                and self.tx_params.get("wave_type") == WAV_DECEPTION_WAVE):
            path = self.tx_params.get("decept_audio_file", "")
            try:
                audio, arate = load_wav_mono(path)
                self.tx_params["decept_audio"] = audio
                self.tx_params["decept_audio_rate"] = arate
                # SQUELCH OTOMATİK-TAŞIMA (5.2.3): CTCSS yoksa ama DCS tespit edildiyse, hedefin
                # yakalanan alt-ses kodunu aldatmaya geçir (geri-oynatma -> kod-squelch açılır).
                if (not self.tx_params.get("decept_ctcss_hz")
                        and getattr(self, "_squelch_type", None) == "DCS"
                        and getattr(self, "_squelch_subaudio", None) is not None):
                    self.tx_params["decept_subaudio"] = self._squelch_subaudio
                    self.tx_params["decept_subaudio_rate"] = float(getattr(self, "_squelch_rate", 0.0))
                    self.log_signal.emit("Backend: Aldatma -> DCS alt-ses kodu geri-oynatılacak "
                                         "(hedefin kod-squelch'i otomatik açılır).")
                else:
                    self.tx_params.pop("decept_subaudio", None)
                dur = len(audio) / float(arate) if arate else 0.0
                self.log_signal.emit(f"Backend: Aldatma ses mesajı yüklendi -> {path} "
                                     f"({dur:.1f} s, {arate} Hz, {self.tx_params.get('decept_mod','NBFM')})")
            except Exception as exc:
                self.tx_params.pop("decept_audio", None)
                self.log_signal.emit(f"⚠️ Aldatma: ses dosyası yüklenemedi ({exc}). "
                                     f"Geçerli bir WAV seçin — sahte ses üretilmeyecek.")

        # WI-FI ENGELLEME (bulgu #9): artık gerçek baraj yayını. Seçilen Wi-Fi bandının merkez
        # frekansına geçilir (2.4 GHz -> kanal 6, 5.8 GHz) ve geniş-bant baraj gönderilir.
        if self.tx_params.get("mode") == "WIFI_JAMMING":
            band = self.tx_params.get("wifi_band", "2.4 GHz (802.11 b/g/n)")
            freq_mhz = WIFI_BAND_CENTERS.get(band, 2437.0)
            if abs(self.center_freq_mhz - freq_mhz) > 1e-3:
                self.set_frequency(freq_mhz)
            self.log_signal.emit(f"Backend: Wi-Fi engelleme (baraj) -> {band} ({freq_mhz:.1f} MHz)")

        # KAPALI-ÇEVRİM AKILLI KARIŞTIRMA: RX'te bir hedef sinyal ölçüldüyse (bant tüm bandı
        # doldurmayan gerçek bir sinyal), baraj modlarında gücü o hedef bandına odakla. Aksi halde
        # (sinyal yok/tüm bant dolu) profil bant genişliği kullanılır.
        bw = self._last_measured_bw_hz
        if 0.0 < bw < 0.85 * self.sample_rate:
            self.tx_params["focus_bw_hz"] = bw
            self.tx_params["focus_offset_hz"] = self._last_peak_offset_hz
            self.log_signal.emit(
                f"Backend: Akıllı karıştırma -> hedef bandına odaklanıldı "
                f"(~{bw/1e3:.0f} kHz @ {self._last_peak_offset_hz/1e3:+.0f} kHz offset)")
        else:
            self.tx_params.pop("focus_bw_hz", None)
            self.tx_params.pop("focus_offset_hz", None)

        # Look-through T/R + kapalı-çevrim durum makinesini sıfırla: her tetiklemede yayın fazından
        # ve AKTİF (sinyal var varsayımı) başla. Dinleme ölçümleri durumu günceller.
        self._lt_tx_on = True
        self._lt_rx_on = False        # jam fazında başlar -> RX kapalı (bkz. _lt_set_rx / trigger_tx)
        self._lt_display_fft = None   # jam penceresinde sabit tutulacak son gerçek dinleme spektrumu
        self._lt_phase_start = time.time()
        self._lt_present = True
        self._lt_active = True
        self._lt_empty_count = 0
        self._lt_bw_hz = 0.0
        self._lt_peak_offset_hz = 0.0
        # OTOMATİK HEDEF TARAMA (round-robin) durumu — hedef susunca band tarayıp yeni aktif
        # frekansları sırayla ez. state: "JAM" | "LISTEN" | "SCAN".
        self._lt_state = "JAM"
        self._lt_targets = [float(self.center_freq_mhz)]   # ilk hedef: operatörün ayarladığı frekans
        self._lt_target_idx = 0
        self._lt_scan_cursor = 0.0
        self._lt_scan_hits = {}       # frekans -> güç (tarama sırasında biriken aktif frekanslar)
        self._lt_scan_settle_until = 0.0
        # İlk odak, tetiklemedeki RX ölçümünden geldiyse onu uygulanmış say (gereksiz rebuild olmasın)
        _fb = self.tx_params.get("focus_bw_hz")
        self._lt_focus_applied = (float(_fb), float(self.tx_params.get("focus_offset_hz", 0.0))) if _fb else None

        self.tx_pre_gen = self._build_tx_buffer()
        self._tx_sim_idx = 0  # simülasyon modunda tampon üzerinde gezinme indeksi
        self._tx_disp_ema = None  # TX spektrum ortalamasını sıfırla (yeni TX eski şekli taşımasın)

        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.set_tx_gain(self.tx_gain_db)   # TX kazancını uygula (arayüzden, kontrollü)
            self.engine.set_tx_buffer(self.tx_pre_gen)
            # JAM FAZINDA RX KAPALI: tam-çift-yönlü USB yükü (RX+TX) TX underflow'una ve zayıf jam
            # gücüne yol açar. Her iki modda da yayın RX kapalı başlar (5.2.1: almaç gerekmez).
            # ARABAKIŞ (look-through): RX yalnızca DİNLEME penceresinde açılır (_service_look_through
            # her faz geçişinde set_rx_enabled ile toggle eder) -> jam penceresi USB'yi tam kullanır,
            # underflow'suz + TAM GÜÇ; dinleme penceresinde kanal ölçülür.
            if hasattr(self.engine, 'set_rx_enabled'):
                self.engine.set_rx_enabled(False)
            self.engine.set_tx_active(True)

        self.tx_active = True
        # Full-duplex uyarısı: B200mini'de "TX/RX" tek fiziksel porttur. RX de aynı porta
        # ayarlıysa, karıştırırken eşzamanlı dinleme (jam-while-listen) fiziksel olarak
        # mümkün değildir; operatör RX antenini "RX2"ye almalıdır.
        if self._is_running and self.antenna_port == "TX/RX":
            self.log_signal.emit(
                "⚠️ UYARI: RX ve TX aynı 'TX/RX' portunu paylaşıyor. Karıştırma sırasında "
                "eşzamanlı dinleme için RX Anten Portunu 'RX2' seçin.")
        tgt = self.tx_params.get("target_signal", "")
        tgt_str = f" | Hedef: {tgt}" if tgt else ""
        self.log_signal.emit(
            f"Backend: TX Karıştırma Aktif -> {self.tx_params['mode']} "
            f"(JSR: {self.tx_params['jsr_db']} dB | TX Gain: {self.tx_gain_db:.1f} dB){tgt_str}")

    def stop_tx(self):
        self.tx_active = False
        self.tx_params["mode"] = "NONE"
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.set_tx_active(False)
            # Karıştırma bitti -> RX'i yeniden aç (spektrum/analiz/DF sürsün)
            if hasattr(self.engine, 'set_rx_enabled'):
                self.engine.set_rx_enabled(True)
        self._lt_tx_on = False
        self.log_signal.emit("Backend: TX Yayın Düzeneği Durduruldu.")

    def _run_signal_analysis(self, iq1):
        """RX SİNYAL ANALİZİ: spektrum (FFT) + event-tetiklemeli modülasyon/protokol/FHSS
        sınıflandırma. run()'dan ayrıldı (God Object azaltma). Dönüş: (fft_dbm, spectrum_info);
        yan etki: self._clf_result, self._last_measured_bw_hz/_last_peak_offset_hz güncellenir.
        DF/nirengi mantığına DOKUNMAZ (o run() içinde ayrı kalır)."""
        if self.tx_active:
            # TX sırasında HAFİF işle: tek FFT (pahalı Welch + kümülant DEĞİL) -> CPU düşük,
            # tx_worker aç kalmaz (underflow olmaz), kendi TX sinyalimiz spektrumda görünür.
            x = iq1 - np.mean(iq1)
            win = np.hamming(len(x))
            fc = np.fft.fftshift(np.fft.fft(x * win, n=FFT_POINTS))
            raw_lin = np.abs(fc) ** 2 / FFT_POINTS
            # TX SPEKTRUM ORTALAMA (GNSS/baraj gibi GÜRÜLTÜ-BENZERİ geniş-bant sinyaller için KRİTİK):
            # tek-atış FFT tepesi her karede ±yüz kHz zıplar -> spektrum "yanlış/rastgele hareket ediyor"
            # görünür (özellikle GNSS spread-spektrum, tek dominant tepe YOKTUR). Doğrusal-güç EMA (~5
            # kare) ile ortalayınca gerçek geniş-bant ŞEKLİ KARARLI çizilir — spektrum analizör 'average/
            # RMS' dedektörünün yaptığı. (LOOK_THROUGH dinleme penceresi aşağıda kendi ham FFT'sini kullanır.)
            # alpha=0.9 (~10 kare): kare-kare değişim 4.8 dB -> 0.3 dB (ölçüldü) -> spektrum KARARLI.
            _prev = getattr(self, "_tx_disp_ema", None)
            if _prev is None or getattr(_prev, "shape", None) != raw_lin.shape:
                self._tx_disp_ema = raw_lin
            else:
                self._tx_disp_ema = 0.9 * _prev + 0.1 * raw_lin
            fft_dbm = 10.0 * np.log10(self._tx_disp_ema + 1e-12)

            if self.tx_params.get("mode") == "LOOK_THROUGH":
                if getattr(self, "_lt_tx_on", True):
                    # JAM penceresi: RX KAPALI -> iq1 bayat. Bu pencerede FFT'yi yeniden hesaplama
                    # (bayat/jammer-sızıntısı veriyle ekran zıplar, seviye aşağı inmez). Bunun yerine
                    # SON GERÇEK DİNLEME görüntüsünü SABİT TUT -> ekran kararlı, gerçek kanal seviyesini
                    # (TX kapalıyken, sızıntısız) gösterir. İlk jam penceresinde henüz görüntü yoksa
                    # hesaplanan bayat kareyi kullan (tek seferlik).
                    held = getattr(self, "_lt_display_fft", None)
                    if held is not None and getattr(held, "shape", None) == fft_dbm.shape:
                        fft_dbm = held
                    spectrum_info = {"occupied_bw_hz": 0.0,
                                     "signal_class": "ARABAKIŞ: yayın penceresi (jammer aktif)",
                                     "flatness": None, "snr_db": 0.0}
                else:
                    # Dinleme penceresi: TX FİZİKSEL kapalı -> RX gerçek kanalı görür. Bu pencerede TX
                    # akmadığı için TAM spektrum (compute_fft_dbm) + analyze_spectrum kullanılır: tek-kare
                    # ham tepe-medyan gürültüde ~13 dB'e çıkıp YANLIŞ 'sinyal var' verir; analyze_spectrum
                    # (present_db, yüzdelik taban) tek-kare gürültüye karşı SAĞLAMDIR.
                    # Enerji KAZANCI (sinyal var) / KAYBI (yayın sonlandı) + hedef bandı (offset/BW) ölç.
                    fft_dbm, _lraw = self.dsp.compute_fft_dbm(iq1)
                    # Bu GERÇEK dinleme görüntüsünü sakla -> sonraki jam pencerelerinde sabit tutulur
                    # (ekran zıplamasın, seviye gerçek kanalda kalsın).
                    self._lt_display_fft = fft_dbm
                    li = self.dsp.analyze_spectrum(fft_dbm, self.sample_rate, present_db=8.0)
                    snr = float(li.get("snr_db", 0.0) or 0.0)
                    occ_bw = float(li.get("occupied_bw_hz", 0.0) or 0.0)
                    # KESİN karar: analyze_spectrum'un sağlam SNR'ı (tepe - yüzdelik taban) eşiği aşmalı.
                    # (Gürültü ~4-6 dB, gerçek hedef ~onlarca dB -> net ayrım; 'zayıf sinyal' bandı sızmaz.)
                    present = snr > LOOK_THROUGH_SIGNAL_TH_DB and occ_bw > 0.0
                    n_fft = len(fft_dbm)
                    peak_bin = int(np.argmax(fft_dbm))
                    peak_off = (peak_bin - n_fft / 2.0) / n_fft * self.sample_rate
                    # Arabakış durum makinesi için sakla
                    self._lt_present = present
                    self._lt_peak_offset_hz = peak_off if present else 0.0
                    self._lt_bw_hz = occ_bw if present else 0.0
                    self._lt_snr_db = snr        # SCAN'de hedef gücünü sıralamak için (round-robin)
                    if getattr(self, "_lt_state", "JAM") == "SCAN":
                        sig_cls = (f"ARABAKIŞ [TARAMA]: {self.center_freq_mhz:.3f} MHz "
                                   + (f"— sinyal VAR (SNR {snr:.0f} dB)" if present else "— boş"))
                    elif present:
                        sig_cls = (f"ARABAKIŞ [enerji KAZANCI]: kanalda sinyal (SNR {snr:.0f} dB, "
                                   f"~{occ_bw/1e3:.0f} kHz @ {peak_off/1e3:+.0f} kHz)")
                    else:
                        sig_cls = "ARABAKIŞ [enerji KAYBI]: kanal boş (yayın sonlandı) — hedef aranıyor"
                    spectrum_info = {"occupied_bw_hz": occ_bw, "signal_class": sig_cls,
                                     "flatness": None, "snr_db": round(snr, 1)}
            else:
                spectrum_info = {"occupied_bw_hz": 0.0, "signal_class": "TX AKTİF (kendi sinyali görünür)",
                                 "flatness": None, "snr_db": 0.0}
            self._clf_result = {"modulation": "TX AKTİF", "confidence": 0.0,
                                "multiplex": "-", "ekkt": "-", "protocol": "-"}
            return fft_dbm, spectrum_info

        # --- RX (TX kapalı): tam analiz ---
        fft_dbm, raw_fft = self.dsp.compute_fft_dbm(iq1)
        self._last_fft_dbm = fft_dbm
        # BANT TARAMA (5.1.1) için: ZAMAN-YUMUŞATILMAMIŞ (per-kare Welch) spektrumu ayrıca sakla.
        # Yumuşatılmış fft_dbm'in zaman-EMA'sı (0.4/0.6) retune'da sıfırlanmadığından ÖNCEKİ frekansın
        # spektrumunu taşır -> yavaş döngüde taramada HAYALET tespit (bir adım kaymış sahte sinyal) ya da
        # gerçek sinyalin sönümlenmesi olur. Ham Welch spektrumu yalnızca GÜNCEL kareden gelir ->
        # frekanslar arası kirlenme YOK; Welch segment-ortalaması sayesinde gürültü de düşük.
        self._last_raw_fft = raw_fft
        # RETUNE/FFT SENKRON: bu FFT'nin HANGİ merkez frekansta alındığını damgala. Tarama, retune
        # sonrası eski merkeze ait bir FFT'yi yeni merkezmiş gibi yorumlayıp SAHTE frekans üretmesin
        # (uzman #10). _service_scan bu damgayı center_freq_mhz ile karşılaştırır.
        self._last_raw_fft_center = self.center_freq_mhz
        # ZAMAN-ORTALAMA (doğrusal güç EMA, ~8 kare): gürültü varyansını düşürür -> zayıf kalıcı
        # sinyaller ortaya çıkar. Ortalanan spektrumda gürültü tepe-tabanı ~3 dB'e iner; bu yüzden
        # present_db düşürülebilir (weak sinyal yakalanır, gürültü yanlış-pozitifi olmaz).
        lin = np.power(10.0, raw_fft / 10.0)
        if self._det_spectrum is None or self._det_spectrum.shape != lin.shape:
            self._det_spectrum = lin
        else:
            self._det_spectrum = 0.82 * self._det_spectrum + 0.18 * lin
        det_dbm = 10.0 * np.log10(self._det_spectrum + 1e-12)
        nf = self.tuned_nf.update(det_dbm)     # tarihsel-min taban (ortalanmış spektrum üzerinde)
        spectrum_info = self.dsp.analyze_spectrum(det_dbm, self.sample_rate, noise_floor=nf, present_db=4.5)

        # UCUZ GÖZCÜ (her kare): sinyal var mı + FHSS zaman-geçmişini besle (event-based watcher).
        now = time.time()
        peak_bin = int(np.argmax(fft_dbm))
        sig_snr = float(fft_dbm[peak_bin] - np.median(fft_dbm))
        peak_norm = peak_bin / float(len(fft_dbm))
        self.hop_tracker.update(now, peak_norm, sig_snr)
        ekkt_hist, hop_n, _hop_info = self.hop_tracker.detect()

        # Kapalı-çevrim akıllı karıştırma için hedef bandını/offsetini sakla (RX ölçümü).
        self._last_measured_bw_hz = float(spectrum_info.get("occupied_bw_hz", 0.0) or 0.0)
        self._last_peak_offset_hz = (peak_bin - len(fft_dbm) / 2.0) / len(fft_dbm) * self.sample_rate

        # OTOMATİK MERKEZLEME (dinleme): ses HER ZAMAN TUNE EDİLEN MERKEZDE demodüle edilir (kullanıcı
        # telsiz frekansına tune eder -> sinyal DC'dedir). Auto-center yalnızca KÜÇÜK bir kaymayı
        # (±AUDIO_CENTER_MAX_HZ) düzeltir; TÜM bandı tarayıp uzak bir spur'a/DC'ye kilitlenmez. (Eski
        # "tüm bant" davranışı geniş bantta (5-10 MHz) yanlış sinyale kilitlenip telsizi kaçırıyordu.)
        # Böylece dinleme HER bant genişliğinde çalışır: yeter ki telsizin frekansına tune et.
        if getattr(self, "audio_player", None) is not None and self.audio_player.is_running():
            in_band = abs(self._last_peak_offset_hz) < AUDIO_CENTER_MAX_HZ
            off = self._last_peak_offset_hz if (sig_snr >= 6.0 and in_band) else 0.0
            self.audio_player.set_tuning_offset(off)
            # SQUELCH: kanalda gerçek sinyal (yeterli SNR) varsa aç, yoksa kapat -> gürültüde sessizlik.
            self.audio_player.set_squelch_open(sig_snr >= self.audio_squelch_snr_db)

        # OLAY-TETİKLEMELİ AĞIR AMC: yalnızca sinyal eşiği aşılınca + periyot dolunca.
        # (Bant TARAMA sırasında kapalı: pencere hızla değişir, sınıflandırma anlamsız + CPU israfı.)
        if self.use_hardware and not self.scan_active and getattr(self, "_adc_clipping", False) and sig_snr >= CLASSIFY_TRIGGER_DB:
            # SİNYAL KIRPIK (ADC doygun): AMC güvenilmez (dijital yapı yok olur, FM/FSK sanılır).
            # Yanlış sınıflandırma göstermek yerine operatörü doğrudan yönlendir + konsolidatörü besleme.
            self._last_valid_sig_time = now
            self._clf_result = {"modulation": "⚠️ Sinyal KIRPIK (ADC doygun) — Gain/atenüasyon düşür",
                                "confidence": 0.0, "multiplex": "-", "ekkt": "-", "protocol": "-"}
        elif self.use_hardware and not self.scan_active and sig_snr >= CLASSIFY_TRIGGER_DB:
            self._last_valid_sig_time = now
            if (now - self._last_classify_time) >= CLASSIFY_PERIOD_SEC:
                snap = self._get_classify_snapshot(iq1)
                self._clf_result = self.classifier.classify(snap, check_hopping=False)
                # ANALOG/SAYISAL KESİNLEŞTİRME (5.1.2): sınıflandırıcı "FM/FSK" grubu döndürdüğünde
                # (özellikten analog FM ↔ sayısal FSK ayrımı düşük SNR'de güvenilmez), SESE demodüle
                # edip kesin karar verir -> "FM (Analog)" veya "FSK/C4FM (Sayısal, ~baud)".
                if "FM/FSK" in self._clf_result.get("modulation", ""):
                    self._refine_fm_fsk(snap)
                self._clf_result["occupied_bw_hz"] = spectrum_info.get("occupied_bw_hz", 0.0)
                # PROTOKOL (5.1.2): bant planı + ölçülen (BW/modülasyon/çoklama/EKKT) -> OLASI protokol.
                self._clf_result["protocol"] = self.classifier.guess_protocol(
                    self.center_freq_mhz, self._clf_result)
                self._last_classify_time = now
                # KONSOLİDATÖRE BESLE (5.1.2 stabilizasyon): bu gerçek sınıflandırma karesini pencereye
                # ekle. Taşıyıcı = merkez + ölçülen tepe offseti. Çoğunluk oyu + medyan payload'da alınır.
                carrier = self.center_freq_mhz + float(getattr(self, "_last_peak_offset_hz", 0.0) or 0.0) / 1e6
                self.param_consol.update({
                    "modulation": self._clf_result.get("modulation"),
                    "analog_digital": self._clf_result.get("analog_digital"),
                    "multiplex": self._clf_result.get("multiplex"),
                    "ekkt": self._clf_result.get("ekkt"),
                    "protocol": self._clf_result.get("protocol"),
                    "carrier_mhz": round(carrier, 4),
                    "occupied_bw_hz": float(spectrum_info.get("occupied_bw_hz", 0.0) or 0.0),
                    "symbol_rate_hz": float(self._clf_result.get("symbol_rate_hz", 0.0) or 0.0),
                }, now)
                # TEŞHİS (sahada "dijital görünmüyor" sorunu): kararı + ham özellikleri + SNR + tepe
                # genliği (kırpma) throttle'lı logla. sdp/kurt/c40 hangi dala gidildiğini gösterir;
                # FM/FSK grubu ise refine 'reason' analog/sayısal kararının NEDENİNİ verir.
                if (now - getattr(self, "_last_clsdiag_time", 0.0)) > 2.0:
                    r = self._clf_result
                    self.log_signal.emit(
                        f"🔬 AMC: {r.get('modulation','?')} | SNR≈{r.get('snr_db','?')} dB | "
                        f"tepe={getattr(self, '_dbg_peak_amp', 0.0):.2f} (>1.0=KIRPIK) | "
                        f"sdp={r.get('sigma_dp','-')} kurt={r.get('if_kurt','-')} c40={r.get('c40','-')}"
                        + (f" | refine={getattr(self, '_dbg_refine_reason', '')}"
                           if "FSK" in r.get('modulation', '') or "FM (" in r.get('modulation', '') else ""))
                    self._last_clsdiag_time = now
        elif self.use_hardware:
            # Sinyal eşik altına düştüğünde (örn. konuşma boşluğu/fading), yazının anında
            # "Sinyal yok" olarak değişip titremesini (flickering) önlemek için 3 saniyelik "Hold Time"
            hold_time = 3.0
            if (now - self._last_valid_sig_time) > hold_time:
                self._clf_result = {"modulation": "Sinyal yok (eşik altı)", "confidence": 0.0}
                self.param_consol.reset()          # sinyal gitti -> pencereyi temizle (yeni kaynağa taşımasın)
                self._detected_ctcss_hz = 0.0      # sinyal gitti -> CTCSS tespitini de temizle
                self._squelch_type = None; self._squelch_subaudio = None

        # FHSS (frekans atlama — dijital EKKT) ÖRTÜŞÜ: zaman-geçmişi izleyicisi (HoppingHistoryTracker)
        # birçok ayrık kanal arasında sıçrama gördüyse, tek-blok AMC'nin üstüne EKKT=FHSS yaz. Bu,
        # atlama süresinden kısa tek bloğun kaçırdığı FHSS'i güvenilir yakalar (uzman eleştirisi #1).
        if ekkt_hist.startswith("FHSS"):
            self._clf_result["ekkt"] = f"FHSS (Frekans Atlama, ~{hop_n} sıçrama)"
        return fft_dbm, spectrum_info

    def _do_periodic_logging(self, target_x, target_y, df, spectrum_info, self_bearing_deg):
        """~1 Hz görev/sinyal/DF loglaması (SQLite). run()'dan ayrıldı (God Object azaltma).
        Görev hedefi + sinyal istihbaratı + DF fix + DF doğruluk (Derece RMS) kayıtları."""
        now = time.time()
        if (now - self.last_log_time) >= 1.0:
            self.logger.log_target(
                freq_mhz=self.center_freq_mhz, x_km=target_x, y_km=target_y,
                tx_mode=self.tx_params["mode"] if self.tx_active else "NONE", notes="Oto-Kestirim")
            self.last_log_time = now

        # Sinyal istihbaratı: gerçek bir sinyal sınıflandırıldıysa
        mod = self._clf_result.get("modulation", "")
        is_real = mod and not any(s in mod for s in ("Sinyal yok", "Ölçülüyor", "TX AKTİF", "Belirlenemedi"))
        if is_real and not self.tx_active and (now - getattr(self, "_last_siglog_time", 0.0)) >= 1.0:
            self.logger.log_signal(
                freq_mhz=self.center_freq_mhz, modulation=mod,
                multiplex=self._clf_result.get("multiplex", "-"),
                ekkt=self._clf_result.get("ekkt", "-"),
                protocol=self._clf_result.get("protocol", "-"),
                occupied_bw_hz=float(spectrum_info.get("occupied_bw_hz", 0.0) or 0.0),
                snr_db=float(spectrum_info.get("snr_db", 0.0) or 0.0),
                confidence=float(self._clf_result.get("confidence", 0.0) or 0.0))
            self._last_siglog_time = now

        # Yön bulma/konum fix logu
        if df["fix"] and (now - getattr(self, "_last_dffix_log_time", 0.0)) >= 1.0:
            self.logger.log_df_fix(
                freq_mhz=self.center_freq_mhz, pos_xyz=df["position_xyz_m"],
                residual_m=df["residual_m"], node_count=df["active_count"],
                target_bearing_deg=df["target_bearing_deg"], target_range_m=df["target_range_m"],
                target_elevation_deg=df["target_elevation_deg"],
                nodes={nid: {k: r.get(k) for k in ("azimuth_deg", "elevation_deg", "amp_dbm",
                                                   "snr_db", "freq_mhz", "is_self")}
                       for nid, r in df["nodes"].items()})
            self._last_dffix_log_time = now

        # DF doğruluk (Derece RMS) — kalibrasyon referansı tanımlıysa
        ref = df["df_reference_deg"]
        if ref is not None and (now - self._last_df_log_time) >= 1.0:
            self.logger.log_df_accuracy(
                reference_deg=ref, measured_deg=round(self_bearing_deg or 0.0, 2),
                rms_error_deg=df["df_rms_deg"] if df["df_rms_deg"] is not None else 0.0,
                sample_count=self.df_tracker.sample_count())
            self._last_df_log_time = now

    def _robust_peak_power_dbm(self, fft_dbm) -> float:
        """Tepe etrafı ±DF_PEAK_WINDOW_BINS bin DOĞRUSAL güç entegrasyonu ile sağlam tepe-güç (dBm).
        Tek-bin max'e göre gürültüye daha dayanıklı -> genlik-DF kerterizi daha kararlı (Derece RMS↓)."""
        n = len(fft_dbm)
        peak_bin = int(np.argmax(fft_dbm))
        w = DF_PEAK_WINDOW_BINS
        lo, hi = max(0, peak_bin - w), min(n, peak_bin + w + 1)
        lin = np.power(10.0, np.asarray(fft_dbm[lo:hi]) / 10.0)
        return float(10.0 * np.log10(np.mean(lin) + 1e-12))

    def _target_power_dbm(self, fft_dbm) -> float:
        """HEDEF KANALI gücü (uzman P0.3): bandın GLOBAL tepesi DEĞİL, MERKEZ ±80 kHz'teki en güçlü
        tepe (±DF_PEAK_WINDOW_BINS entegre). Kullanıcı hedefe tune eder; yön bulurken ortamda daha
        güçlü bir parazit belirse bile ana cihaz HEDEFE sadık kalır (uzak parazit ±80 kHz dışında
        kalır -> ölçüme girmez). Hedef tam merkezde (DC) olsa bile tepe-arama onu bulur (mean-removal
        LO dikenini zaten azaltır; ayrıca DC-hariç bastırma YOK -> merkezdeki gerçek hedef silinmez)."""
        n = len(fft_dbm)
        if n < 8:
            return -120.0
        c = n // 2
        bin_hz = self.sample_rate / n if n > 0 else 1.0
        wbin = int(np.clip(80e3 / bin_hz, 8, n // 4)) if bin_hz > 0 else 8
        lo, hi = max(0, c - wbin), min(n, c + wbin + 1)
        band = np.asarray(fft_dbm[lo:hi], dtype=float)
        pk = int(np.argmax(band))
        w = DF_PEAK_WINDOW_BINS
        lo2, hi2 = max(0, pk - w), min(len(band), pk + w + 1)
        lin = np.power(10.0, band[lo2:hi2] / 10.0)
        return float(10.0 * np.log10(np.mean(lin) + 1e-12))

    def _run_direction_finding(self, iq1, spectrum_info=None):
        """GERÇEK YÖN BULMA + KONUM (genlik-tabanlı DF + 3B LOB üçgenleme).

        1) YEREL KERTERİZ: anten elle döndürülürken enkoder azimutu + ölçülen (kalibre) genlik
           AmplitudeDFEstimator'a beslenir; en yüksek genliğin azimutu = yerel düğüm kerterizi.
           Genlik örneği YALNIZCA sinyal SNR eşiğini (DF_SIGNAL_PRESENT_DB) aşınca beslenir —
           kaynak sustuğunda (sıralı yayın) gürültü tepesi haritayı kirletip sahte kerteriz üretmesin.
        2) FÜZYON: yerel + uzak (ağdan JSON) taze kerterizler, düğüm konumlarıyla 3B üçgenlenir
           -> kaynağın (x,y,z) konumu + kesişim kalitesi (kalıntı).
        3) DERECE RMS: yerel kerteriz, kalibrasyon referansıyla kıyaslanır (şartname 5.1.4).
        Dönüş: payload'a konacak df sözlüğü. (Sentetik iq2/iq3 faz-DF tamamen kaldırıldı.)"""
        now = time.time()
        # GENLİK-DF için HEDEF KANALI gücünü kullan — bandın GLOBAL en güçlüsünü DEĞİL (uzman P0.3).
        # Kullanıcı taramayla hedefi bulup ORAYA tune eder; yön bulurken ortamda başka güçlü bir sinyal
        # belirse bile ana cihaz HEDEFE sadık kalmalı (aux ile aynı yöntem). _target_power_dbm: merkez
        # ±80 kHz'te tepe ARAYIP ±W bin entegre eder (uzak parazit dışlanır, hedef merkezde bile ölçülür).
        fft_dbm = getattr(self, "_last_fft_dbm", None)
        if fft_dbm is not None and len(fft_dbm) > 0:
            self_amp = self._target_power_dbm(fft_dbm)
        else:
            self_amp = self.dsp.compute_channel_power_dbm(iq1)

        # SİNYAL VARLIĞI: analiz SNR'si (zaman-ortalamalı, sağlam). Yoksa tepe-medyan farkına düş.
        if spectrum_info is not None:
            sig_snr = float(spectrum_info.get("snr_db", 0.0) or 0.0)
        elif fft_dbm is not None and len(fft_dbm) > 0:
            sig_snr = float(self_amp - float(np.median(fft_dbm)))
        else:
            sig_snr = 0.0
        present = sig_snr >= DF_SIGNAL_PRESENT_DB
        self_bearing = None

        if not self.is_auto_df:
            # MANUEL mod: operatörün girdiği açılar doğrudan düğüm kerterizleri olarak kullanılır.
            ids = list(self.df_registry.keys())
            for i, nid in enumerate(ids[:3]):
                self.node_store.set_self_bearing(nid, self.manual_angles[i], 0.0, self_amp,
                                                 snr_db=sig_snr, freq_mhz=self.center_freq_mhz)
            self_bearing = self.manual_angles[0]
        else:
            # OTONOM (genlik-tabanlı): enkoder bağlıysa döndürme taramasından tepe azimutu bul.
            # Genlik haritasını YALNIZCA sinyal varken besle (gürültü kirletmesini önle); ama birikmiş
            # kerterizi her karede oku (kaynak kısa sustuğunda son geçerli kerteriz decay süresince kalır).
            if self.hw_ctrl.is_connected:
                enc_az = self.hw_ctrl.get_angle()
                if present:
                    # AGC KOMPANZASYONU (uzman P0.1): anten dönerken AGC kazancı değişirse ölçülen güç
                    # yalnızca anten yönlülüğünü DEĞİL kazanç değişimini de taşır -> DF biası. Ölçülen
                    # dBFS'ten O ANKİ kazancı ÇIKAR -> antene-referanslı (kazanç-bağımsız) güç. Böylece
                    # AGC açık kalsa da (kırpma koruması sürer) DF haritası kazançtan etkilenmez.
                    self.self_amp_df.update(enc_az, self_amp - self.gain_db, now)
                self_bearing, pk, conf, n = self.self_amp_df.bearing(now)
                if self_bearing is not None:
                    self.node_store.set_self_bearing(self.self_id, self_bearing, 0.0, self_amp,
                                                     snr_db=sig_snr, freq_mhz=self.center_freq_mhz,
                                                     sigma_deg=self.self_amp_df.last_sigma())
                else:
                    # Yerel kerteriz üretilemiyor (sinyal yok / yetersiz tarama) -> yerel kerterizi
                    # füzyondan düşür (bayat kalıp yanlış üçgenlemeye girmesin). Yalnızca ENKODER
                    # bağlıyken; aksi halde depodaki (ağ) verilerine dokunma.
                    self.node_store.set_self_bearing(self.self_id, None)

        # Derece RMS (kalibrasyon referansı tanımlıysa)
        if self_bearing is not None:
            self.df_tracker.add_measurement(self_bearing)
        df_rms_deg = self.df_tracker.rms_error_deg()

        # FÜZYON: taze kerterizleri düğüm konumlarıyla eşleyip 3B üçgenle
        active = self.node_store.active_bearings(now)
        positions, directions, sigmas = [], [], []
        for nid, rec in active.items():
            cfg = self.df_registry.get(nid)
            if not cfg:
                continue
            # FREKANS EŞLEŞTİRME (uzman P0.5): yalnızca ŞU ANKİ hedef frekansındaki (±tolerans) kerterizler
            # füzyona girsin. Farklı frekanstaki bir aux BAŞKA bir hedefi ölçüyordur -> onun kerterizi bu
            # hedefin üçgenlemesine sokulmamalı (yanlış konum önlenir). freq_mhz=0 ise bilinmiyor, dahil et.
            node_freq = float(rec.get("freq_mhz", 0.0) or 0.0)
            if node_freq > 0 and abs(node_freq - self.center_freq_mhz) > DF_FUSION_FREQ_TOL_MHZ:
                continue
            positions.append(cfg["pos"])
            directions.append(azel_to_unit(rec["azimuth_deg"], rec["elevation_deg"]))
            sigmas.append(float(rec.get("sigma_deg", 8.0)))   # pattern σ (küçük=hassas) -> 1/σ² ağırlık
        if len(positions) >= 2:
            # AĞIRLIKLI üçgenleme: pattern-eşleşmiş (küçük σ) kerterizler baskın, belirsizler az etkiler.
            pos, residual_m, fix, cross_deg = triangulate_lob(positions, directions, sigmas=sigmas)
        else:
            pos, residual_m, fix, cross_deg = (np.zeros(3), 0.0, False, 0.0)

        # Arayüz izleme: yalnızca REGISTRY'de tanımlı düğümler (bilinmeyen id'ler konumsuz olduğu
        # için PPI'da merkeze çizilip yanıltırdı; bunları eleyip logluyoruz).
        raw_snap = self.node_store.snapshot()
        snap = {}
        for nid, rec in raw_snap.items():
            cfg = self.df_registry.get(nid)
            if cfg is None:
                if (now - getattr(self, "_last_unknown_node_warn", 0.0)) > 5.0:
                    self.log_signal.emit(f"⚠️ DF: bilinmeyen düğüm id '{nid}' (df_nodes.json'da tanımlı değil) — yok sayıldı.")
                    self._last_unknown_node_warn = now
                continue
            rec["pos"] = cfg.get("pos", [0.0, 0.0, 0.0])
            rec["is_self"] = bool(cfg.get("self"))
            snap[nid] = rec

        # Ana düğümden kaynağa kerteriz/menzil (PPI için hazır)
        self_pos = self.df_registry.get(self.self_id, {}).get("pos", [0.0, 0.0, 0.0])
        tgt_az, tgt_range, tgt_el = bearing_from_positions(self_pos, pos) if fix else (0.0, 0.0, 0.0)

        # UZAK DÜĞÜM (self olmayan) CANLI BAĞLANTI DURUMU — arayüzde ANT-2/ANT-3 yeşil/kırmızı için.
        # REGISTRY tabanlı: hiç veri göndermemiş bir düğüm bile listede olur (bağlı değil=kırmızı).
        # connected = kayıt VAR ve BAYAT DEĞİL (son stale_sec içinde JSON geldi).
        live_ang = self.node_store.live_angles(now)     # {id: canlı enkoder açısı} (kerterizden bağımsız)
        remote_nodes = []
        for nid, cfg in self.df_registry.items():
            if cfg.get("self"):
                continue
            rec = raw_snap.get(nid)
            # BAĞLI: taze kerteriz VEYA taze canlı-açı paketi geldiyse (aux tepe bulmadan da canlıdır)
            connected = bool((rec is not None and not rec.get("stale", True)) or (nid in live_ang))
            remote_nodes.append({
                "id": nid,
                "connected": connected,
                "age_sec": (rec.get("age_sec") if rec is not None else None),
                "live_angle_deg": live_ang.get(nid),
            })

        return {
            "self_bearing_deg": self_bearing,
            "position_xyz_m": [round(float(v), 2) for v in pos],
            "residual_m": residual_m,
            "fix": fix,
            "fix_quality_deg": cross_deg,     # LOB kesişim açısı (GDOP): büyük=iyi geometri
            "dimensionality": ("3B" if abs(float(pos[2])) > 1e-6 else "2B (yer izdüşümü)"),
            "nodes": snap,
            "remote_nodes": remote_nodes,     # uzak düğüm canlı bağlantı durumu (ANT-2/3 yeşil/kırmızı)
            "node_live_angles": tuple(live_ang.get(nid) for nid in list(self.df_registry.keys())[:3]),
            "active_count": len(positions),
            "target_bearing_deg": tgt_az,
            "target_range_m": tgt_range,
            "target_elevation_deg": tgt_el,
            "df_rms_deg": df_rms_deg,
            "df_reference_deg": self.df_tracker.reference_deg,
            "df_sample_count": self.df_tracker.sample_count(),
        }

    def _refine_fm_fsk(self, snap):
        """FM/FSK 'Ortak' sinyali SESE demodüle edip ANALOG mı SAYISAL mı KESİN belirler (5.1.2).
        En güçlü tepeyi DC'ye kaydırır -> ham FM ayrımlayıcı (~48 kHz) -> classify_fm_or_fsk (ayrık
        seviye/dwell). Sonucu _clf_result.modulation'a yazar; böylece worker'ın analog/sayısal etiket
        mantığı ('Sayısal' -> SAYISAL KESİN, 'FM' -> ANALOG KESİN) otomatik doğru çalışır. Kararsızsa
        (is_digital None) 'FM/FSK Ortak' bırakır (dürüst). Sahte üretmez."""
        if snap is None or len(snap) < 8192:
            return
        try:
            snap = np.asarray(snap, dtype=np.complex64)
            off = float(getattr(self, "_last_peak_offset_hz", 0.0) or 0.0)
            if abs(off) > 1.0:
                t = np.arange(len(snap)) / self.sample_rate
                snap = (snap * np.exp(-1j * 2 * np.pi * off * t)).astype(np.complex64)
            # Ham FM ayrımlayıcı (de-emphasis/AGC yok) — NBFM kanalına daralt, ~48 kHz'e indir
            if (getattr(self, "_refine_demod", None) is None
                    or abs(self._refine_demod.sample_rate - self.sample_rate) > 1.0):
                self._refine_demod = StreamingDemodulator(self.sample_rate, mode="FM",
                                                          digital=True, channel_bw=12500.0)
            self._refine_demod.reset()
            disc = self._refine_demod.process(snap)
            if len(disc) < 512:
                return
            r = classify_fm_or_fsk(disc, self._refine_demod.out_rate)
            self._dbg_refine_reason = r.get("reason", "")   # teşhis log'u için (analog/sayısal nedeni)
            if r.get("is_digital") is True:
                baud = r.get("symbol_rate_hz", 0.0)
                self._clf_result["modulation"] = (f"FSK/C4FM (Sayısal-Frekans, ~{baud:.0f} baud)"
                                                  if baud else "FSK/C4FM (Sayısal-Frekans)")
                self._clf_result["analog_digital"] = "Sayısal"
            elif r.get("is_digital") is False:
                self._clf_result["modulation"] = "FM (Analog-Frekans)"
                self._clf_result["analog_digital"] = "Analog"
                # CTCSS TESPİTİ (5.2.3 için kritik): analog FM ayrımlayıcısından alt-ses ton-squelch'i
                # tespit et. Aldatmada hedefin ton-squelch'ini AÇMAK için bu ton gerekir; operatörün
                # tahmin etmesi yerine ölçülür ve aldatma sekmesine otomatik taşınır.
                try:
                    orate = self._refine_demod.out_rate
                    c = detect_ctcss(disc, orate)
                    if c.get("present") and c.get("ctcss_std_hz"):
                        self._detected_ctcss_hz = float(c["ctcss_std_hz"])
                        self._squelch_type = "CTCSS"
                        self._squelch_subaudio = None
                    else:
                        self._detected_ctcss_hz = 0.0
                        # CTCSS yok -> DCS mi? (dijital alt-ses akışı). Varsa GERÇEK alt-ses kodunu
                        # yakala (aldatmada geri-oynatılır -> hedefin kod-squelch'i açılır).
                        d = detect_dcs(disc, orate)
                        if d.get("present"):
                            self._squelch_type = "DCS"
                            self._squelch_subaudio = extract_subaudio(disc, orate)[:int(orate * 0.5)]
                            self._squelch_rate = float(orate)
                        else:
                            self._squelch_type = None
                            self._squelch_subaudio = None
                except Exception:
                    self._detected_ctcss_hz = 0.0
                    self._squelch_type = None
                    self._squelch_subaudio = None
            # is_digital None -> "FM/FSK (Frekans Mod.)" olduğu gibi kalır (dürüstçe Ortak)
        except Exception as exc:
            self.log_signal.emit(f"Analog/Sayısal kesinleştirme atlandı: {exc}")

    def _get_classify_snapshot(self, fallback_iq):
        """Sınıflandırma için ring buffer'dan KAYIPSIZ uzun kayıt al (son 2048 yerine
        CLASSIFY_SNAPSHOT_N örnek); yoksa canlı iq'ya düş. Uzun kayıt -> daha iyi frekans
        çözünürlüğü + güvenilir OFDM CP otokorelasyonu + AMC. (Ring buffer mimarisinin meyvesi.)"""
        if self.use_hardware and hasattr(self, 'engine') and hasattr(self.engine, 'get_snapshot'):
            try:
                snap = self.engine.get_snapshot(CLASSIFY_SNAPSHOT_N)
                if snap is not None and len(snap) >= 4096:
                    return snap
            except Exception:
                pass
        return fallback_iq

    def _service_look_through(self):
        """ARABAKIŞLI KARIŞTIRMA — GERÇEK T/R ZAMAN-PAYLAŞIMI + KAPALI-ÇEVRİM (5.2.2).

        Donanım TX'i, duty periyodunun yayın kısmında AÇIK, dinleme kısmında fiziksel olarak KAPALI
        (set_tx_active False) -> dinleme penceresinde RX GERÇEK kanalı görür. Dinleme ölçümü
        (_lt_present/_lt_bw_hz/_lt_peak_offset_hz, bkz. _run_signal_analysis) durum makinesini sürer:

          * ENERJİ KAZANCI (kanalda sinyal): karıştırmayı SÜRDÜR (AKTİF); gücü tespit edilen banda
            odakla — hedef offseti kayarsa TX tamponunu yeniden odaklı üret (dinamik BW takibi).
          * ENERJİ KAYBI (yayın sonlandı): LOOK_THROUGH_ENDED_WINDOWS ardışık boş pencereden sonra
            BEKLEME (STANDBY) — TX kapalı kalır, yalnızca periyodik dinleme sürer (sınırlı karıştırma
            gücü boşa harcanmaz). Sinyal geri gelince (enerji kazancı) karıştırma otomatik sürdürülür.

        Böylece 'karıştırma kaynaklarının verimli kullanılması' (temporal + spektral) sağlanır."""
        if not (self.use_hardware and hasattr(self, 'engine')):
            return
        now = time.time()
        # Süreler SANİYE cinsinden (ms değil). Donanım koruması için taban LOOK_THROUGH_MIN_PERIOD_SEC.
        jam_dur = max(LOOK_THROUGH_MIN_PERIOD_SEC, float(self.tx_params.get("jam_sec", 5.0)))
        listen_dur = max(LOOK_THROUGH_MIN_PERIOD_SEC, float(self.tx_params.get("listen_sec", 2.0)))
        auto_scan = bool(self.tx_params.get("lt_auto_scan", False))
        state = getattr(self, "_lt_state", "JAM")
        elapsed = now - getattr(self, "_lt_phase_start", now)

        if state == "SCAN":
            self._lt_scan_step(now)
            return

        if state == "JAM":
            # KARIŞTIRMA penceresi (TX açık, RX kapalı). Süresi dolunca DİNLEME'ye geç.
            if elapsed >= jam_dur:
                self.engine.set_tx_active(False)
                self._lt_set_rx(True)
                self._lt_tx_on = False
                self._lt_state = "LISTEN"
                self._lt_phase_start = now
            return

        # state == "LISTEN": TX kapalı, RX açık -> hedef hâlâ yayında mı? (present, _run_signal_analysis'te ölçülür)
        if elapsed < listen_dur:
            return
        present = bool(getattr(self, "_lt_present", False))

        if not auto_scan:
            # Klasik arabakış: tek frekans; hedef sussa da SÜREKLİ bastırma (operatör bu kanalı seçti).
            if present:
                self._lt_refocus_if_moved()
            self._lt_resume_jam(now)
            return

        # OTO-TARAMA açık — round-robin hedef yönetimi:
        if present:
            self._lt_refocus_if_moved()                       # hedef konuşuyor -> güç odakla, sıradakine geç
            self._lt_target_idx = (self._lt_target_idx + 1) % max(1, len(self._lt_targets))
        else:
            # Hedef sustu -> listeden DÜŞÜR (kalan öğe idx'e kayar, yani idx zaten sonrakini gösterir).
            if 0 <= self._lt_target_idx < len(self._lt_targets):
                dropped = self._lt_targets.pop(self._lt_target_idx)
                self.log_signal.emit(f"Arabakış: {dropped:.3f} MHz sustu -> hedeften düşürüldü.")
            if self._lt_targets:
                self._lt_target_idx %= len(self._lt_targets)

        if self._lt_targets:
            nxt = self._lt_targets[self._lt_target_idx]
            if abs(nxt - self.center_freq_mhz) > 1e-6:
                self.set_frequency(nxt, quiet=True)
                self.log_signal.emit(f"Arabakış: sıradaki hedef {nxt:.3f} MHz -> karıştırılıyor.")
            self._lt_resume_jam(now)
        else:
            self._lt_begin_scan(now)                          # bilinen hedef kalmadı -> band tara

    def _lt_resume_jam(self, now):
        """Karıştırma penceresine (JAM) dön: RX kapat, TX aç."""
        self._lt_empty_count = 0
        self._lt_active = True
        self._lt_set_rx(False)                # RX KAPALI -> USB tamamen TX'te (tam güç)
        self.engine.set_tx_active(True)
        self._lt_tx_on = True
        self._lt_state = "JAM"
        self._lt_phase_start = now

    def _lt_begin_scan(self, now):
        """SCAN durumuna geç: jam kapalı, RX açık; [start,stop] bandını süpürerek aktif frekans ara."""
        self.engine.set_tx_active(False)
        self._lt_set_rx(True)
        self._lt_tx_on = False
        self._lt_active = False          # tarama sırasında karıştırma DURAKLI (payload/arayüz için)
        self._lt_state = "SCAN"
        self._lt_scan_hits = {}
        self._lt_scan_cursor = float(self.tx_params.get("lt_scan_start_mhz", 430.0))
        self.set_frequency(self._lt_scan_cursor, quiet=True)
        self._lt_scan_settle_until = now + LOOK_THROUGH_SCAN_SETTLE_SEC
        self.log_signal.emit("Arabakış: aktif hedef kalmadı -> band taranıyor (yeni hedef aranıyor)...")

    def _lt_scan_step(self, now):
        """SCAN adımı: her retune sonrası oturmayı bekle, dinleme ölçümünden (present/peak) aktif
        frekansı kaydet, sonraki adıma geç. Band bitince hedef listesini kurup (güce göre) JAM'e döner."""
        if now < getattr(self, "_lt_scan_settle_until", 0.0):
            return
        # Bu center'da sinyal var mı? (_run_signal_analysis dinleme dalı _lt_present/offset/snr'ı güncelledi)
        if bool(getattr(self, "_lt_present", False)):
            f_hit = self.center_freq_mhz + float(getattr(self, "_lt_peak_offset_hz", 0.0)) / 1e6
            snr = float(getattr(self, "_lt_snr_db", 0.0) or 0.0)
            key = round(f_hit / SCAN_MERGE_MHZ)
            prev = self._lt_scan_hits.get(key)
            if prev is None or snr > prev[1]:
                self._lt_scan_hits[key] = (f_hit, snr)

        step = max(0.5, self.bandwidth_mhz * 0.8)     # pencereler hafif örtüşsün (kaçak olmasın)
        self._lt_scan_cursor += step
        stop = float(self.tx_params.get("lt_scan_stop_mhz", 440.0))
        if self._lt_scan_cursor > stop:
            # Tarama bitti -> aktif hedefleri GÜCE göre (azalan) sırala; round-robin listesi kur.
            hits = sorted(self._lt_scan_hits.values(), key=lambda t: -t[1])
            self._lt_targets = [round(f, 4) for (f, s) in hits]
            self._lt_target_idx = 0
            if self._lt_targets:
                self.log_signal.emit(
                    f"Arabakış: {len(self._lt_targets)} aktif hedef bulundu -> sırayla karıştırılıyor "
                    f"({', '.join(f'{f:.3f}' for f in self._lt_targets[:5])}{'...' if len(self._lt_targets) > 5 else ''} MHz).")
                self.set_frequency(self._lt_targets[0], quiet=True)
                self._lt_resume_jam(now)
            else:
                # Hiç aktif frekans yok -> baştan tekrar tara (beklemede; TX kapalı, güç boşa gitmez).
                self._lt_scan_hits = {}
                self._lt_scan_cursor = float(self.tx_params.get("lt_scan_start_mhz", 430.0))
                self.set_frequency(self._lt_scan_cursor, quiet=True)
                self._lt_scan_settle_until = now + LOOK_THROUGH_SCAN_SETTLE_SEC
            return
        self.set_frequency(self._lt_scan_cursor, quiet=True)
        self._lt_scan_settle_until = now + LOOK_THROUGH_SCAN_SETTLE_SEC

    def _lt_set_rx(self, enabled: bool):
        """Look-through T/R geçişinde RX akışını aç/kapat. Jam penceresinde RX KAPALI (USB tamamen
        TX'te -> tam güç, underflow yok); dinleme penceresinde RX AÇIK (kanal ölçülür). Tekrarlı
        aynı-durum çağrılarını (gereksiz stream toggle) elemek için son durumu hatırlar."""
        if not (self.use_hardware and hasattr(self, 'engine') and hasattr(self.engine, 'set_rx_enabled')):
            return
        if getattr(self, "_lt_rx_on", None) == enabled:
            return
        self._lt_rx_on = enabled
        self.engine.set_rx_enabled(enabled)

    def _service_gnss_drift(self, period_s: float = 1.0):
        """GNSS DYNAMIC_DRIFT: TX baseband'ini ~period_s'de bir yeniden üretir. _build_tx_buffer,
        time_sec'i güncelleyip drift'li koordinatla yeni çok-sistem baseband üretir; donanıma yükler.
        Böylece hedef konumu gerçekten kuzeye kayar (yalnızca looplanan sabit bloktan ibaret kalmaz)."""
        now = time.time()
        last = getattr(self, "_gnss_last_rebuild", 0.0)
        if now - last < period_s:
            return
        self._gnss_last_rebuild = now
        try:
            self.tx_pre_gen = self._build_tx_buffer()
            self._tx_sim_idx = 0
            if self.use_hardware and hasattr(self, 'engine'):
                self.engine.set_tx_buffer(self.tx_pre_gen)
        except Exception as exc:
            self.log_signal.emit(f"GNSS drift: baseband yenileme başarısız: {exc}")

    def _lt_refocus_if_moved(self):
        """Arabakış dinleme penceresinde ölçülen hedef bandı/offseti belirgin değiştiyse, TX
        tamponunu yeni odakla YENİDEN üret ve donanıma yükle (gücü tespit edilen banda aktar, 5.2.2)."""
        bw = float(getattr(self, "_lt_bw_hz", 0.0) or 0.0)
        off = float(getattr(self, "_lt_peak_offset_hz", 0.0) or 0.0)
        if not (0.0 < bw < 0.85 * self.sample_rate):
            return
        applied = getattr(self, "_lt_focus_applied", None)
        moved = (applied is None
                 or abs(off - applied[1]) > LOOK_THROUGH_REFOCUS_HZ
                 or abs(bw - applied[0]) > max(0.3 * applied[0], LOOK_THROUGH_REFOCUS_HZ))
        if not moved:
            return
        self.tx_params["focus_bw_hz"] = bw
        self.tx_params["focus_offset_hz"] = off
        try:
            self.tx_pre_gen = self._build_tx_buffer()
            self.engine.set_tx_buffer(self.tx_pre_gen)
            self._lt_focus_applied = (bw, off)
            self.log_signal.emit(f"ARABAKIŞ: güç yeniden odaklandı -> ~{bw/1e3:.0f} kHz "
                                 f"@ {off/1e3:+.0f} kHz (tespit edilen banda).")
        except Exception as exc:
            self.log_signal.emit(f"ARABAKIŞ: yeniden odaklama başarısız: {exc}")

    def _poll_stream_health(self, payload: dict):
        """C++ akış-sağlığı sayaçlarını (RX overflow / stream error / TX underflow) Python'a
        yansıtır. Sayaçlar C++'ta zaten tutuluyordu ama hiçbir yere aktarılmıyordu; artık
        payload'a eklenir ve artış olduğunda operatöre loglanır (bulgu #6 tamamlayıcısı)."""
        if not (self.use_hardware and hasattr(self, 'engine')):
            return
        try:
            ov = int(self.engine.get_overflow_count())
            se = int(self.engine.get_stream_error_count())
            uf = int(self.engine.get_tx_underflow_count())
        except Exception:
            return
        payload["rx_overflow"] = ov
        payload["rx_stream_error"] = se
        payload["tx_underflow"] = uf

        now = time.time()
        if (now - getattr(self, "_last_health_time", 0.0)) >= HEALTH_POLL_SEC:
            prev = getattr(self, "_last_health", (0, 0, 0))
            d_ov, d_se, d_uf = ov - prev[0], se - prev[1], uf - prev[2]
            if d_ov or d_se or d_uf:
                self.log_signal.emit(
                    f"Backend: Akış sağlığı — RX overflow +{d_ov} (top {ov}), "
                    f"stream hata +{d_se} (top {se}), TX underflow +{d_uf} (top {uf})")
            self._last_health = (ov, se, uf)
            self._last_health_time = now

    def set_frequency(self, freq_mhz: float, quiet: bool = False):
        self.center_freq_mhz = float(freq_mhz)
        # Pattern eşleştirme frekansa bağlı -> DF kestiriciye o anki frekansı bildir (kerteriz
        # o frekansın kalibre pattern'iyle oturtulur; pattern yoksa etkisiz).
        self.self_amp_df.set_freq(self.center_freq_mhz * 1e6)
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.set_frequency(self.center_freq_mhz * 1e6)
        if not quiet:
            # Tarama (scan) sırasında her adımda log/DF-reset yapmak istemeyiz -> quiet=True.
            self.hop_tracker.reset()      # bant değişti -> eski tepe-frekans geçmişi anlamsız
            self.self_amp_df.reset()      # frekans değişti -> eski azimut-genlik haritası geçersiz
            self.tuned_nf.reset()         # bant değişti -> tarihsel-min gürültü tabanı yeniden öğrenilmeli
            self._det_spectrum = None     # bant değişti -> zaman-ortalama sıfırlanmalı
            if hasattr(self, "param_consol"):
                self.param_consol.reset()   # yeni frekans = yeni kaynak -> parametre penceresini temizle
            self._detected_ctcss_hz = 0.0   # yeni kaynak -> CTCSS/DCS tespitini temizle
            self._squelch_type = None; self._squelch_subaudio = None
            self.log_signal.emit(f"Backend: Taşıyıcı Frekansı güncellendi -> {self.center_freq_mhz} MHz")

    # ------------------------------------------------------------------ #
    #  RF BANT TARAMA / SİNYAL TESPİTİ (şartname 5.1.1)
    # ------------------------------------------------------------------ #
    def start_scan_rf(self, start_mhz: float, stop_mhz: float):
        """Merkez frekansı [start, stop] aralığında süpürerek gürültü üstü sinyalleri otomatik
        tespit etmeye başlar. Gerçek donanım gerektirir (sim'de tespit üretmez — sahte yok)."""
        lo, hi = sorted((float(start_mhz), float(stop_mhz)))
        lo = max(70.0, lo)
        hi = min(6000.0, hi)
        self.scan_start_mhz, self.scan_stop_mhz = lo, hi
        self.scan_step_mhz = max(0.5, self.bandwidth_mhz * 0.8)   # %20 örtüşme: pencere KENARINDAKİ
                                                                  # sinyal komşu pencerede merkeze yakın
                                                                  # yakalanır (uzman #11); hız kaybı ~%12
        self.scan_cursor_mhz = lo
        self.scan_detections = {}
        self._scan_candidates = {}      # onaylanmamış adaylar: key -> {..., hits, first_ts, last_ts}
        self._nf_trackers = {}          # tarihsel-min gürültü tabanları sıfırlanır
        self.set_frequency(lo, quiet=True)
        self._scan_settle_until = time.time() + SCAN_SETTLE_SEC
        self.scan_active = True
        self.log_signal.emit(
            f"🔍 BANT TARAMA BAŞLADI: {lo:.1f}–{hi:.1f} MHz (adım {self.scan_step_mhz:.1f} MHz, "
            f"eşik gürültü+{SCAN_DETECT_DB:.0f} dB)")

    def set_scan_sensitivity(self, prominence_db: float):
        """Tarama hassasiyeti (CFAR yerel-belirginlik eşiği, dB). DÜŞÜK = daha hassas (zayıf sinyali
        de yakalar, gürültü artabilir); YÜKSEK = daha seçici (sadece net sinyaller, temiz liste).
        Arayüzdeki sürgü bunu çağırır. Makul aralık ~6–22 dB; varsayılan SCAN_PROMINENCE_DB."""
        self.scan_prominence_db = float(np.clip(prominence_db, 3.0, 30.0))
        self.log_signal.emit(f"Backend: Tarama hassasiyeti -> yerel-belirginlik ≥ {self.scan_prominence_db:.0f} dB "
                             f"({'seçici' if self.scan_prominence_db >= 15 else 'hassas' if self.scan_prominence_db <= 9 else 'dengeli'}).")

    def stop_scan_rf(self):
        self.scan_active = False
        self.log_signal.emit(f"⏹ Bant tarama durduruldu. Toplam {len(self.scan_detections)} sinyal tespit edildi.")

    def _service_scan(self):
        """Tarama adımı: mevcut pencerede (GERÇEK ölçülen FFT) gürültü üstü tepeleri tespit et,
        birleştir/logla, sonra bir sonraki frekansa geç. Yalnızca donanımda çalışır."""
        if not (self.use_hardware and hasattr(self, 'engine')):
            return
        now = time.time()
        if now < self._scan_settle_until:
            return   # retune sonrası oturma; bu pencerede tespit yapma (geçiş bozması olmasın)

        # HAM (zaman-yumuşatılmamış) Welch spektrumu kullan -> retune'da frekanslar arası EMA
        # kirlenmesi/hayalet tespit YOK (bkz. _run_signal_analysis'teki _last_raw_fft notu).
        fft = getattr(self, "_last_raw_fft", None)
        # RETUNE GUARD (uzman #10): FFT, ŞU ANKİ merkez frekansa ait değilse KULLANMA. Aksi halde
        # retune sonrası henüz taze kare gelmeden eski merkezin spektrumunu yeni merkezmiş gibi
        # yorumlayıp SAHTE frekans üretiriz (ör. 433.9'u 450 gösterir). Eşleşene dek bu turu atla.
        fft_center = getattr(self, "_last_raw_fft_center", None)
        if fft_center is None or abs(float(fft_center) - self.center_freq_mhz) > 1e-3:
            return
        if fft is not None and len(fft) > 0:
            # TARİHSEL-MİN gürültü tabanı (bu merkez frekans için) — self-masking'e bağışık (uzman #1)
            ckey = round(self.center_freq_mhz * 10)
            tracker = self._nf_trackers.setdefault(ckey, NoiseFloorTracker())
            nf = tracker.update(fft)
            # ADA/ENERJİ tespiti — dar + geniş bant birlikte, bant genişliğiyle (uzman #2).
            # DC_GUARD: pencere merkezindeki dar DC/LO sızıntı tepesini eler (B200 offset, her retune'da).
            # CLOSE_GAP: dalgalı geniş sinyal tek tespit olsun (çok parçaya bölünmesin).
            bin_hz = (self.bandwidth_mhz * 1e6) / max(1, len(fft))
            close_gap = int(SCAN_CLOSE_GAP_HZ / bin_hz) if bin_hz > 0 else 0
            for freq, pwr, snr, bw in self.dsp.detect_signals(
                    fft, nf, self.center_freq_mhz, self.bandwidth_mhz, SCAN_DETECT_DB,
                    dc_guard_bins=SCAN_DC_GUARD_BINS, close_gap_bins=close_gap,
                    prominence_db=self.scan_prominence_db):
                key = round(freq / SCAN_MERGE_MHZ)
                prev = self.scan_detections.get(key)
                if prev is not None:
                    # ZATEN ONAYLI -> Max-Hold ile güncelle (sıralı-yayın senaryosunda kalıcı kalır).
                    prev["count"] += 1
                    prev["ts"] = now
                    if pwr > prev["power_dbfs"]:
                        prev.update(freq_mhz=freq, power_dbfs=pwr, snr_db=snr, bw_mhz=bw)
                    continue
                # HENÜZ ONAYSIZ -> aday havuzunda biriktir. Tek-karelik gürültü sıçraması rastgele
                # bin'de olur; aynı frekansta SCAN_CONFIRM_HITS ayrı turda tekrar GELMEDİKÇE listeye
                # girmez -> yalancı-pozitif seli önlenir.
                cand = self._scan_candidates.get(key)
                if cand is None:
                    cand = {"freq_mhz": freq, "power_dbfs": pwr, "snr_db": snr,
                            "bw_mhz": bw, "first_ts": now, "last_ts": now, "hits": 1}
                    self._scan_candidates[key] = cand
                else:
                    cand["hits"] += 1
                    cand["last_ts"] = now
                    if pwr > cand["power_dbfs"]:                 # aday da Max-Hold ile güçlensin
                        cand.update(freq_mhz=freq, power_dbfs=pwr, snr_db=snr, bw_mhz=bw)
                # ONAY kontrolü HEM ilk görüşte HEM sonrakilerde (CONFIRM=1 -> ilk görüşte anında).
                if cand["hits"] >= SCAN_CONFIRM_HITS:            # ONAYLANDI -> listeye + loga TEK sefer
                    self.scan_detections[key] = {
                        "freq_mhz": cand["freq_mhz"], "power_dbfs": cand["power_dbfs"],
                        "snr_db": cand["snr_db"], "bw_mhz": cand["bw_mhz"], "ts": now,
                        "first_ts": cand["first_ts"], "count": cand["hits"]}
                    del self._scan_candidates[key]
                    cbw, cf = cand["bw_mhz"], cand["freq_mhz"]
                    bw_str = f", BG ~{cbw*1000:.0f} kHz" if cbw < 1.0 else f", BG ~{cbw:.1f} MHz"
                    self.log_signal.emit(f"📡 TESPİT: {cf:.3f} MHz @ {cand['power_dbfs']:.0f} dBFS "
                                         f"(SNR {cand['snr_db']:.0f} dB{bw_str})")
                    self.logger.log_detection(cf, cand["power_dbfs"], cand["snr_db"])
            # Yeniden görülmeyen (onaylanmamış) adayları düşür -> gezici gürültü birikmez.
            if self._scan_candidates:
                stale = [k for k, c in self._scan_candidates.items()
                         if now - c["last_ts"] > SCAN_CANDIDATE_TTL_SEC]
                for k in stale:
                    del self._scan_candidates[k]
            while len(self.scan_detections) > SCAN_MAX_DETECTIONS:  # cap: en zayıfları toptan buda
                weakest = min(self.scan_detections, key=lambda k: self.scan_detections[k]["power_dbfs"])
                del self.scan_detections[weakest]

        # Bir sonraki frekansa geç (sınıra ulaşınca başa sar -> sürekli izleme)
        self.scan_cursor_mhz += self.scan_step_mhz
        if self.scan_cursor_mhz > self.scan_stop_mhz:
            self.scan_cursor_mhz = self.scan_start_mhz
        self.set_frequency(self.scan_cursor_mhz, quiet=True)
        self._scan_settle_until = now + SCAN_SETTLE_SEC

    def _scan_detections_view(self):
        """Onaylı tespitleri arayüz/CSV için TEMİZLER: baskın bir taşıyıcının derin 'omuz/etek'
        gölgesindeki zayıf tespitler AYRI sinyal sayılmaz (tek güçlü sinyal düzinelerce sahte komşu
        üretmesin). NOT: Bu yalnızca GÖRÜNÜM temizliğidir; ADC doygunluğunun ürettiği geniş-bant
        spur'ları GİDERMEZ -> onun tek çözümü GAIN'i düşürmektir. Dönüş: freq'e göre sıralı liste."""
        dets = list(self.scan_detections.values())
        strong_first = sorted(dets, key=lambda d: -d["power_dbfs"])
        kept = []
        for d in strong_first:
            shadowed = False
            for s in kept:
                span = max(SCAN_SHOULDER_MIN_MHZ, s.get("bw_mhz", 0.0) * SCAN_SHOULDER_SPAN_MULT)
                if abs(d["freq_mhz"] - s["freq_mhz"]) <= span \
                        and (s["power_dbfs"] - d["power_dbfs"]) >= SCAN_SHOULDER_DB:
                    shadowed = True
                    break
            if not shadowed:
                kept.append(d)
        return sorted(
            ({"freq_mhz": round(d["freq_mhz"], 3), "power_dbfs": d["power_dbfs"],
              "snr_db": d["snr_db"], "bw_mhz": d.get("bw_mhz", 0.0),
              "first_ts": d.get("first_ts", d.get("ts", 0.0))} for d in kept),
            key=lambda x: x["freq_mhz"])

    def set_gain(self, gain_db: float):
        self.gain_db = float(gain_db)
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.set_gain(self.gain_db)
        self.log_signal.emit(f"Backend: Güç Seviyesi (Gain) güncellendi -> {self.gain_db} dB")

    def set_agc(self, enabled: bool):
        self.agc_enabled = bool(enabled)
        self.log_signal.emit(f"Backend: AGC (Otomatik Kazanç) {'AÇIK' if enabled else 'KAPALI'}.")

    def _service_agc(self, peak_amp: float):
        """OTOMATİK KAZANÇ KONTROLÜ (AGC): tepe genliği ADC doygunluğunun altında ideal bantta
        tutar. Operatörün elle gain ayarlaması gerekmez (dinamik güç süpürmesi testinde kritik).
        Histerezis + hız sınırı: kazanç en fazla AGC_INTERVAL_SEC'de bir değişir; doygunluğa
        yakınken hızlı düşer, zayıf sinyalde yavaş yükselir. TX aktifken (self-reception) devre dışı."""
        if not (self.agc_enabled and self.use_hardware and hasattr(self, 'engine')) or self.tx_active:
            return
        now = time.time()
        # ACİL DÜŞÜŞ (SERT kırpma): telsiz PTT gibi ANİ güçlü darbede normal 3 dB/0.3s çok YAVAŞ —
        # kazanç güvenliye inene dek 2+ s ADC doygun kalıp spektrumu harmonik/intermod tepeleriyle
        # doldurur (±MHz her yerde sahte tepe). Sert kırpmada (tepe≥0.95) hız/adım sınırını ATLA:
        # 0.1 s'de 10 dB düş -> kazanç ~0.3 s'de güvenliye iner, sahte tepeler hızla kaybolur.
        if peak_amp >= 0.95 and self.gain_db > 0.0 and (now - self._last_agc_time) >= 0.1:
            self.gain_db = float(np.clip(self.gain_db - 10.0, 0.0, RX_GAIN_MAX_DB))
            self.engine.set_gain(self.gain_db)
            self._last_agc_time = now
            self.log_signal.emit(f"AGC ACİL: sert kırpma (tepe {peak_amp:.2f}) -> kazanç {self.gain_db:.0f} dB")
            return
        if (now - self._last_agc_time) < AGC_INTERVAL_SEC:
            return
        new_gain = None
        if peak_amp >= AGC_CLIP_PEAK:                       # doygunluk yakın -> ACİL düşür
            new_gain = self.gain_db - AGC_STEP_DOWN_DB
        elif peak_amp > AGC_TARGET_HI:                      # biraz yüksek -> yavaş düşür
            new_gain = self.gain_db - AGC_STEP_UP_DB
        elif peak_amp < AGC_TARGET_LO and self.gain_db < AGC_HUNT_CEILING_DB:
            # zayıf -> yükselt (ANCAK yükseliş tavanına kadar; sessizlikte 76 dB'ye tırmanıp
            # PTT'de sert kırpma tuzağı kurmasın). Tavana ulaşınca daha fazla yükseltmez.
            new_gain = min(self.gain_db + AGC_STEP_UP_DB, AGC_HUNT_CEILING_DB)
        if new_gain is None:
            return
        new_gain = float(np.clip(new_gain, 0.0, RX_GAIN_MAX_DB))
        if abs(new_gain - self.gain_db) < 0.1:              # sınırda -> değişiklik yok
            return
        self.gain_db = new_gain
        self.engine.set_gain(self.gain_db)
        self._last_agc_time = now
        self.log_signal.emit(f"AGC: kazanç -> {self.gain_db:.0f} dB (tepe {peak_amp:.2f})")

    def calibrate_power_dbm(self, known_dbm: float):
        """MUTLAK GÜÇ KALİBRASYONU: sinyal jeneratörü BİLİNEN bir güç (known_dbm) yayınlarken
        çağrılır. O anki ham IQ'dan ölçülen dBFS ile arasındaki ofset hesaplanıp kalıcı saklanır;
        bundan sonra panelin gösterdiği güç gerçek dBm'e yakınsar. (Büyük test için kritik.)"""
        with self.iq_lock:
            iq = self.latest_iq.copy()
        if self.use_hardware and hasattr(self, 'engine'):
            try:
                iq = self.engine.get_snapshot(CLASSIFY_SNAPSHOT_N)
            except Exception:
                pass
        if iq is None or len(iq) < 64:
            self.log_signal.emit("Backend: Kalibrasyon başarısız — yeterli IQ örneği yok.")
            return
        offset = self.dsp.calibrate_power(iq, float(known_dbm))
        self.log_signal.emit(
            f"Backend: GÜÇ KALİBRE EDİLDİ — referans {known_dbm:.1f} dBm -> ofset {offset:+.1f} dB "
            f"(bundan sonra güçler mutlak dBm'e yakınsar).")

    def set_power_cal_offset(self, offset_db: float):
        """Kalibrasyon ofsetini elle ayarla (bilinen ofset varsa)."""
        self.dsp.set_cal_offset(float(offset_db))
        self.log_signal.emit(f"Backend: Güç kalibrasyon ofseti elle ayarlandı -> {offset_db:+.1f} dB")

    def set_bandwidth(self, bw_mhz: float):
        self.sample_rate = float(bw_mhz * 1e6)
        # Spektrum ekseni ve waterfall genişliği bandwidth_mhz'i kullanır; örnekleme
        # hızıyla senkron tutulmalı (aksi halde eksen yanlış ölçeklenir).
        self.bandwidth_mhz = float(bw_mhz)
        # Sinyal jeneratörleri yeni hıza göre baştan kurulmalı
        self.jam_gen = JammingGenerator(sample_rate_hz=self.sample_rate)
        self.gnss_gen = GNSSSpoofer(sample_rate_hz=self.sample_rate)
        self.audio_demod = AudioDemodulator(sample_rate=self.sample_rate, audio_rate=48000)
        self.classifier = SignalClassifier(sample_rate_hz=self.sample_rate)
        # TX üreticisi yeni jeneratörlere/örnekleme hızına yeniden bağlanmalı
        self.tx_builder = TxWaveformBuilder(self.jam_gen, self.gnss_gen, self.sample_rate)
        self.hop_tracker.reset()   # örnekleme hızı değişti -> geçmiş anlamsız
        self.self_amp_df.reset()   # örnekleme hızı değişti -> azimut-genlik haritası geçersiz

        self.log_signal.emit(f"Backend: Bant Genişliği (Sample Rate) güncellendi -> {bw_mhz} MHz")

        # YÜKSEK-HIZ RX UYARISI (kritik): B200mini USB'den yüksek örnekleme hızında sürekli RX,
        # host DSP'si yetişemeyince taşmaya (overflow "O" seli) ve süreç OOM ile ÖLMESİNE yol açar.
        # Dinleme/tespit için ~2.4 MHz fazlasıyla yeter (PMR kanalı 12.5 kHz). Yüksek BW YALNIZCA
        # kısa geniş-bant TARAMA veya baraj JAMMING içindir. Operatörü net uyar (sessiz çökme olmasın).
        if bw_mhz >= 10.0:
            self.log_signal.emit(
                f"⚠️ UYARI: {bw_mhz:.0f} MHz çok yüksek — bu hızda SÜREKLİ DİNLEME host'u boğar "
                f"(overflow 'O' seli + program çökmesi/OOM). Telsiz/dinleme için Bant Genişliğini "
                f"2.4 MHz yapın. Yüksek BW yalnızca kısa tarama veya baraj karıştırma içindir.")

    def set_antenna(self, port_name: str):
        self.antenna_port = port_name
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.set_antenna(self.antenna_port)
        self.log_signal.emit(f"Backend: Anten Portu değiştirildi -> {port_name}")

    def start_df_calibration(self, reference_deg: float):
        """
        Yön Bulma (DF) kalibrasyon modunu başlatır. Bilinen azimuttaki bir referans
        (kalibrasyon) vericisi kullanılarak, SDR-1 kanalının ölçtüğü kerteriz ile
        bu referans arasındaki RMS hata (Derece RMS, şartname 5.1.4) izlenir.
        """
        self.df_tracker.set_reference(reference_deg)
        self.log_signal.emit(f"Backend: DF Kalibrasyonu başlatıldı -> Referans: {reference_deg:.1f}°")

    def stop_df_calibration(self):
        """DF kalibrasyon modunu kapatır."""
        self.df_tracker.clear_reference()
        self.log_signal.emit("Backend: DF Kalibrasyonu durduruldu.")

    def get_aux_node_positions(self) -> list:
        """Yardımcı (self-olmayan) düğümlerin KUTUPSAL konumunu döndürür: [(id, mesafe_m, açı°), ...].
        Ana cihaz (self) daima 0,0'dır; listede yer almaz. Arayüz başlangıçta girdileri bununla doldurur."""
        out = []
        for nid, cfg in self.df_registry.items():
            if cfg.get("self"):
                continue
            dist, bearing = enu_to_polar(cfg.get("pos", [0.0, 0.0, 0.0]))
            out.append((nid, dist, bearing))
        return out

    def set_aux_node_positions(self, polar_list: list) -> dict:
        """Yardımcı düğümlerin konumunu MESAFE (m) + PUSULA AÇISI (°) ile ayarlar.
        polar_list: [(mesafe_m, açı°), ...] — registry sırasındaki self-olmayan düğümlerle eşlenir.
        Ana cihaz (self) daima [0,0,0] sabit tutulur. ENU'ya çevrilip df_nodes.json'a yazılır
        (kalıcı). Konum belirleme üçgenlemesi (5.1.5) bu konumları kullanır."""
        aux_ids = [nid for nid, cfg in self.df_registry.items() if not cfg.get("self")]
        applied = []
        for (dist, bearing), nid in zip(polar_list, aux_ids):
            self.df_registry[nid]["pos"] = polar_to_enu(dist, bearing)
            applied.append(f"{nid}={float(dist):.0f}m @ {float(bearing):.0f}°")
        # Ana cihaz konumu her zaman orijin (0,0,0)
        if self.self_id in self.df_registry:
            self.df_registry[self.self_id]["pos"] = [0.0, 0.0, 0.0]
        saved = save_node_registry(self.df_registry)
        self.log_signal.emit("Backend: Düğüm konumları güncellendi -> " + ", ".join(applied)
                             + (" (kaydedildi)" if saved else " (KAYDEDİLEMEDİ)"))
        return {"applied": applied, "saved": saved}

    def stop(self):
        self._is_running = False
        # Ses oynatıcıyı durdur (hoparlör akışı + üretici thread)
        self._stop_audio()
        # TX aktifse ÖNCE güvenli kapat: aksi halde yayın sürer ve ET DURUMU "AKTİF" donar.
        if self.tx_active:
            self.tx_active = False
            self.tx_params["mode"] = "NONE"
            if self.use_hardware and hasattr(self, 'engine'):
                self.engine.set_tx_active(False)
        if self.udp_socket:
            try:
                self.udp_socket.close()
            except OSError:
                # Soket zaten kapalı/geçersiz olabilir; kapatma hatası kritik değil.
                pass
        self.wait()

        # Donanımı güvenli kapatma
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.stop()

        csv_path = self.logger.export_to_csv()
        self.log_signal.emit(f"Operasyon tamamlandı. Görev raporu (CSV) oluşturuldu: {csv_path}")
        self.logger.close()

    def dump_raw_iq(self):
        """Mevcut (en son) IQ buffer'ını diske kaydeder (Uzman Madde 7)"""
        with self.iq_lock:
            iq_copy = self.latest_iq.copy()
        try:
            filename = f"raw_iq_dump_{int(time.time())}.npy"
            np.save(filename, iq_copy)
            self.log_signal.emit(f"Hata Ayıklama (Debug): Ham IQ verisi kaydedildi -> {filename}")
        except Exception as e:
            self.log_signal.emit(f"Ham IQ kaydı başarısız: {e}")

    # ------------------------------------------------------------- Ses (spec 5.1.3)
    def _ensure_audio_player(self):
        """Donanım çalışıyorsa AudioPlayer'ı (bir kez) oluşturur. Aksi halde None döner."""
        if not (self.use_hardware and hasattr(self, "engine")):
            return None
        if self.audio_player is None:
            from backend.audio_player import AudioPlayer
            self.audio_player = AudioPlayer(self.engine, self.sample_rate, mode=self.audio_mode)
        return self.audio_player

    def set_audio_mode(self, mode: str):
        """Demod modunu değiştir (FM/AM/USB/LSB). Oynatıcı yoksa yalnızca tercihi saklar."""
        self.audio_mode = (mode or "FM").upper()
        if self.audio_player is not None:
            self.audio_player.set_mode(self.audio_mode)

    def set_squelch_snr_db(self, snr_db: float):
        """Ses SQUELCH eşiği (dB SNR). DÜŞÜK = daha çok açık (zayıf sesi de duyar, cızırtı artabilir);
        YÜKSEK = daha çok kapalı (sadece net sinyalde ses, sessizlik daha fazla). 0 = squelch KAPALI
        (her şeyi duy). Makul aralık ~0–20 dB."""
        self.audio_squelch_snr_db = float(np.clip(snr_db, 0.0, 40.0))
        self.log_signal.emit(f"Backend: Ses squelch eşiği -> {self.audio_squelch_snr_db:.0f} dB "
                             f"({'kapalı (hepsini duy)' if self.audio_squelch_snr_db <= 0 else 'açık'}).")

    def set_audio_listen(self, listen: bool) -> bool:
        """Dinle/Sustur. Dinle: oynatıcıyı başlat + sesi aç. Döner: gerçekten dinleniyor mu."""
        if listen:
            player = self._ensure_audio_player()
            if player is None:
                self.log_signal.emit("Ses: Donanım aktif değil — önce alımı başlatın.")
                return False
            if not player.is_running():
                if not player.start():
                    self.log_signal.emit(f"Ses: Başlatılamadı ({player.last_error}).")
                    return False
            player.set_muted(False)
            self.log_signal.emit(f"Ses: {self.audio_mode} demodülasyonu dinleniyor.")
            return True
        else:
            if self.audio_player is not None:
                self.audio_player.set_muted(True)
            self.log_signal.emit("Ses: Susturuldu.")
            return False

    def set_digital_decode(self, enable: bool, key: str = None) -> dict:
        """Sayısal amatör telsiz çözme (spec 5.1.3): 4FSK/C4FM tespiti + harici DSD-FME köprüsü.
        key verilirse (ondalık DMR Basic Privacy anahtarı) ŞİFRELİ yayın çözülür.
        Döner: {'enabled','dsd_available','hint'}. Donanım yoksa uyarır."""
        if enable:
            player = self._ensure_audio_player()
            if player is None:
                self.log_signal.emit("Sayısal ses: Donanım aktif değil — önce alımı başlatın.")
                return {"enabled": False, "dsd_available": False, "hint": "donanım yok"}
            if not player.is_running():
                if not player.start():
                    self.log_signal.emit(f"Sayısal ses: Başlatılamadı ({player.last_error}).")
                    return {"enabled": False, "dsd_available": False, "hint": player.last_error or ""}
            info = player.set_digital(True, key=key)
            if info.get("dsd_available"):
                self.log_signal.emit("Sayısal ses: DSD-FME ile çözülüyor (DMR/YSF/P25/NXDN)."
                                     + (f" [şifre anahtarı: {key}]" if key else ""))
            else:
                self.log_signal.emit("Sayısal ses: DSD-FME kurulu değil — yalnızca 4FSK/C4FM tespiti aktif. "
                                     + info.get("hint", ""))
            return info
        else:
            if self.audio_player is not None:
                self.audio_player.set_digital(False)
            self.log_signal.emit("Sayısal ses: Kapatıldı.")
            return {"enabled": False, "dsd_available": False, "hint": ""}

    def _update_monitor(self, iq1, spectrum_info):
        """Sinyal izleme (spec 5.1.3): ayarlı frekansa kilitlen; sürekliliği + parametre geçmişini
        izle; kayıp/geri-gelme olaylarını logla. Tarama/TX sırasında askıya alınır."""
        if self.scan_active or self.tx_active or not self.use_hardware:
            return
        now = time.time()
        # Ayarlı frekansa (yeniden) kilitlen — kullanıcı frekans değiştirdiyse yeniden başla.
        if (not self.monitor.is_locked() or self.monitor.locked_freq_mhz is None
                or abs(self.monitor.locked_freq_mhz - self.center_freq_mhz) > 1e-6):
            self.monitor.lock(self.center_freq_mhz)
        snr = float(spectrum_info.get("snr_db", 0.0) or 0.0)
        present = snr >= 6.0
        carrier = (self.center_freq_mhz + self._last_peak_offset_hz / 1e6) if present else None
        power = None
        bw = None
        if present:
            try:
                power = float(self.dsp.compute_channel_power_dbm(iq1))
            except Exception:
                power = None
            bw = float(spectrum_info.get("occupied_bw_hz", 0.0) or 0.0)
        event = self.monitor.update(now, present, carrier_mhz=carrier, power_dbm=power,
                                    bw_hz=bw, snr_db=snr)
        if event == "drop":
            self.log_signal.emit(f"İzleme: {self.center_freq_mhz:.3f} MHz sinyali KAYBOLDU (kesinti).")
        elif event == "reacquire":
            self.log_signal.emit(f"İzleme: {self.center_freq_mhz:.3f} MHz sinyali GERİ GELDİ.")

    def _stop_audio(self):
        if self.audio_player is not None:
            try:
                self.audio_player.stop()
            except Exception:
                pass
            self.audio_player = None

    def set_df_mode(self, auto: bool):
        self.is_auto_df = auto

    def set_manual_angles(self, angles: tuple):
        self.manual_angles = angles

    def _udp_listener_loop(self):
        """Uzak DF düğümlerinden (PlutoSDR + bilgisayar) ETHERNET/UDP üzerinden gelen JSON kerteriz
        mesajlarını OTONOM dinler ve depoya işler. Beklenen şema (her düğüm periyodik gönderir):
            {"id":"NODE-2","freq_mhz":433.9,"azimuth_deg":137.5,
             "elevation_deg":12.0,"amp_dbm":-52.3,"snr_db":18.0}
        Port 5005. Geçersiz/eksik mesajlar sessizce atlanır (kabul edilmez)."""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.udp_socket.bind(("0.0.0.0", 5005))
            self.udp_socket.settimeout(1.0)
            while self._is_running:
                try:
                    data, _addr = self.udp_socket.recvfrom(2048)
                    msg = json.loads(data.decode("utf-8"))
                    if self.node_store.update_from_json(msg):
                        # SOHBET için: bu düğümün IP'sini öğren (chat mesajlarını buraya yollarız)
                        if msg.get("id"):
                            self._chat_peers[str(msg["id"])] = _addr[0]
                except socket.timeout:
                    continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                except OSError:
                    break
        except OSError as e:
            self.log_signal.emit(f"Backend: UDP Listener başlatılamadı ({e})")
        finally:
            if self.udp_socket:
                self.udp_socket.close()

    # ------------------------------------------------------------------ SOHBET (aux <-> merkez)
    def _chat_listener_loop(self):
        """SOHBET dinleyici (UDP 5006). Gelen mesajı arayüze iletir VE hub olarak diğer düğümlere dağıtır."""
        self.chat_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.chat_socket.bind(("0.0.0.0", self.CHAT_PORT))
            self.chat_socket.settimeout(1.0)
            while self._is_running:
                try:
                    data, addr = self.chat_socket.recvfrom(4096)
                    msg = json.loads(data.decode("utf-8"))
                    if msg.get("type") != "chat" or "text" not in msg:
                        continue
                    frm = str(msg.get("from", "?"))
                    if frm and frm != "MERKEZ":
                        self._chat_peers[frm] = addr[0]        # yanıt için IP öğren
                    self.chat_signal.emit({"from": frm, "text": str(msg["text"])[:500],
                                           "ts": float(msg.get("ts", time.time()))})
                    self._relay_chat(msg, exclude_ip=addr[0])  # HUB: diğer düğümlere dağıt
                except socket.timeout:
                    continue
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
                    continue
                except OSError:
                    break
        except OSError as e:
            self.log_signal.emit(f"Backend: Sohbet dinleyici başlatılamadı ({e})")
        finally:
            if self.chat_socket:
                self.chat_socket.close()

    def _relay_chat(self, msg, exclude_ip=None):
        """Sohbet mesajını bilinen tüm düğümlere (exclude_ip hariç) yolla."""
        if self.chat_socket is None:
            return
        raw = json.dumps(msg).encode("utf-8")
        for nid, ip in list(self._chat_peers.items()):
            if ip == exclude_ip:
                continue
            try:
                self.chat_socket.sendto(raw, (ip, self.CHAT_PORT))
            except OSError:
                pass

    def send_chat(self, text: str):
        """Merkezden sohbet mesajı gönder (tüm aux'lara) + arayüzde kendi mesajını göster."""
        text = str(text).strip()[:500]
        if not text:
            return
        msg = {"type": "chat", "from": "MERKEZ", "text": text, "ts": time.time()}
        self.chat_signal.emit({"from": "MERKEZ", "text": text, "ts": msg["ts"]})
        self._relay_chat(msg, exclude_ip=None)

    def run(self):
        self._is_running = True
        self.start_time = time.time()
        
        # Her başlatmada logger bağlantısını yenile (Durdur/Başlat döngüsünde kapandığı için)
        self.logger = MissionLogger(db_name="sdr_mission_logs.db")
        
        # Donanımı denemeye çalış (Yoksa False döner, simülasyon akar)
        self.use_hardware = self._init_hardware()
        
        # Ağ Dinleyicisini Başlat
        self.udp_thread = threading.Thread(target=self._udp_listener_loop, daemon=True)
        self.udp_thread.start()
        # Sohbet Dinleyicisini Başlat (aux <-> merkez, UDP 5006)
        self.chat_thread = threading.Thread(target=self._chat_listener_loop, daemon=True)
        self.chat_thread.start()
        
        # C++ SDR Motorunu başlat
        if self.use_hardware and hasattr(self, 'engine'):
            self.engine.start()
            
            # Gain Refresh: Akış başladıktan hemen sonra kazanç değerini tekrar gönder. 
            # Böylece cihaz "sağır" kalmaz.
            time.sleep(0.05) 
            self.engine.set_gain(self.gain_db)
            self.hardware_initialized = True
        
        self.log_signal.emit("SDR Worker Thread başlatıldı. Sinyal işleme aktif.")
        self.status_signal.emit(True)

        while self._is_running:
            try:
                loop_start = time.time()

                start_freq = self.center_freq_mhz - (self.bandwidth_mhz / 2.0)
                end_freq = self.center_freq_mhz + (self.bandwidth_mhz / 2.0)
                x_freqs = np.linspace(start_freq, end_freq, FFT_POINTS)

                # --- 1. I/Q SİNYAL OKUMA (MOTOR B - YALNIZCA DSP) ---
                if self.use_hardware:
                    if hasattr(self, 'engine'):
                        iq1 = self.engine.get_latest_iq()
                    else:
                        with self.iq_lock:
                            iq1 = self.latest_iq.copy()

                    # --- ADC DOYGUNLUK (CLIPPING) TESPİTİ ---
                    # ADC tam-skala genliği 1.0'dır. Gain çok yüksekse (ör. 70 dB, yoğun 2.4 GHz
                    # ortamı) sinyal 1.0'ı aşar; ADC kırpılır. Kırpılan sinyal harmonik/intermod
                    # üreterek TÜM bandı gürültüyle doldurur (spektrum "çim" görünür, tümsek kaybolur).
                    # Bunu sessizce geçmek yerine operatörü uyarıp gain düşürmeye yönlendiriyoruz.
                    peak_amp = float(np.max(np.abs(iq1))) if iq1.size else 0.0
                    # KIRPMA BAYRAĞI: sınıflandırma bunu okur. Kırpık sinyalde zarf ±tam-skalaya railed
                    # olur -> dijital (PSK/QAM) yapı yok olup FM/FSK'ye benzer (sahte). Bu yüzden kırpıkken
                    # AMC sonucunu göstermek yerine "gain düşür" uyarısı basılır (yanlış sayısal/analog kararı yok).
                    self._adc_clipping = bool(peak_amp >= 0.98)
                    self._dbg_peak_amp = peak_amp        # AMC teşhis log'u için (sahada dijital ayrımı)
                    if peak_amp >= 0.98 and (time.time() - getattr(self, '_last_sat_warn', 0.0)) > 2.0:
                        self.log_signal.emit(
                            f"⚠️ ADC DOYGUNLUĞU! Tepe genlik={peak_amp:.2f} (>1.0 = kırpma). "
                            f"Gain {self.gain_db:.0f} dB çok yüksek; spektrum harmoniklerle 'çim' gibi "
                            f"dolar. Güç Seviyesini (Gain) düşürün (ör. 30-45 dB).")
                        self._last_sat_warn = time.time()

                    # OTOMATİK KAZANÇ KONTROLÜ: tepe genliği ideal (doygunluk-altı) bantta tut.
                    self._service_agc(peak_amp)
                    # NOT: Sentetik iq2/iq3 (sahte 3-kanal faz-DF) KALDIRILDI. DF artık gerçek:
                    # yerel genlik-tabanlı kerteriz + uzak düğümlerden ağ (JSON) kerterizleri.
                else:
                    iq1 = np.full(FFT_POINTS, 1e-5, dtype=np.complex64)

                    # Simülasyon modunda TX eklemesi: donanıma gönderilen GERÇEK baseband
                    # tamponundan (moda özgü: baraj/ton/GNSS/analog/look-through) döngüsel
                    # bir dilim ekleyerek spektrumda seçilen modun etkisini gösterir.
                    if self.tx_active:
                        tx_buf = getattr(self, 'tx_pre_gen', None)
                        if tx_buf is not None and len(tx_buf) > 0:
                            idx = getattr(self, '_tx_sim_idx', 0)
                            tx_iq = np.take(tx_buf, np.arange(idx, idx + FFT_POINTS), mode='wrap')
                            self._tx_sim_idx = (idx + FFT_POINTS) % len(tx_buf)
                        else:
                            target_amp = self.jam_gen.jsr_to_amplitude(self.tx_params["jsr_db"], signal_amplitude=1.0)
                            tx_iq = self.jam_gen.generate_barrage_noise(FFT_POINTS, amplitude=target_amp)
                        iq1 = iq1 + tx_iq.astype(np.complex64)

                # --- 3. SİNYAL İŞLEME (DSP) + YÖN BULMA ---
                # ARABAKIŞLI KARIŞTIRMA: gerçek T/R aç/kapa döngüsünü sür (bulgu #3). Dinleme
                # penceresinde donanım TX'i fiziksel olarak kapanır; RX gerçek kanalı görür.
                if self.tx_active and self.tx_params.get("mode") == "LOOK_THROUGH":
                    self._service_look_through()

                # GNSS DİNAMİK DRIFT: "Kuzeye sürekli kaydır" modu gerçekten kaysın diye TX tamponunu
                # periyodik (~1 s) yeniden üret. Aksi halde buffer trigger anında bir kez üretilip
                # loop'landığından hedef konumu sabit kalır (drift görünmez). MANUAL/AUTONOMOUS statiktir.
                if (self.tx_active and self.tx_params.get("mode") == "GNSS_SPOOF"
                        and self.tx_params.get("spoof_mode") == "DYNAMIC_DRIFT"):
                    self._service_gnss_drift()

                # RX SİNYAL ANALİZİ (spektrum + event-tetiklemeli sınıflandırma). God Object'i
                # azaltmak için ayrı metoda taşındı; DF/nirengi (aşağıda) buna dokunmadan sürer.
                fft_dbm, spectrum_info = self._run_signal_analysis(iq1)

                # RF BANT TARAMA / SİNYAL TESPİTİ (şartname 5.1.1): aktifse mevcut pencerede gerçek
                # tepeleri yakala ve bir sonraki frekansa geç.
                if self.scan_active:
                    self._service_scan()

                # --- GERÇEK YÖN BULMA + KONUM (genlik-tabanlı DF + 3B LOB üçgenleme) ---
                # SNR geçidi için analiz sonucunu (spectrum_info) geçir: sinyal yokken genlik-DF
                # haritası gürültüyle kirlenmesin (sıralı yayın senaryosu).
                df = self._run_direction_finding(iq1, spectrum_info)
                df_rms_deg = df["df_rms_deg"]
                df_reference_deg = df["df_reference_deg"]

                # Arayüz geriye-uyumluluğu: kayıt sırasına göre 3 düğüm kerterizi/genliği (yoksa 0)
                node_ids = list(self.df_registry.keys())
                _nodes = df["nodes"]
                def _nb(i, key, default=0.0):
                    if i < len(node_ids) and node_ids[i] in _nodes:
                        return _nodes[node_ids[i]].get(key, default)
                    return default
                ang1_deg, ang2_deg, ang3_deg = (_nb(0, "azimuth_deg"), _nb(1, "azimuth_deg"), _nb(2, "azimuth_deg"))
                amp1, amp2, amp3 = (_nb(0, "amp_dbm", -120.0), _nb(1, "amp_dbm", -120.0), _nb(2, "amp_dbm", -120.0))
                # Konum: metre (ENU) -> arayüz ölçeği için km
                target_x = round(df["position_xyz_m"][0] / 1000.0, 3)
                target_y = round(df["position_xyz_m"][1] / 1000.0, 3)

                # --- GERÇEK DSP SES DEMODÜLASYONU ---
                audio_y = self.audio_demod.fm_demodulate(iq1, output_points=100)

                # --- GÖREV/SİNYAL/DF LOGLAMA (~1 Hz) — God Object azaltma: ayrı metoda taşındı ---
                self._do_periodic_logging(target_x, target_y, df, spectrum_info, ang1_deg)

                # --- SİNYAL İZLEME/TAKİP (5.1.3): süreklilik + parametre geçmişi ---
                self._update_monitor(iq1, spectrum_info)

                # --- ANALOG/SAYISAL AYRIMI (5.1.2) ---
                # Modülasyon etiketinden türetilir. ANALOG: AM, FM. SAYISAL: PSK/QAM/OFDM/FSK-C4FM/FHSS.
                # "FM/FSK" (henüz ses-kesinleştirmesi olmayan grup) -> "belirleniyor" (dürüst, sahte değil).
                mod_str = self._clf_result.get("modulation", "")
                ad = self._clf_result.get("analog_digital")   # _refine_fm_fsk kesinleştirdiyse dolu
                if "KIRPIK" in mod_str:
                    ana_dig_tag = " (KIRPIK — güç düşür, sınıflandırılamaz)"
                elif any(s in mod_str for s in ("Sinyal yok", "Belirlenemedi", "Ölçülüyor")):
                    ana_dig_tag = " (Sınıflandırılıyor...)"
                elif "FM/FSK" in mod_str:                      # henüz ses-kesinleştirmesi yok (grup)
                    ana_dig_tag = " (Analog/Sayısal: sesle belirleniyor)"
                elif ad == "Sayısal" or any(t in mod_str for t in
                        ("PSK", "QAM", "OFDM", "FSK", "C4FM", "FHSS", "Sayısal")):
                    ana_dig_tag = " [SAYISAL KESİN]"
                elif ad == "Analog" or "AM (" in mod_str or "FM (" in mod_str:
                    ana_dig_tag = " [ANALOG KESİN]"
                else:
                    ana_dig_tag = ""

                # --- 5. ARAYÜZE VERİ AKTARIMI ---
                payload = {
                    "x_freqs": x_freqs,
                    "fft_dbm": fft_dbm,
                    "audio_y": audio_y,
                    "angles": (ang1_deg, ang2_deg, ang3_deg),
                    "node_live_angles": df.get("node_live_angles"),   # aux canlı anten açısı (anlık)
                    "amps": (amp1, amp2, amp3),
                    "target_pos": (target_x, target_y),
                    "freq_bounds": (start_freq, end_freq),
                    "df_rms_deg": df_rms_deg,
                    "df_reference_deg": df_reference_deg,
                    "df_sample_count": self.df_tracker.sample_count(),
                    # GERÇEK DF/KONUM (3B LOB üçgenleme) — PPI ve düğüm izleme için
                    "df_nodes": df["nodes"],
                    "df_position_xyz_m": df["position_xyz_m"],
                    "df_residual_m": df["residual_m"],
                    "df_fix": df["fix"],
                    "df_fix_quality_deg": df.get("fix_quality_deg", 0.0),   # LOB kesişim açısı (GDOP)
                    "df_dimensionality": df.get("dimensionality", "-"),     # 2B (yer izdüşümü) / 3B
                    "df_active_count": df["active_count"],
                    "df_remote_nodes": df.get("remote_nodes", []),   # uzak düğüm bağlantı durumu
                    "df_self_bearing_deg": df["self_bearing_deg"],
                    "df_target_bearing_deg": df["target_bearing_deg"],
                    "df_target_range_m": df["target_range_m"],
                    "df_target_elevation_deg": df["target_elevation_deg"],
                    "bandwidth_mhz": self.bandwidth_mhz,
                    "gain_db": self.gain_db,
                    "agc_enabled": self.agc_enabled,
                    "power_cal_offset_db": self.dsp.cal_offset_db,
                    # RF bant tarama / sinyal tespiti (5.1.1) — tespit edilen sinyaller (frekans+güç+SNR)
                    "scan_active": self.scan_active,
                    # Omuz/etek bastırılmış TEMİZ görünüm (ilk görülme = 5.1.1 sıra kanıtı korunur)
                    "scan_detections": self._scan_detections_view(),
                    "tx_active": self.tx_active,
                    "tx_mode": self.tx_params["mode"] if self.tx_active else "NONE",
                    # Arabakış (5.2.2) kapalı-çevrim durumu: karıştırma aktif mi yoksa (yayın sonlandı)
                    # beklemede mi + tespit edilen hedef bandı
                    "look_through": ({"jamming": bool(getattr(self, "_lt_active", True)),
                                      "present": bool(getattr(self, "_lt_present", False)),
                                      "bw_hz": float(getattr(self, "_lt_bw_hz", 0.0) or 0.0),
                                      "offset_hz": float(getattr(self, "_lt_peak_offset_hz", 0.0) or 0.0)}
                                     if (self.tx_active and self.tx_params.get("mode") == "LOOK_THROUGH")
                                     else None),
                    "ant_status": self.hw_ctrl.is_connected,
                    # CANLI HAM ENKODER AÇISI (Serial Monitor'deki gibi anlık anten yönü). Kerteriz
                    # (df_self_bearing) sinyal-tabanlı bir TEPE'dir; bu ise antenin O ANKİ fiziksel
                    # yönü -> hızlı dönüşte bile akıcı takip. DF paneli + radar canlı yön çizgisi kullanır.
                    "encoder_angle_deg": (self.hw_ctrl.get_angle()
                                          if (self.hw_ctrl is not None and self.hw_ctrl.is_connected) else None),
                    # Gerçek ölçülen sinyal analizi (5.1.2)
                    "occupied_bw_hz": spectrum_info["occupied_bw_hz"],
                    "signal_class": spectrum_info["signal_class"] + ana_dig_tag,
                    "spectral_flatness": spectrum_info["flatness"],
                    "snr_db": spectrum_info["snr_db"],
                    # Otomatik Modülasyon Sınıflandırma (Faz 1) + Çoklama (Faz 2)
                    "modulation": self._clf_result["modulation"],
                    "mod_confidence": self._clf_result["confidence"],
                    "multiplex": self._clf_result.get("multiplex", "Belirlenemedi"),
                    "ekkt": self._clf_result.get("ekkt", "Belirlenemedi"),
                    "protocol": self._clf_result.get("protocol", "Belirlenemedi"),
                    # Taşıyıcı frekansı (5.1.2): merkez + ölçülen tepe offseti (sinyal varken)
                    "carrier_mhz": (round(self.center_freq_mhz + self._last_peak_offset_hz / 1e6, 4)
                                    if float(spectrum_info.get("snr_db", 0.0) or 0.0) >= 6.0 else None),
                    # Diğer sayısal özellikler (5.1.2): sembol/baud hızı
                    "symbol_rate_hz": float(self._clf_result.get("symbol_rate_hz", 0.0) or 0.0),
                    # KONSOLİDE PARAMETRELER (5.1.2 stabilizasyon): çoğunluk oyu + medyan + güven etiketi
                    # -> arayüz titrek ham değer yerine KARARLI, güven-etiketli değeri gösterir.
                    "params_consolidated": self.param_consol.consolidated(),
                    # CTCSS/DCS (5.2.3): analog FM hedefin alt-ses squelch'i (aldatmaya otomatik taşınır)
                    "ctcss_hz": float(getattr(self, "_detected_ctcss_hz", 0.0) or 0.0),
                    "squelch_type": getattr(self, "_squelch_type", None),   # "CTCSS" / "DCS" / None
                    # Sinyal izleme/takip (5.1.3): süreklilik + parametre geçmişi özeti
                    "monitor": self.monitor.status(),
                    # Sayısal ses (5.1.3): 4FSK/C4FM tespit özeti (sayısal çözme aktifken)
                    "digital_voice": (self.audio_player.get_last_fsk()
                                      if (self.audio_player is not None
                                          and self.audio_player.digital_enabled) else None),
                    # Sayısal VERİ (5.1.3 "ses ve/veya veriye ulaşılması"): DSD-FME'den çözülen
                    # çağrı/sync/renk-kodu/TG satırları (sayısal çözme aktif + DSD-FME çalışıyorsa)
                    "digital_data": (self.audio_player.get_digital_data()
                                     if (self.audio_player is not None
                                         and self.audio_player.digital_enabled) else None),
                }

                # C++ akış-sağlığı sayaçlarını payload'a yansıt + artışları logla (bulgu #6 tamamlayıcısı)
                self._poll_stream_health(payload)

                # GUI'yi boğmamak için sadece 33ms (30 FPS) geçince veriyi gönder
                # İşlemciyi boğmamak ve ekranı sabit FPS'te tutmak için bekleme
                process_time = time.time() - loop_start
                sleep_time = max(0.001, UPDATE_INTERVAL_SEC - process_time)
                time.sleep(sleep_time)
                # GUI BACKPRESSURE (kritik — donma/OOM önlemi): GUI thread'i önceki kareyi HENÜZ
                # işlemediyse bu kareyi GÖNDERME (drop). Güçlü/sürekli sinyalde CPU doyunca GUI 30 FPS'i
                # yetiştiremez; her kareyi kuyruğa atmak Qt olay kuyruğunu SINIRSIZ şişirir -> bellek
                # tükenir ve süreç "Killed" (OOM) olur. Bu bayrak kuyruğu en fazla 1 bekleyen kareye
                # sınırlar: GUI yavaşsa FPS düşer ama bellek sabit kalır (donma/çökme olmaz).
                # WATCHDOG: GUI slot'u bir istisna atıp bayrağı geri açamazsa (ya da bir kare kaybolursa)
                # bayrak sonsuza dek False kalıp GUI'yi tamamen dondurmasın; ~1 sn'dir onay gelmediyse
                # GUI'yi hazır varsay ve akışı sürdür (en kötü ihtimalle 1 fazladan kare kuyruğa girer).
                if not getattr(self, "_gui_ready", True) and (time.time() - getattr(self, "_last_gui_time", 0.0)) > 1.0:
                    self._gui_ready = True
                if getattr(self, "_gui_ready", True):
                    self._gui_ready = False
                    self.data_ready.emit(payload)
                    self._last_gui_time = time.time()
            except Exception as _loop_exc:
                # KRİTİK DAYANIKLILIK: bir kareyi işlerken beklenmedik bir hata olursa worker
                # QThread'ini ÖLDÜRME. Aksi halde run() sonlanır -> GUI'ye artık kare gelmez ->
                # arayüz KALICI DONAR ve PyQt yakalanmamış exception'da uygulamayı ÇÖKERTİR
                # (kullanıcı: 'grafiği 2-3 sn görüp sonra donuyor/çöküyor'). Hatayı logla, kısa
                # bekle, sonraki kareye geç. Böylece tek bir kötü kare tüm sistemi düşürmez.
                import traceback
                traceback.print_exc()
                try:
                    self.log_signal.emit(f'⚠️ İşleme hatası (kare atlandı, sistem ayakta): {_loop_exc}')
                except Exception:
                    pass
                self._gui_ready = True   # backpressure bayrağı kilitlenmesin
                time.sleep(0.1)          # hata sağanağında CPU'yu boğma
                
        self.log_signal.emit("SDR Worker Thread durduruldu.")
        self.status_signal.emit(False)