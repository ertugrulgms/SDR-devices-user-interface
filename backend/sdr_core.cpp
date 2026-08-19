#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

struct GNSSSatellite {
    int prn;
    double code_phase_chips;
    double doppler_hz;
    std::vector<int> nav_bits;
    int current_bit_index;
    bool active;
    
    GNSSSatellite() : prn(0), code_phase_chips(0), doppler_hz(0), current_bit_index(0), active(false) {}
    GNSSSatellite(int p) : prn(p), code_phase_chips(0), doppler_hz(0), current_bit_index(0), active(true) {}
};

// GPS PRN 1-32 phase shifts for G2 generator
const int G2_SHIFTS[32] = {
    5, 6, 7, 8, 17, 18, 139, 140, 141, 251, 252, 254, 255, 256, 257, 258,
    469, 470, 471, 472, 473, 474, 509, 512, 513, 514, 515, 516, 859, 860, 861, 862
};

std::vector<int> generate_gold_code(int prn) {
    if (prn < 1 || prn > 32) return std::vector<int>(1023, 1);
    
    std::vector<int> g1(1023), g2(1023);
    int reg1[10], reg2[10];
    for (int i = 0; i < 10; i++) { reg1[i] = 1; reg2[i] = 1; }
    
    for (int i = 0; i < 1023; i++) {
        g1[i] = reg1[9];
        g2[i] = reg2[9];
        
        int fb1 = reg1[2] ^ reg1[9];
        int fb2 = reg2[1] ^ reg2[2] ^ reg2[5] ^ reg2[8] ^ reg2[9];
        
        for (int j = 9; j > 0; j--) {
            reg1[j] = reg1[j-1];
            reg2[j] = reg2[j-1];
        }
        reg1[0] = fb1;
        reg2[0] = fb2;
    }
    
    std::vector<int> code(1023);
    int shift = G2_SHIFTS[prn - 1];
    for (int i = 0; i < 1023; i++) {
        code[i] = (g1[i] ^ g2[(i + 1023 - shift) % 1023]) ? -1 : 1;
    }
    return code;
}

namespace py = pybind11;

class SDREngine {
private:
    SoapySDR::Device* sdr_device;
    SoapySDR::Stream* rx_stream;
    SoapySDR::Stream* tx_stream;

    std::atomic<bool> is_running;
    std::atomic<bool> is_tx_active;
    // RX akışı açık mı? SÜREKLİ KARIŞTIRMADA (5.2.1: almaç gerekmez) RX kapatılır ki tam-çift-yönlü
    // (full-duplex) USB yükü ikiye katlanmasın. 56 Msps'de RX+TX aynı anda = 448 MB/s -> USB3 sınırını
    // aşar -> TX underflow. RX kapanınca tüm USB bandı TX'e kalır -> underflow biter, jamming sürekli olur.
    std::atomic<bool> rx_enabled;

    std::thread rx_thread;
    std::thread tx_thread;

    // Buffer to hold the latest 2048 IQ points for GUI
    std::vector<std::complex<float>> latest_iq;
    std::mutex iq_mutex;

    // Buffer holding the TX signal to loop
    std::vector<std::complex<float>> tx_buffer;
    std::mutex tx_mutex;

    // GNSS State (Aşama 4: C++ Tabanlı Sinyal Sentezi)
    std::vector<GNSSSatellite> gnss_sats;
    std::mutex gnss_mutex;
    bool gnss_active = false;
    double gnss_time_sec = 0.0;
    
    // GPS C/A Code (1023 chips) cache for each PRN
    std::vector<std::vector<int>> ca_code_cache;

    double sample_rate;
    double center_freq;
    double gain;
    double tx_gain;   // TX kazancı; önceden init içinde 70.0 sabit kodluydu, artık ayarlanabilir

    // Akış sağlığı sayaçları (Python tarafı raporlayabilsin diye atomik).
    std::atomic<unsigned long> overflow_count;       // RX: host yetişemedi (örnek kaybı)
    std::atomic<unsigned long> stream_error_count;   // RX: gerçek akış hatası
    std::atomic<unsigned long> tx_underflow_count;   // TX: yazma gecikmesi (underflow)

    // İlk TX aktivasyonu 100 ms başlangıç marjı ister (DMA dolsun, ilk-paket underflow'u
    // olmasın). ANCAK arabakışlı (look-through) karıştırma TX'i saniyede onlarca kez aç/kapar;
    // her reaktivasyonda 100 ms beklemek yayın penceresini yutar (TX fiilen yayın yapamaz).
    // Bu yüzden yalnızca OTURUMUN İLK aktivasyonu 100 ms, sonraki reaktivasyonlar kısa marj kullanır.
    std::atomic<bool> tx_first_activation;

    // --- KAYIPSIZ HALKA TAMPON (RING BUFFER) ---
    // rx_worker gelen TÜM örnekleri buraya yazar (geçmişi ATMAZ). Python kendi hızında
    // snapshot alır -> olay-tetiklemeli/uzun-kayıt analiz için (FHSS zaman-geçmişi, daha iyi
    // OFDM/AMC çözünürlüğü) yeterli sürekli geçmiş korunur. Önceki sürüm yalnızca son 2048
    // örneği tutup gerisini atıyordu (uzmanın "veriyi çöpe atıyorsunuz" eleştirisi buydu).
    static const size_t RING_SIZE = 1 << 22;   // 4.194.304 örnek (~1.75 s @2.4MHz, ~75 ms @56MHz)
    std::vector<std::complex<float>> ring;
    std::atomic<unsigned long long> ring_write;  // toplam yazılan örnek (monotonik; wrap için % RING_SIZE)
    std::atomic<unsigned long long> ring_read;   // SES akışı okuma imleci (gapless dinleme için)
    std::mutex ring_mutex;

    // CİHAZ KONTROL MUTEX'İ: Python (UI) thread'i frekans/gain/anten değiştirirken bu çağrılar
    // birbirleriyle çakışmasın (race condition). UHD çoğu kontrol işlemi için thread-safe olsa da
    // bu, eşzamanlı kontrol çağrılarını serileştirerek "device busy"/sessiz çökmeyi önler. Akış
    // (readStream/writeStream) bu kilidi TUTMAZ -> streaming bloke olmaz.
    std::mutex dev_mutex;

    void rx_worker() {
        size_t mtu = sdr_device->getStreamMTU(rx_stream);
        size_t chunk_size = std::max<size_t>(16384, mtu);
        std::vector<std::complex<float>> rx_buff(chunk_size);
        void* buffs[] = {rx_buff.data()};

        int flags;
        long long timeNs;
        bool rx_active = true;   // start() akışı aktive etti
        size_t consecutive_overflows = 0;   // ardışık overflow sayacı (adaptif toparlanma için)

        while(is_running) {
            // SÜREKLİ KARIŞTIRMA: RX kapatıldıysa akışı DEAKTİVE et (cihaz RX örneği göndermeyi bırakır
            // -> USB bandı tamamen TX'e kalır, underflow biter). Yeniden açılınca reaktive et.
            if (rx_enabled && !rx_active) {
                std::lock_guard<std::mutex> lock(dev_mutex);
                sdr_device->activateStream(rx_stream);
                rx_active = true;
            } else if (!rx_enabled && rx_active) {
                std::lock_guard<std::mutex> lock(dev_mutex);
                sdr_device->deactivateStream(rx_stream);
                rx_active = false;
            }
            if (!rx_active) {
                // RX duraklatıldı. Kısa uyku (1 ms): look-through dinleme penceresine geçince RX
                // akışı en fazla ~1 ms'de yeniden aktive olur -> kısa dinleme penceresi (ör. 12 ms)
                // içinde bile taze örnek toplanıp kanal ölçülebilir.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            // No GIL release needed here because std::thread doesn't hold the Python GIL
            int ret = sdr_device->readStream(rx_stream, buffs, chunk_size, flags, timeNs, 200000);

            if (ret > 0) {
                // 1) KAYIPSIZ HALKA TAMPONA yaz: gelen TÜM örnekler (get_snapshot uzun kayıt alsın).
                //    PERFORMANS: örnek-örnek modulo (% RING_SIZE) yerine BLOK KOPYA (std::copy/memcpy).
                //    Yüksek örnekleme hızında (56 Msps) örnek-başına bölme işlemi rx_worker'ı yavaşlatıp
                //    overflow'a yol açıyordu; sarma (wrap) en fazla 2 bitişik kopyayla halledilir.
                {
                    std::lock_guard<std::mutex> lock(ring_mutex);
                    unsigned long long w = ring_write.load();
                    size_t start = static_cast<size_t>(w % RING_SIZE);
                    size_t first = std::min(static_cast<size_t>(ret), RING_SIZE - start);
                    std::copy(rx_buff.begin(), rx_buff.begin() + first, ring.begin() + start);
                    if (static_cast<size_t>(ret) > first) {   // tampon sonunu aştı -> başa sar
                        std::copy(rx_buff.begin() + first, rx_buff.begin() + ret, ring.begin());
                    }
                    ring_write.store(w + static_cast<unsigned long long>(ret));
                }
                // 2) latest_iq (son 2048, canlı FFT/spektrum çizimi için) — mevcut davranış korunur
                std::lock_guard<std::mutex> lock(iq_mutex);
                size_t points_to_copy = std::min<size_t>(static_cast<size_t>(ret), static_cast<size_t>(2048));

                // Shift old data left
                std::rotate(latest_iq.begin(), latest_iq.begin() + points_to_copy, latest_iq.end());
                // Copy new data to the right
                std::copy(rx_buff.begin() + ret - points_to_copy, rx_buff.begin() + ret, latest_iq.end() - points_to_copy);
                consecutive_overflows = 0;   // başarılı okuma -> ardışık overflow zinciri kırıldı
            } else if (ret == SOAPY_SDR_OVERFLOW) {
                // Host akışı bir an yetiştiremedi -> örnek kaybı (akış sürer). ADAPTİF TOPARLANMA:
                // TEKİL/ARA SIRA overflow'da (geçici CPU sıçraması, ör. classify anı) BEKLEME —
                // hemen tekrar oku, hızlı toparlan, en az örnek kaybet. Yalnızca SÜREKLİ overflow'da
                // (host kalıcı yetişemiyor, ör. 56 Msps) kısa uyu; aksi halde döngü %100 CPU spin'e
                // girip UHD'yi 'O' seliyle boğar. Böylece çalışan sistemde ara sıra 'O' minimuma iner.
                overflow_count++;
                if (++consecutive_overflows > 8) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            } else if (ret == SOAPY_SDR_TIMEOUT) {
                // Zaman aşımı: bu pencerede veri gelmedi. Beklenen bir durum (sinyalsiz an);
                // latest_iq eski kalır, döngü sürer. Sessizce yutmak yerine ayırt ediyoruz.
                continue;
            } else if (ret < 0) {
                // Gerçek akış hatası (ör. cihaz koptu, stream bozuldu). Önceki sürüm bu durumu
                // sessizce yutup CPU'yu %100 döndüren sıkı bir döngüye giriyordu; kısa bekleme
                // ile hem CPU'yu koruyoruz hem sayaç üzerinden görünür kılıyoruz.
                stream_error_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void tx_worker() {
        size_t current_mtu = 0;                       // 0 = henüz alınmadı (aktivasyonda bir kez al)
        size_t tx_write_size = 0;                     // her writeStream'de yazılacak örnek (MTU katı)
        std::vector<std::complex<float>> local_tx_buff;
        size_t tx_idx = 0;
        bool first_packet = true;

        while(is_running) {
            if (!is_tx_active || tx_stream == nullptr) {
                first_packet = true;
                current_mtu = 0;                       // yeniden aktive olunca MTU'yu tazele
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            // MTU'yu YALNIZCA BİR KEZ al: her iterasyonda getStreamMTU çağırmak pahalı bir
            // sistem çağrısıdır ve writeStream'i geciktirip underflow (U) yaratır.
            if (current_mtu == 0) {
                size_t mtu = sdr_device->getStreamMTU(tx_stream);
                current_mtu = (mtu > 0) ? mtu : 4096;
                // TX YAZIM BLOĞU: tek MTU yerine ~8k örneklik (MTU katı) blok yaz. Yüksek örnekleme
                // hızında (56 Msps) tek-MTU yazımı saniyede on binlerce writeStream çağrısı demektir;
                // her çağrının yükü DMA'yı boşaltıp underflow (U) üretir. Büyük blok -> ~8× az çağrı,
                // DMA sürekli dolu kalır. tx_write_size, MTU'nun tam katıdır (kısmi paket olmaz).
                size_t mult = std::max<size_t>(1, (size_t)8192 / current_mtu);
                tx_write_size = current_mtu * mult;
                local_tx_buff.assign(tx_write_size, std::complex<float>(0, 0));
            }

            bool do_gnss = false;
            std::vector<GNSSSatellite> local_sats;
            double local_time = 0.0;
            size_t n_write = tx_write_size;           // bu iterasyonda yazılacak örnek sayısı

            {
                std::lock_guard<std::mutex> lock(gnss_mutex);
                do_gnss = gnss_active;
                if (do_gnss) {
                    local_sats = gnss_sats;
                    local_time = gnss_time_sec;
                }
            }

            if (do_gnss) {
                // GNSS örnek-başına ağır hesaplanır -> tek MTU'luk blok yaz (düşük hız zaten yeterli)
                n_write = current_mtu;
                double dt = 1.0 / sample_rate;
                for (size_t i = 0; i < current_mtu; i++) {
                    std::complex<double> sample(0, 0);
                    for (auto& sat : local_sats) {
                        if (!sat.active || sat.prn < 1 || sat.prn > 32) continue;
                        
                        int chip_idx = (int)(sat.code_phase_chips) % 1023;
                        if (chip_idx < 0) chip_idx += 1023;
                        
                        double bpsk = ca_code_cache[sat.prn - 1][chip_idx];
                        
                        if (!sat.nav_bits.empty() && sat.current_bit_index < sat.nav_bits.size()) {
                            int nav_bit = sat.nav_bits[sat.current_bit_index];
                            bpsk *= (nav_bit == 1 ? 1.0 : -1.0);
                        }
                        
                        double phase = 2.0 * M_PI * sat.doppler_hz * local_time;
                        sample += std::complex<double>(bpsk * cos(phase), bpsk * sin(phase));
                        
                        sat.code_phase_chips += (1.023e6) * dt;
                    }
                    local_tx_buff[i] = std::complex<float>(sample.real() * 0.1, sample.imag() * 0.1);
                    local_time += dt;
                }
                
                {
                    std::lock_guard<std::mutex> lock(gnss_mutex);
                    if (gnss_active) {
                        gnss_time_sec = local_time;
                        for (size_t i = 0; i < gnss_sats.size() && i < local_sats.size(); i++) {
                            gnss_sats[i].code_phase_chips = local_sats[i].code_phase_chips;
                        }
                    }
                }
            } else {
                std::lock_guard<std::mutex> lock(tx_mutex);
                if (tx_buffer.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // boş dönüşte CPU yakma
                    continue;
                }
                // tx_write_size örneği (büyük blok) döngüsel tx_buffer'dan doldur -> az writeStream çağrısı
                size_t remaining = tx_write_size;
                size_t out_idx = 0;
                while (remaining > 0) {
                    size_t chunk = std::min(remaining, tx_buffer.size() - tx_idx);
                    std::copy(tx_buffer.begin() + tx_idx,
                              tx_buffer.begin() + tx_idx + chunk,
                              local_tx_buff.begin() + out_idx);
                    tx_idx += chunk;
                    if (tx_idx >= tx_buffer.size()) tx_idx = 0;
                    out_idx += chunk;
                    remaining -= chunk;
                }
            }

            // KISMİ-YAZIM DÖNGÜSÜ: büyük blok tek writeStream'de tamamen yazılamayabilir (SoapySDR
            // yazdığı örnek sayısını döner). Kalanı yazana kadar ilerlet. Yalnızca İLK paket zaman
            // damgası taşır (burst başlangıcı); sonrası akışa bağlı (flags=0).
            size_t written = 0;
            while (written < n_write && is_tx_active && is_running) {
                const void* buffs[] = {local_tx_buff.data() + written};
                int flags = 0;
                long long ts = 0;
                if (first_packet) {
                    first_packet = false;
                    if (tx_first_activation.exchange(false)) {
                        // YALNIZCA İLK aktivasyon: DMA dolsun diye 100 ms marjlı zaman-damgalı başlangıç.
                        flags |= SOAPY_SDR_HAS_TIME;
                        ts = sdr_device->getHardwareTime("") + 100000000LL;
                    }
                    // Sonraki REAKTİVASYONLAR (look-through jam aç/kapa): flags=0 -> HEMEN yayına başla.
                    // Bu fazda RX KAPALI (USB tamamen TX'te) olduğundan başlangıç underflow'u olmaz;
                    // eski 15 ms marj her jam penceresinden ~15 ms yiyip gücü düşürüyordu. Artık jam
                    // penceresi TAM SÜRE + TAM GÜÇ radyasyon yapar.
                }
                // Timeout 200 ms: writeStream, DMA tamponu boşalana kadar BLOKE olsun ki host TX'i
                // sürekli beslesin ve underflow oluşmasın. Dönüş <0 ise underflow/timeout sayılır.
                int ret = sdr_device->writeStream(tx_stream, buffs, n_write - written, flags, ts, 200000);
                if (ret < 0) {
                    tx_underflow_count++;
                    break;                            // hata/timeout -> bu bloğu bırak, döngü başına dön
                }
                written += (size_t)ret;
            }
        }
    }

public:
    SDREngine() : sdr_device(nullptr), rx_stream(nullptr), tx_stream(nullptr),
                  is_running(false), is_tx_active(false), rx_enabled(true), tx_gain(70.0),
                  overflow_count(0), stream_error_count(0), tx_underflow_count(0),
                  tx_first_activation(true), ring_write(0), ring_read(0) {
        latest_iq.resize(2048, std::complex<float>(0, 0));
        ring.resize(RING_SIZE, std::complex<float>(0, 0));  // kayıpsız halka tampon
        
        ca_code_cache.resize(32);
        for(int i=0; i<32; i++) {
            ca_code_cache[i] = generate_gold_code(i + 1);
        }
    }

    ~SDREngine() {
        stop();
        if (sdr_device) {
            if (rx_stream) sdr_device->closeStream(rx_stream);
            if (tx_stream) sdr_device->closeStream(tx_stream);
            SoapySDR::Device::unmake(sdr_device);
            sdr_device = nullptr;
        }
    }

    bool init_hardware(double rate, double freq, double g) {
        sample_rate = rate;
        center_freq = freq;
        gain = g;
        
        try {
            sdr_device = SoapySDR::Device::make("driver=uhd");
            if (!sdr_device) return false;

            try { sdr_device->setMasterClockRate(sample_rate); } catch(...) {}
            sdr_device->setSampleRate(SOAPY_SDR_RX, 0, sample_rate);
            sdr_device->setFrequency(SOAPY_SDR_RX, 0, center_freq);
            sdr_device->setGain(SOAPY_SDR_RX, 0, gain);
            sdr_device->setAntenna(SOAPY_SDR_RX, 0, "TX/RX");
            // OTOMATİK ÖN-UÇ DÜZELTMELERİ (zero-IF ayna görüntüsü + DC tepesi giderme):
            //  * DC offset auto: merkez frekansta (LO) sahte DC tepesini bastırır.
            //  * IQ balance auto: I/Q kazanç/faz dengesizliğinin yarattığı AYNA GÖRÜNTÜSÜNÜ
            //    (merkeze göre simetrik sahte tepe) bastırır. Bu OLMADAN, güçlü bir sinyal
            //    (ör. telsiz PTT) merkezin +Δ tarafındaysa -Δ tarafında ~35 dB'lik sahte bir
            //    ikiz tepe belirir. UHD/B200 bu düzeltmeyi FPGA'da uygular; try/catch ile
            //    desteklemeyen sürücülerde sessizce atlanır.
            try { sdr_device->setDCOffsetMode(SOAPY_SDR_RX, 0, true); } catch(...) {}
            try { sdr_device->setIQBalanceMode(SOAPY_SDR_RX, 0, true); } catch(...) {}
            
            double bw_hz = std::min(std::max(sample_rate, 200e3), 56e6);
            try { sdr_device->setBandwidth(SOAPY_SDR_RX, 0, bw_hz); } catch(...) {}

            sdr_device->setSampleRate(SOAPY_SDR_TX, 0, sample_rate);
            sdr_device->setFrequency(SOAPY_SDR_TX, 0, center_freq);
            sdr_device->setGain(SOAPY_SDR_TX, 0, tx_gain);   // sabit 70.0 yerine ayarlanabilir alan
            sdr_device->setAntenna(SOAPY_SDR_TX, 0, "TX/RX");

            SoapySDR::Kwargs args;
            // RX DMA derinliği (1024->4096): yüksek örnekleme hızında (56 Msps) host kısa süre
            // gecikse bile USB tamponu taşmadan bekler -> overflow toleransı artar.
            args["num_recv_frames"] = "4096";
            args["num_send_frames"] = "2048";   // TX DMA derinliğini artır (512->2048): underflow toleransı
            rx_stream = sdr_device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, std::vector<size_t>{0}, args);
            tx_stream = sdr_device->setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32, std::vector<size_t>{0}, args);

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Init failed: " << e.what() << std::endl;
            // Kısmi açılmış kaynakları temizle (aksi halde cihaz/stream sızar ve sonraki
            // deneme "device busy" verir).
            if (sdr_device) {
                if (rx_stream) { sdr_device->closeStream(rx_stream); rx_stream = nullptr; }
                if (tx_stream) { sdr_device->closeStream(tx_stream); tx_stream = nullptr; }
                SoapySDR::Device::unmake(sdr_device);
                sdr_device = nullptr;
            }
            return false;
        }
    }

    void start() {
        if (!sdr_device || is_running) return;
        is_running = true;
        sdr_device->activateStream(rx_stream);
        rx_thread = std::thread(&SDREngine::rx_worker, this);
        tx_thread = std::thread(&SDREngine::tx_worker, this);
    }

    void stop() {
        if (!is_running) return;
        is_running = false;
        if (rx_thread.joinable()) rx_thread.join();
        if (tx_thread.joinable()) tx_thread.join();
        if (sdr_device) {
            sdr_device->deactivateStream(rx_stream);
            if (is_tx_active) {
                sdr_device->deactivateStream(tx_stream);
                is_tx_active = false;
            }
        }
    }

    void set_tx_active(bool active) {
        if (active && !is_tx_active) {
            // TX akışını AKTİVE ET. Önceki kapatmadaki deactivate/END_BURST sonrası stream'i
            // temiz yeniden başlatır; bu olmadan İKİNCİ açış bozuk durumda başlayıp underflow
            // üretiyordu. tx_worker'daki first_packet mantığı zamanlamayı yeniden kurar.
            if (sdr_device && tx_stream) {
                sdr_device->activateStream(tx_stream);
            }
            is_tx_active = true;
        } else if (!active && is_tx_active) {
            is_tx_active = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(15));  // tx_worker dursun
            if (sdr_device && tx_stream) {
                // Burst'ü bitir, sonra akışı DEAKTİVE ET (temiz kapanış -> temiz tekrar-açılış)
                int flags = SOAPY_SDR_END_BURST;
                std::complex<float> dummy(0, 0);
                const void* buffs[] = {&dummy};
                sdr_device->writeStream(tx_stream, buffs, 0, flags, 0, 100000);
                sdr_device->deactivateStream(tx_stream);
            }
        }
    }

    // SÜREKLİ KARIŞTIRMA için RX'i duraklat/sürdür. Python (worker) sürekli jamming başlarken
    // false, dururken true çağırır. false -> rx_worker RX akışını deaktive eder, USB tamamen TX'e
    // kalır (underflow biter). Arabakış (look-through) modunda RX açık bırakılır (dinleme gerekir).
    void set_rx_enabled(bool en) {
        rx_enabled = en;
    }

    void set_tx_buffer(py::array_t<std::complex<float>> input) {
        py::buffer_info buf = input.request();
        std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
        std::lock_guard<std::mutex> lock(tx_mutex);
        tx_buffer.assign(ptr, ptr + buf.size);
    }

    py::array_t<std::complex<float>> get_latest_iq() {
        std::vector<std::complex<float>> copy_iq(2048);
        {
            std::lock_guard<std::mutex> lock(iq_mutex);
            std::copy(latest_iq.begin(), latest_iq.end(), copy_iq.begin());
        }
        
        auto result = py::array_t<std::complex<float>>(2048);
        py::buffer_info buf = result.request();
        std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
        std::copy(copy_iq.begin(), copy_iq.end(), ptr);
        return result;
    }

    // KAYIPSIZ SNAPSHOT: halka tampondaki SON n örneği (bitişik, zaman-sıralı) döndürür.
    // Sınıflandırma tetiklendiğinde çağrılır -> son 2048 yerine daha uzun/temiz bir kayıt
    // (daha iyi frekans çözünürlüğü + OFDM CP otokorelasyonu + AMC için yeterli örnek).
    py::array_t<std::complex<float>> get_snapshot(size_t n) {
        if (n > RING_SIZE) n = RING_SIZE;
        std::vector<std::complex<float>> out;
        {
            std::lock_guard<std::mutex> lock(ring_mutex);
            unsigned long long w = ring_write.load();
            unsigned long long avail = std::min<unsigned long long>(w, RING_SIZE);
            if (n > avail) n = static_cast<size_t>(avail);
            out.resize(n);
            // BLOK KOPYA (yüksek hız için): örnek-örnek modulo yerine en çok 2 bitişik std::copy.
            // 56 Msps'de rx_worker halka kilidini sık ister; uzun modulo döngüsü kilidi tutarsa RX
            // taşar. Blok kopya kilit süresini ~sıfıra indirir -> yüksek hızda daha az overflow.
            size_t start = static_cast<size_t>((w - n) % RING_SIZE);
            size_t first = std::min(n, RING_SIZE - start);
            std::copy(ring.begin() + start, ring.begin() + start + first, out.begin());
            if (n > first) {
                std::copy(ring.begin(), ring.begin() + (n - first), out.begin() + first);
            }
        }
        auto result = py::array_t<std::complex<float>>(out.size());
        py::buffer_info buf = result.request();
        std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
        std::copy(out.begin(), out.end(), ptr);
        return result;
    }

    // GAPLESS SES AKIŞI: okuma imlecinden bu yana yazılan YENİ örnekleri (sıralı, bitişik) döndürür
    // ve imleci ilerletir. Ses thread'i bunu periyodik çağırıp demodüle eder -> kesintisiz dinleme.
    // Host yetişemez de tampon aşılırsa en eski örnekler atlanır (imleç en gerine sıçrar).
    py::array_t<std::complex<float>> get_stream_new(size_t max_n) {
        std::vector<std::complex<float>> out;
        {
            std::lock_guard<std::mutex> lock(ring_mutex);
            unsigned long long w = ring_write.load();
            unsigned long long r = ring_read.load();
            if (w - r > RING_SIZE) r = w - RING_SIZE;      // taşma -> eski örnekleri atla
            unsigned long long avail = w - r;
            size_t n = (avail < (unsigned long long)max_n) ? (size_t)avail : max_n;
            out.resize(n);
            // BLOK KOPYA (yüksek hız): örnek-örnek modulo yerine en çok 2 bitişik std::copy ->
            // kilit süresi ~sıfır -> rx_worker bloke olmaz -> 56 Msps'de daha az overflow.
            size_t start = static_cast<size_t>(r % RING_SIZE);
            size_t first = std::min(n, RING_SIZE - start);
            std::copy(ring.begin() + start, ring.begin() + start + first, out.begin());
            if (n > first) {
                std::copy(ring.begin(), ring.begin() + (n - first), out.begin() + first);
            }
            ring_read.store(r + n);
        }
        auto result = py::array_t<std::complex<float>>(out.size());
        py::buffer_info buf = result.request();
        std::copy(out.begin(), out.end(), static_cast<std::complex<float>*>(buf.ptr));
        return result;
    }

    // Ses akışını başlatırken imleci ŞU ANA al (birikmiş eski veriyi çalma).
    void reset_audio_cursor() { ring_read.store(ring_write.load()); }

    void set_frequency(double freq) {
        std::lock_guard<std::mutex> lock(dev_mutex);   // eşzamanlı kontrol çağrılarını serileştir
        center_freq = freq;
        if(sdr_device) {
            sdr_device->setFrequency(SOAPY_SDR_RX, 0, center_freq);
            sdr_device->setFrequency(SOAPY_SDR_TX, 0, center_freq);
        }
    }

    void set_gain(double g) {
        std::lock_guard<std::mutex> lock(dev_mutex);
        gain = g;
        if(sdr_device) {
            sdr_device->setGain(SOAPY_SDR_RX, 0, gain);
        }
    }

    void set_tx_gain(double g) {
        std::lock_guard<std::mutex> lock(dev_mutex);
        tx_gain = g;
        if(sdr_device) {
            sdr_device->setGain(SOAPY_SDR_TX, 0, tx_gain);
        }
    }

    void update_gnss_sats(const std::vector<GNSSSatellite>& sats, double time_sec) {
        std::lock_guard<std::mutex> lock(gnss_mutex);
        gnss_sats = sats;
        gnss_time_sec = time_sec;
        gnss_active = true;
    }

    // RX akış sağlığı: kaç overflow (örnek kaybı) ve kaç gerçek akış hatası oldu.
    unsigned long get_overflow_count() const { return overflow_count.load(); }
    unsigned long get_stream_error_count() const { return stream_error_count.load(); }
    unsigned long get_tx_underflow_count() const { return tx_underflow_count.load(); }

    void set_bandwidth(double bw) {
        std::lock_guard<std::mutex> lock(dev_mutex);
        sample_rate = bw;
        if(sdr_device) {
            sdr_device->setSampleRate(SOAPY_SDR_RX, 0, sample_rate);
            sdr_device->setSampleRate(SOAPY_SDR_TX, 0, sample_rate);
        }
    }

    void set_antenna(const std::string& rx_ant) {
        std::lock_guard<std::mutex> lock(dev_mutex);
        if (sdr_device) {
            try { sdr_device->setAntenna(SOAPY_SDR_RX, 0, rx_ant); } catch(...) {}
        }
    }
};

PYBIND11_MODULE(sdr_core, m) {
    py::class_<GNSSSatellite>(m, "GNSSSatellite")
        .def(py::init<int>())
        .def_readwrite("prn", &GNSSSatellite::prn)
        .def_readwrite("code_phase_chips", &GNSSSatellite::code_phase_chips)
        .def_readwrite("doppler_hz", &GNSSSatellite::doppler_hz)
        .def_readwrite("nav_bits", &GNSSSatellite::nav_bits)
        .def_readwrite("current_bit_index", &GNSSSatellite::current_bit_index)
        .def_readwrite("active", &GNSSSatellite::active);

    py::class_<SDREngine>(m, "SDREngine")
        .def(py::init<>())
        .def("init_hardware", &SDREngine::init_hardware)
        .def("start", &SDREngine::start)
        .def("stop", &SDREngine::stop)
        .def("set_tx_active", &SDREngine::set_tx_active)
        .def("set_rx_enabled", &SDREngine::set_rx_enabled)
        .def("set_tx_buffer", &SDREngine::set_tx_buffer)
        .def("get_latest_iq", &SDREngine::get_latest_iq)
        .def("get_snapshot", &SDREngine::get_snapshot)
        .def("get_stream_new", &SDREngine::get_stream_new)
        .def("reset_audio_cursor", &SDREngine::reset_audio_cursor)
        .def("set_frequency", &SDREngine::set_frequency)
        .def("set_gain", &SDREngine::set_gain)
        .def("set_tx_gain", &SDREngine::set_tx_gain)
        .def("set_bandwidth", &SDREngine::set_bandwidth)
        .def("set_antenna", &SDREngine::set_antenna)
        .def("get_overflow_count", &SDREngine::get_overflow_count)
        .def("get_stream_error_count", &SDREngine::get_stream_error_count)
        .def("get_tx_underflow_count", &SDREngine::get_tx_underflow_count)
        .def("update_gnss_sats", &SDREngine::update_gnss_sats);
}
