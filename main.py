import sys
import os
# RX OVERFLOW ("O" seli) KÖK-NEDEN ÇÖZÜMÜ: numpy/scipy'nin arka planındaki BLAS (OpenBLAS/MKL)
# varsayılan olarak TÜM CPU çekirdeklerinde thread açar. FFT/resample/classify çalışırken bu,
# tüm çekirdekleri doldurup C++ RX okuma thread'ini (rx_worker) zamanlanamaz hale getirir ->
# SDR'ın USB tamponu taşar -> UHD ekrana durmadan "O" basar. Thread sayısını 1'e sabitlemek
# numpy işlemlerini tek çekirdeğe hapseder, kalan çekirdekleri RX thread'ine bırakır (overflow
# biter). Küçük FFT'lerde (2048) tek-thread zaten hızlıdır. NOT: numpy import'undan ÖNCE olmalı.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Grafik sürücüsü hatasını engellemek için Qt niteliğini içeri aktarıyoruz
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from ui.main_window import SDRMainWindow

def main():
    # SİYAH EKRAN ÇÖZÜMÜ: 
    # Qt'ye grafik çizerken sorun çıkartan donanım sürücülerini zorlamamasını söylüyoruz.
    # Bu komut Linux/Ubuntu üzerindeki Qt grafik hatalarını kökten çözer.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES)
    
    app = QApplication(sys.argv)
    
    window = SDRMainWindow()

    # Pencereyi açılışta EKRANI KAPLAT (F11 tam-ekran değil; normal maksimize). Bazı pencere
    # yöneticileri (özellikle Wayland/GNOME) programatik showMaximized'i yok sayabildiği için,
    # önce pencereyi ekranın kullanılabilir (görev çubuğu hariç) alanına oturtuyoruz, sonra
    # maksimize durumunu da ayarlıyoruz. İçerik plotları sıkışabilir olduğundan pencere ekrana
    # tam sığar (eskiden içerik minimumu ekrandan büyük olduğu için pencere yarım açılıyordu).
    screen = app.primaryScreen()
    if screen is not None:
        window.setGeometry(screen.availableGeometry())
    window.showMaximized()

    from PyQt6.QtCore import QTimer
    QTimer.singleShot(0, window.showMaximized)   # WM inatçılığına karşı tekrar dene

    sys.exit(app.exec())

if __name__ == "__main__":
    main()