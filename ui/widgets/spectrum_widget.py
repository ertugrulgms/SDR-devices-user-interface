from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg
import numpy as np

class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Başlık
        self.lbl_graph_title = QLabel("SPEKTRUM VE ŞELALE (WATERFALL) GÖRSELLEŞTİRME")
        self.lbl_graph_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        layout.addWidget(self.lbl_graph_title)

        # Anlık FFT Spektrum
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.getAxis('left').setTickSpacing(5.0, 5.0)
        self.plot_widget.setLabel('left', 'Güç', units='dBFS')
        self.plot_widget.showGrid(x=True, y=True)
        self.signal_line = self.plot_widget.plot(pen=pg.mkPen(color='#00ff00', width=2.0))
        # Sıkışabilir minimum: pencere ekrana sığabilsin (pyqtgraph varsayılan minimumu pencereyi
        # ekrandan büyük tutuyordu -> maximize bozuluyordu). Oran (3:2) ve görünüm değişmez.
        self.plot_widget.setMinimumHeight(120)
        layout.addWidget(self.plot_widget, stretch=3)

        # Şelale (Waterfall / Spectrogram) Grafiği
        self.waterfall_widget = pg.PlotWidget()
        self.waterfall_widget.setBackground('#1e1e1e')
        self.waterfall_widget.setLabel('bottom', 'Frekans Ekseni', units='MHz')
        self.waterfall_widget.setLabel('left', 'Zaman (Akan)')
        self.waterfall_widget.hideAxis('left')
        
        self.waterfall_image = pg.ImageItem()
        self.waterfall_widget.addItem(self.waterfall_image)
        
        # Profesyonel Renk Haritası
        colors = [
            (0, 0, 0),        # Siyah (Gürültü tabanı -100 dBm)
            (0, 0, 128),      # Koyu Mavi
            (0, 255, 255),    # Turkuaz
            (0, 255, 0),      # Yeşil
            (255, 255, 0),    # Sarı
            (255, 0, 0)       # Kırmızı (Güçlü sinyal tepeleri)
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.waterfall_image.setLookupTable(cmap.getLookupTable(start=0.0, stop=1.0, alpha=False))
        self.waterfall_image.setLevels([-100, -20])
        
        self.waterfall_widget.setMinimumHeight(80)      # sıkışabilir (bkz. üstteki not)
        layout.addWidget(self.waterfall_widget, stretch=2)

    def update_spectrum(self, freqs, spectrum_dbm):
        self.signal_line.setData(freqs, spectrum_dbm)
        self.plot_widget.setXRange(freqs[0], freqs[-1], padding=0)

    def update_waterfall(self, waterfall_data):
        self.waterfall_image.setImage(waterfall_data, autoLevels=False)
