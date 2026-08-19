#!/usr/bin/env python3
"""Sistem performans analizi grafikleri — video/sunum/rapor için.

İKİ grafik üretir, İKİSİ DE GERÇEK PROJE KODUYLA çalışır (sonuçlar uydurma DEĞİL):

  1) SINIFLANDIRMA DOĞRULUĞU – SNR :  gerçek SignalClassifier ile sentetik AM/FM
     sinyalleri farklı SNR'de sınıflandırılır, doğru-oranı ölçülür.
  2) YÖN BULMA HATASI (Derece RMS) – SNR :  anten 360° döndürülür (gerçek anten
     hüzme deseni + AWGN), her azimutta GERÇEK DSP güç ölçümü alınır, GERÇEK
     AmplitudeDFEstimator kerterizi kestirir, gerçek yönle karşılaştırılır.

Çıktı:  data/perf_siniflandirma_snr.png  ve  data/perf_yonbulma_rms_snr.png

Çalıştır:  python3 tools/plot_performance.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.signal_classifier import SignalClassifier
from backend.direction_finding import AmplitudeDFEstimator
from backend.dsp_processor import DSPProcessor
from tests.synth_signals import gen_am, gen_fm, add_noise

# --- Artifact/konsol paletiyle uyumlu görsel kimlik (taktik/enstrüman) ---
BG      = "#0a0e14"
PANEL   = "#111823"
GRID    = "#1e2b3a"
INK     = "#c9d6e2"
INK_DIM = "#7d8da0"
PHOS    = "#3ee6b0"   # AM / fosfor yeşili
CYAN    = "#5bc8ff"   # FM
AMBER   = "#ffb547"   # ortalama / vurgu

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK_DIM, fontsize=10.5)
    ax.set_ylabel(ylabel, color=INK_DIM, fontsize=10.5)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9.5)


# ------------------------------------------------------------------ #
# 1) SINIFLANDIRMA DOĞRULUĞU – SNR
# ------------------------------------------------------------------ #
def plot_classification_accuracy(fs=2.4e6, n=8192, snr_list=None, trials=25):
    if snr_list is None:
        snr_list = list(range(0, 31, 3))
    clf = SignalClassifier(sample_rate_hz=fs)

    def run(gen_fn, label):
        acc = []
        for snr in snr_list:
            correct = 0
            for t in range(trials):
                rng = np.random.default_rng(1000 * int(snr) + t)
                x = add_noise(gen_fn(n, fs).astype(np.complex64), snr, rng)
                mod = clf.classify(x, check_hopping=False)["modulation"]
                if mod.startswith(label):
                    correct += 1
            acc.append(100.0 * correct / trials)
            print(f"  {label:3s}  SNR {snr:2d} dB -> %{acc[-1]:.0f} dogru")
        return acc

    print("[1/2] Siniflandirma dogrulugu olculuyor (gercek SignalClassifier)...")
    am_acc = run(lambda n, fs: gen_am(n, fs, mod_index=0.6), "AM")
    fm_acc = run(lambda n, fs: gen_fm(n, fs, freq_dev=20e3), "FM")
    overall = [(a + f) / 2 for a, f in zip(am_acc, fm_acc)]

    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.plot(snr_list, am_acc, "-o", color=PHOS, lw=2.2, ms=5, label="AM (Analog-Genlik)")
    ax.plot(snr_list, fm_acc, "-s", color=CYAN, lw=2.2, ms=5, label="FM (Analog-Frekans)")
    ax.plot(snr_list, overall, "--", color=AMBER, lw=1.6, label="Ortalama")
    ax.axhline(95, color=INK_DIM, lw=0.9, ls=":", alpha=0.7)
    ax.text(snr_list[0], 96, "%95 hedef", color=INK_DIM, fontsize=8.5)
    _style_ax(ax, "Modülasyon Sınıflandırma Doğruluğu — SNR",
              "SNR (dB)", "Doğru Sınıflama (%)")
    ax.set_ylim(0, 104)
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9.5, loc="lower right")
    fig.text(0.012, 0.015,
             f"Gerçek SignalClassifier · her nokta {trials} deneme · fs={fs/1e6:.1f} MHz · N={n}",
             color=INK_DIM, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(DATA_DIR, "perf_siniflandirma_snr.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"  -> kaydedildi: {out}\n")
    return out


# ------------------------------------------------------------------ #
# 2) YÖN BULMA HATASI (Derece RMS) – SNR
# ------------------------------------------------------------------ #
def antenna_gain_db(delta_deg, hpbw_deg=60.0, front_to_back_db=18.0):
    """Yönlü antenin göreli kazancı (dB). Ana lob için standart parabolik yaklaşım
    (-12·(Δ/HPBW)²), arka lob 'front_to_back' ile sınırlanır. Gerçek Yagi/panel benzeri."""
    d = ((delta_deg + 180.0) % 360.0) - 180.0
    g = -12.0 * (d / hpbw_deg) ** 2
    return np.maximum(g, -front_to_back_db)


def plot_df_rms(fs=1.0e6, n=1024, snr_list=None, trials=16, az_step=5.0):
    if snr_list is None:
        snr_list = list(range(0, 31, 3))
    dsp = DSPProcessor()
    azimuths = np.arange(0.0, 360.0, az_step)

    print("[2/2] Yon bulma RMS hatasi olculuyor (gercek AmplitudeDFEstimator + DSP)...")
    rms_curve, p10_curve, p90_curve = [], [], []
    for snr in snr_list:
        errs = []
        for t in range(trials):
            rng = np.random.default_rng(7000 * int(snr) + t)
            true_bearing = float(rng.uniform(0, 360))          # gerçek geliş yönü
            est = AmplitudeDFEstimator(bin_deg=1.0, window_deg=25.0, decay_sec=1e9)
            now = 0.0
            for az in azimuths:
                # Anten bu azimuta bakınca hedefe göre açısal fark -> kazanç -> genlik ölçeği
                g_db = antenna_gain_db(az - true_bearing)
                amp = 10.0 ** (g_db / 20.0)
                base = gen_am(n, fs, mod_index=0.3).astype(np.complex64) * amp
                iq = add_noise(base, snr, rng)
                p_dbm = dsp.compute_channel_power_dbm(iq)        # GERÇEK DSP güç ölçümü
                now += 0.01
                est.update(az, p_dbm, now=now)
            bearing, _, _, _ = est.bearing(now=now)
            if bearing is not None:
                err = abs(((bearing - true_bearing + 180.0) % 360.0) - 180.0)
                errs.append(err)
        errs = np.array(errs)
        rms = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else np.nan
        rms_curve.append(rms)
        p10_curve.append(float(np.percentile(errs, 10)) if len(errs) else np.nan)
        p90_curve.append(float(np.percentile(errs, 90)) if len(errs) else np.nan)
        print(f"  SNR {snr:2d} dB -> RMS {rms:.2f}°  (n={len(errs)})")

    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.fill_between(snr_list, p10_curve, p90_curve, color=PHOS, alpha=0.12,
                    label="%10–%90 dağılım")
    ax.plot(snr_list, rms_curve, "-o", color=PHOS, lw=2.4, ms=5, label="Derece RMS (hata)")
    ax.axhline(5, color=AMBER, lw=1.0, ls=":", alpha=0.8)
    ax.text(snr_list[-1], 5.3, "5° hedef", color=AMBER, fontsize=8.5, ha="right")
    _style_ax(ax, "Yön Bulma Doğruluğu — Derece RMS vs SNR  (şartname 5.1.4)",
              "SNR (dB)", "Kerteriz Hatası (Derece RMS)")
    ax.set_ylim(0, max(12, np.nanmax(p90_curve) * 1.1))
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=INK, fontsize=9.5, loc="upper right")
    fig.text(0.012, 0.015,
             f"Gerçek AmplitudeDFEstimator + DSP güç ölçümü · 360° tarama/{int(az_step)}° · "
             f"her nokta {trials} deneme · HPBW=60°",
             color=INK_DIM, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(DATA_DIR, "perf_yonbulma_rms_snr.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"  -> kaydedildi: {out}\n")
    return out


if __name__ == "__main__":
    print("=" * 60)
    print(" SISTEM PERFORMANS ANALIZI — grafik uretimi")
    print("=" * 60)
    o1 = plot_classification_accuracy()
    o2 = plot_df_rms()
    print("TAMAMLANDI. Grafikler:")
    print(f"  {o1}")
    print(f"  {o2}")
