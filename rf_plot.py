#!/usr/bin/env python3
"""
rf_plot.py — Plot waterfall + PSD from cuda_spectrum output
Supports both CSV (--db single frame) and binary (-o binary multi-frame) input.

Examples:
    # Single frame from CSV (16M-point):
    python3 rf_plot.py rf_16M.csv --fft 16777216 --rate 10e6 --center 435e6

    # Multi-frame waterfall from binary (4M-point, 141 frames):
    python3 rf_plot.py rf_4M_allframes.bin --binary --fft 4194304 --rate 10e6 --center 435e6

    # Zoom into a band:
    python3 rf_plot.py rf_16M.csv --fft 16777216 --rate 10e6 --center 435e6 --zoom 434.5 435.5
"""

import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from scipy.signal import find_peaks as sp_peaks
import argparse
import sys
import os

# ── colour theme ──────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
GRID_COL  = '#21262d'
TEXT_COL  = '#c9d1d9'
ACCENT    = '#58a6ff'
PEAK_COL  = '#ff7b72'
WF_CMAP   = 'inferno'

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_csv(path, fft_size):
    """Read cuda_spectrum CSV output (--db mode). Returns list of 1-D arrays."""
    frames = []
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('FRAME_START'):
                buf = []
            elif line.startswith('FRAME_END'):
                if buf:
                    frames.append(np.array(buf, dtype=np.float32))
                    buf = []
            else:
                try:
                    buf.append(float(line))
                except ValueError:
                    pass
    return frames


def read_binary(path, fft_size, max_frames=0):
    """Read cuda_spectrum binary output (-o binary). Returns list of 1-D arrays."""
    frame_bytes = 4 + fft_size * 4          # int32 frame_num + float32[fft_size]
    file_size   = os.path.getsize(path)
    n_frames    = file_size // frame_bytes
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)

    frames = []
    with open(path, 'rb') as fh:
        for i in range(n_frames):
            hdr = fh.read(4)
            if len(hdr) < 4:
                break
            raw = fh.read(fft_size * 4)
            if len(raw) < fft_size * 4:
                break
            frames.append(np.frombuffer(raw, dtype=np.float32).copy())
            if (i + 1) % 20 == 0:
                print(f"  loaded {i+1}/{n_frames} frames…", end='\r', file=sys.stderr)
    print(f"  loaded {len(frames)} frames.        ", file=sys.stderr)
    return frames

# ── frequency axis ────────────────────────────────────────────────────────────

def freq_axis(fft_size, sample_rate, center_freq):
    bins = np.arange(fft_size) - fft_size // 2
    return (bins * sample_rate / fft_size + center_freq) / 1e6   # MHz

# ── peak finder ───────────────────────────────────────────────────────────────

def find_peaks(spectrum, freqs_mhz, prominence=3.0, min_sep_mhz=0.05, n=12):
    min_sep_bins = max(1, int(min_sep_mhz / abs(freqs_mhz[1] - freqs_mhz[0])))
    idx, props = sp_peaks(spectrum, prominence=prominence, distance=min_sep_bins)
    if len(idx) == 0:
        return [], []
    order = np.argsort(props['prominences'])[::-1][:n]
    return idx[order], props['prominences'][order]

# ── main plot ─────────────────────────────────────────────────────────────────

def plot_rf(frames, fft_size, sample_rate, center_freq,
             title, save_path, vmin, vmax, zoom):

    freqs = freq_axis(fft_size, sample_rate, center_freq)
    psd   = np.mean(np.stack(frames), axis=0)   # average PSD across all frames
    bin_hz = sample_rate / fft_size
    n_frames = len(frames)
    integration = fft_size / sample_rate        # seconds per frame

    print(f"  Frames     : {n_frames}", file=sys.stderr)
    print(f"  FFT size   : {fft_size:,}", file=sys.stderr)
    print(f"  Bin width  : {bin_hz:.3f} Hz", file=sys.stderr)
    print(f"  Freq span  : {freqs[0]:.3f} – {freqs[-1]:.3f} MHz", file=sys.stderr)
    print(f"  Total time : {n_frames * integration:.1f} s", file=sys.stderr)

    # zoom mask
    if zoom:
        lo, hi = zoom
        mask = (freqs >= lo) & (freqs <= hi)
        freqs_p = freqs[mask]
        psd_p   = psd[mask]
        frames_p = [f[mask] for f in frames]
    else:
        freqs_p = freqs
        psd_p   = psd
        frames_p = frames

    # ── figure layout ─────────────────────────────────────────────────────────
    has_waterfall = n_frames > 1
    if has_waterfall:
        fig, (ax_psd, ax_wf) = plt.subplots(
            2, 1, figsize=(18, 10),
            gridspec_kw={'height_ratios': [2, 3]},
            facecolor=DARK_BG
        )
    else:
        fig, ax_psd = plt.subplots(1, 1, figsize=(18, 6), facecolor=DARK_BG)
        ax_wf = None

    for ax in ([ax_psd, ax_wf] if ax_wf else [ax_psd]):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)

    # ── PSD panel ─────────────────────────────────────────────────────────────
    ax_psd.plot(freqs_p, psd_p, color=ACCENT, linewidth=0.6, alpha=0.9)
    ax_psd.fill_between(freqs_p, psd_p.min() - 2, psd_p,
                        color=ACCENT, alpha=0.15)
    ax_psd.set_ylabel('Power (dBFS)', color=TEXT_COL)
    ax_psd.grid(True, color=GRID_COL, linewidth=0.5)
    ax_psd.set_xlim(freqs_p[0], freqs_p[-1])

    # noise floor estimate (10th percentile)
    noise_floor = np.percentile(psd_p, 10)
    ax_psd.axhline(noise_floor, color='#3d9970', linewidth=0.8,
                   linestyle='--', alpha=0.7, label=f'Noise floor ≈ {noise_floor:.1f} dBFS')

    # peak annotations
    peak_idx, _ = find_peaks(psd_p, freqs_p)
    for i, pi in enumerate(peak_idx):
        f   = freqs_p[pi]
        pwr = psd_p[pi]
        off = (f - center_freq / 1e6) * 1e3       # kHz from center
        ax_psd.axvline(f, color=PEAK_COL, linewidth=0.6, alpha=0.5)
        ax_psd.annotate(
            f'{f:.4f} MHz\n({off:+.1f} kHz)',
            xy=(f, pwr), xytext=(0, 6), textcoords='offset points',
            fontsize=5.5, color='#7ee787', ha='center', va='bottom',
            rotation=55
        )

    # header info
    info = (f"FFT: {fft_size:,} pts  |  Bin: {bin_hz:.2f} Hz  |  "
            f"Integration: {integration:.2f} s/frame  |  "
            f"Center: {center_freq/1e6:.3f} MHz  |  "
            f"BW: {sample_rate/1e6:.1f} MHz  |  Frames: {n_frames}")
    fig.suptitle(title, color=TEXT_COL, fontsize=13, fontweight='bold', y=0.98)
    ax_psd.set_title(info, color='#8b949e', fontsize=8, pad=4)
    ax_psd.legend(fontsize=7, facecolor=DARK_BG, edgecolor=GRID_COL,
                  labelcolor=TEXT_COL, loc='upper right')

    # ── waterfall panel ───────────────────────────────────────────────────────
    if ax_wf is not None:
        wf_data = np.stack(frames_p)             # shape: (n_frames, n_bins)

        _vmin = vmin if vmin is not None else np.percentile(wf_data, 5)
        _vmax = vmax if vmax is not None else np.percentile(wf_data, 99)

        extent = [freqs_p[0], freqs_p[-1], n_frames * integration, 0]
        im = ax_wf.imshow(
            wf_data,
            aspect='auto',
            extent=extent,
            cmap=WF_CMAP,
            norm=Normalize(vmin=_vmin, vmax=_vmax),
            interpolation='nearest',
            origin='upper'
        )
        cbar = fig.colorbar(im, ax=ax_wf, pad=0.01, fraction=0.015)
        cbar.set_label('Power (dBFS)', color=TEXT_COL, fontsize=8)
        cbar.ax.tick_params(colors=TEXT_COL, labelsize=7)

        ax_wf.set_xlabel('Frequency (MHz)', color=TEXT_COL)
        ax_wf.set_ylabel('Time (s)', color=TEXT_COL)
        ax_wf.set_xlim(freqs_p[0], freqs_p[-1])
        ax_wf.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax_wf.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    else:
        ax_psd.set_xlabel('Frequency (MHz)', color=TEXT_COL)
        ax_psd.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax_psd.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    plt.tight_layout(pad=1.5)

    out = save_path or 'rf_analysis.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"Saved → {out}", file=sys.stderr)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Plot waterfall + PSD from cuda_spectrum output (CSV or binary)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument('input',             help='CSV or binary file (use - for stdin CSV)')
    ap.add_argument('--binary',          action='store_true',
                                         help='Input is binary format (-o binary)')
    ap.add_argument('--fft',   type=int,   default=16777216, metavar='N',
                                         help='FFT size (default: 16777216)')
    ap.add_argument('--rate',  type=float, default=10e6,     metavar='HZ',
                                         help='Sample rate Hz (default: 10e6)')
    ap.add_argument('--center',type=float, default=435e6,    metavar='HZ',
                                         help='Center frequency Hz (default: 435e6)')
    ap.add_argument('--title', default='rf Analysis 430-440 MHz')
    ap.add_argument('--save',  default='rf_analysis.png',   metavar='FILE')
    ap.add_argument('--vmin',  type=float, default=None,
                                         help='Waterfall colour min dBFS (auto)')
    ap.add_argument('--vmax',  type=float, default=None,
                                         help='Waterfall colour max dBFS (auto)')
    ap.add_argument('--zoom',  type=float, nargs=2, metavar=('LO', 'HI'),
                                         help='Zoom MHz range, e.g. --zoom 434.5 435.5')
    ap.add_argument('--max-frames', type=int, default=0,
                                         help='Max frames to load from binary (0=all)')
    args = ap.parse_args()

    print(f"rf_plot.py  fft={args.fft:,}  rate={args.rate/1e6:.1f} Msps"
          f"  center={args.center/1e6:.3f} MHz", file=sys.stderr)

    if args.binary:
        frames = read_binary(args.input, args.fft, args.max_frames)
    else:
        frames = read_csv(args.input, args.fft)

    if not frames:
        print("ERROR: no frames loaded.", file=sys.stderr)
        sys.exit(1)

    plot_rf(
        frames      = frames,
        fft_size    = args.fft,
        sample_rate = args.rate,
        center_freq = args.center,
        title       = args.title,
        save_path   = args.save,
        vmin        = args.vmin,
        vmax        = args.vmax,
        zoom        = args.zoom,
    )


if __name__ == '__main__':
    main()
