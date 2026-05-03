#!/usr/bin/env python3
"""
find_periodicity.py — Find periodic structure in a power envelope time series.

Reads CSV from extract_power_envelope.py (or stdin), FFTs the power-vs-time
series to find dominant periodicities, and compares against known digital
radio frame/slot timings.

Output: console report + optional PNG plot

Known timing references:
    TETRA  : frame=56.667 ms, slot=14.167 ms  (4 slots/frame, 18 frames/multiframe)
    DMR    : frame=60.000 ms, slot=30.000 ms  (2 slots/frame)
    GSM    : frame=4.615 ms,  slot=0.577 ms   (8 slots/frame)
    P25    : frame=180.000 ms (varies by phase)
    DECT   : frame=10.000 ms, slot=0.417 ms   (24 slots/frame)

Examples:
    # From file
    python3 find_periodicity.py tetra_envelope.csv --plot tetra_periodicity.png

    # From pipe
    python3 extract_power_envelope.py example_2m5_1M.bin \\
        --fft 1048576 --rate 2.5e6 --center 431e6 \\
        --lo 431.338e6 --hi 431.363e6 | \\
        python3 find_periodicity.py --plot tetra_periodicity.png
"""

import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import sys

# ── Known digital radio timings (ms) ─────────────────────────────────────────
KNOWN_TIMINGS = {
    'TETRA slot':       14.167,
    'TETRA frame':      56.667,
    'DMR slot':         30.000,
    'DMR frame':        60.000,
    'GSM slot':          0.577,
    'GSM frame':         4.615,
    'P25 frame':       180.000,
    'DECT slot':         0.417,
    'DECT frame':       10.000,
}

DARK_BG  = '#0d1117'
GRID_COL = '#21262d'
TEXT_COL = '#c9d1d9'
ACCENT   = '#58a6ff'
MATCH_COL = '#ff7b72'

# ── I/O ───────────────────────────────────────────────────────────────────────

def read_envelope(path):
    """Read time_s, power_db from CSV. Accepts '-' for stdin."""
    times, powers = [], []
    src = sys.stdin if path == '-' else open(path)
    for line in src:
        line = line.strip()
        if line.startswith('#') or line.startswith('time_s'):
            continue
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                times.append(float(parts[0]))
                powers.append(float(parts[1]))
            except ValueError:
                pass
    if path != '-':
        src.close()
    return np.array(times), np.array(powers)

# ── Analysis ──────────────────────────────────────────────────────────────────

def find_periodicity(times, powers, n_peaks=10):
    """
    FFT the power time series to find dominant periodicities.
    Returns (freqs_hz, spectrum, peak_periods_ms, peak_powers).
    """
    # Uniform time grid check
    dt = np.median(np.diff(times))        # frame period in seconds
    fs = 1.0 / dt                         # "sample rate" of envelope = frames/sec

    # Remove DC (mean) before FFT
    x = powers - np.mean(powers)

    # Zero-pad for finer frequency resolution
    N = len(x)
    Nfft = max(N * 8, 65536)             # 8x zero-padding
    X = np.abs(np.fft.rfft(x, n=Nfft))
    freqs = np.fft.rfftfreq(Nfft, d=dt)  # Hz

    # Avoid DC bin
    X[0] = 0

    # Find peaks in periodicity spectrum
    from scipy.signal import find_peaks as sp_peaks
    min_sep = max(1, int(0.5 / (freqs[1] - freqs[0])))   # 0.5 Hz separation min
    peaks, props = sp_peaks(X, prominence=np.max(X) * 0.05, distance=min_sep)

    if len(peaks) == 0:
        return freqs, X, np.array([]), np.array([])

    # Sort by prominence
    order = np.argsort(props['prominences'])[::-1][:n_peaks]
    top_peaks = peaks[order]
    top_powers = X[top_peaks]
    top_periods_ms = 1000.0 / freqs[top_peaks]

    return freqs, X, top_periods_ms, top_powers


def match_known(period_ms, tolerance_pct=5.0):
    """Return list of (name, ref_ms, error_pct) for close known timings."""
    matches = []
    for name, ref_ms in KNOWN_TIMINGS.items():
        err_pct = abs(period_ms - ref_ms) / ref_ms * 100
        if err_pct <= tolerance_pct:
            matches.append((name, ref_ms, err_pct))
    return sorted(matches, key=lambda x: x[2])


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(times, powers, periods_ms, peak_powers, dt):
    print()
    print("=" * 60)
    print("  PERIODICITY ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Frames      : {len(times)}")
    print(f"  Duration    : {times[-1]:.2f} s")
    print(f"  Frame period: {dt*1000:.3f} ms  ({1/dt:.2f} frames/s)")
    print(f"  Power range : {powers.min():.2f} – {powers.max():.2f} dBFS")
    print(f"  Power std   : {powers.std():.3f} dB")
    print()

    if len(periods_ms) == 0:
        print("  No significant periodicities found.")
        return

    print(f"  {'Period (ms)':>12}  {'Rel Power':>10}  {'Match'}") 
    print(f"  {'-'*12}  {'-'*10}  {'-'*30}")

    norm = peak_powers[0]
    for period, pwr in zip(periods_ms, peak_powers):
        rel = pwr / norm
        matches = match_known(period)
        match_str = ', '.join(f"{m[0]} ({m[2]:.1f}% err)" for m in matches) if matches else ''
        flag = ' ◄' if matches else ''
        print(f"  {period:>12.3f}  {rel:>10.3f}  {match_str}{flag}")

    print()
    print("=" * 60)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(times, powers, freqs, spectrum, periods_ms, peak_powers, save_path):
    fig, (ax_env, ax_fft) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)

    for ax in [ax_env, ax_fft]:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5)

    # ── Power envelope ────────────────────────────────────────────────────────
    ax_env.plot(times, powers, color=ACCENT, linewidth=0.8)
    ax_env.set_xlabel('Time (s)', color=TEXT_COL)
    ax_env.set_ylabel('Band Power (dBFS)', color=TEXT_COL)
    ax_env.set_title('Power Envelope', color=TEXT_COL, fontsize=10)
    ax_env.set_xlim(times[0], times[-1])

    # ── Periodicity spectrum ──────────────────────────────────────────────────
    # Plot in period domain (ms) for readability — flip axis
    valid = freqs > 0
    period_ms_axis = 1000.0 / freqs[valid]
    spec_plot = spectrum[valid]

    # Plot from short to long period
    sort_idx = np.argsort(period_ms_axis)
    ax_fft.plot(period_ms_axis[sort_idx], spec_plot[sort_idx],
                color=ACCENT, linewidth=0.8)
    ax_fft.set_xlabel('Period (ms)', color=TEXT_COL)
    ax_fft.set_ylabel('Amplitude', color=TEXT_COL)
    ax_fft.set_title('Periodicity Spectrum', color=TEXT_COL, fontsize=10)

    # Limit x range to useful region (1 ms – 500 ms)
    ax_fft.set_xlim(1, min(500, period_ms_axis.max()))

    # Mark detected peaks
    norm = peak_powers[0] if len(peak_powers) > 0 else 1
    for period, pwr in zip(periods_ms, peak_powers):
        if 1 <= period <= 500:
            ax_fft.axvline(period, color=MATCH_COL, linewidth=1.0, alpha=0.7)
            matches = match_known(period)
            label = matches[0][0] if matches else f'{period:.1f} ms'
            ax_fft.text(period, pwr * 0.95, label,
                       color='#7ee787', fontsize=7, rotation=90,
                       ha='right', va='top')

    # Mark known TETRA/DMR timings for reference
    for name, ref_ms in KNOWN_TIMINGS.items():
        if 1 <= ref_ms <= 500:
            ax_fft.axvline(ref_ms, color='#3d9970', linewidth=0.5,
                          linestyle=':', alpha=0.4)

    fig.suptitle('TDMA Periodicity Analysis', color=TEXT_COL,
                 fontsize=13, fontweight='bold')
    plt.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"Saved → {save_path}", file=sys.stderr)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Find periodic structure in power envelope time series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument('input', nargs='?', default='-',
                    help='Envelope CSV file (default: stdin)')
    ap.add_argument('--plot', metavar='FILE', default=None,
                    help='Save plot to PNG file')
    ap.add_argument('--n-peaks', type=int, default=10,
                    help='Number of peaks to report (default: 10)')
    ap.add_argument('--tolerance', type=float, default=5.0,
                    help='Match tolerance %% vs known timings (default: 5.0)')
    args = ap.parse_args()

    times, powers = read_envelope(args.input)
    if len(times) < 10:
        print("ERROR: need at least 10 frames.", file=sys.stderr)
        sys.exit(1)

    dt = np.median(np.diff(times))
    freqs, spectrum, periods_ms, peak_powers = find_periodicity(
        times, powers, n_peaks=args.n_peaks)

    print_report(times, powers, periods_ms, peak_powers, dt)

    if args.plot:
        plot_results(times, powers, freqs, spectrum,
                     periods_ms, peak_powers, args.plot)


if __name__ == '__main__':
    main()
