#!/usr/bin/env python3
"""
extract_power_envelope.py — Extract per-frame power envelope from cuda_spectrum binary output.

Integrates power across a specified frequency band for each frame, producing a
time series that reveals temporal structure such as TDMA slot patterns.

Output: CSV with columns: time_s, power_db, frame_num

Examples:
    # Extract TETRA signal band (431.338–431.363 MHz) from 2.5 Msps 1M-point binary
    python3 extract_power_envelope.py example_2m5_1M.bin \\
        --fft 1048576 --rate 2.5e6 --center 431e6 \\
        --lo 431.338e6 --hi 431.363e6 \\
        --out tetra_envelope.csv

    # Pipe directly into periodicity finder
    python3 extract_power_envelope.py example_2m5_1M.bin \\
        --fft 1048576 --rate 2.5e6 --center 431e6 \\
        --lo 431.338e6 --hi 431.363e6 | \\
        python3 find_periodicity.py
"""

import numpy as np
import argparse
import sys
import os


def extract_envelope(filepath, fft_size, sample_rate, center_freq, lo_hz, hi_hz):
    """
    Read binary frames, integrate power within [lo_hz, hi_hz], yield (time_s, power_db, frame_num).
    """
    frame_bytes  = 4 + fft_size * 4
    file_size    = os.path.getsize(filepath)
    n_frames_est = file_size // frame_bytes
    frame_time   = fft_size / sample_rate          # seconds per frame

    # Build frequency axis (fftshift'd — DC at center)
    freqs = (np.arange(fft_size) - fft_size // 2) * (sample_rate / fft_size) + center_freq

    # Signal band mask
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    n_bins = mask.sum()
    if n_bins == 0:
        print(f"ERROR: frequency range {lo_hz/1e6:.4f}–{hi_hz/1e6:.4f} MHz contains no bins.",
              file=sys.stderr)
        print(f"       Bin width: {sample_rate/fft_size:.2f} Hz, "
              f"span: {freqs[0]/1e6:.3f}–{freqs[-1]/1e6:.3f} MHz", file=sys.stderr)
        sys.exit(1)

    print(f"# extract_power_envelope", file=sys.stderr)
    print(f"#   file       : {filepath}", file=sys.stderr)
    print(f"#   fft_size   : {fft_size:,}", file=sys.stderr)
    print(f"#   sample_rate: {sample_rate/1e6:.2f} Msps", file=sys.stderr)
    print(f"#   center     : {center_freq/1e6:.3f} MHz", file=sys.stderr)
    print(f"#   band       : {lo_hz/1e6:.4f} – {hi_hz/1e6:.4f} MHz  ({n_bins} bins)", file=sys.stderr)
    print(f"#   frame_time : {frame_time*1000:.2f} ms/frame", file=sys.stderr)
    print(f"#   est frames : {n_frames_est}", file=sys.stderr)

    frame_num_out = 0
    with open(filepath, 'rb') as fh:
        while True:
            hdr = fh.read(4)
            if len(hdr) < 4:
                break
            raw = fh.read(fft_size * 4)
            if len(raw) < fft_size * 4:
                break

            frame_num = int(np.frombuffer(hdr, dtype=np.int32)[0])
            spectrum  = np.frombuffer(raw, dtype=np.float32)

            # Mean power in dBFS across signal band
            band_power = float(np.mean(spectrum[mask]))

            time_s = frame_num_out * frame_time
            yield time_s, band_power, frame_num

            frame_num_out += 1
            if frame_num_out % 50 == 0:
                print(f"  {frame_num_out}/{n_frames_est} frames", end='\r', file=sys.stderr)

    print(f"\n  done: {frame_num_out} frames extracted.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description='Extract per-frame power envelope from cuda_spectrum binary output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument('input',            help='Binary file from cuda_spectrum -o binary')
    ap.add_argument('--fft',  type=int,   required=True,  metavar='N',   help='FFT size')
    ap.add_argument('--rate', type=float, required=True,  metavar='HZ',  help='Sample rate Hz')
    ap.add_argument('--center',type=float,required=True,  metavar='HZ',  help='Center frequency Hz')
    ap.add_argument('--lo',   type=float, required=True,  metavar='HZ',  help='Band low edge Hz')
    ap.add_argument('--hi',   type=float, required=True,  metavar='HZ',  help='Band high edge Hz')
    ap.add_argument('--out',  default='-', metavar='FILE',
                    help='Output CSV file (default: stdout)')
    args = ap.parse_args()

    out = open(args.out, 'w') if args.out != '-' else sys.stdout

    print('time_s,power_db,frame_num', file=out)
    for t, p, fn in extract_envelope(
            args.input, args.fft, args.rate, args.center, args.lo, args.hi):
        print(f'{t:.6f},{p:.4f},{fn}', file=out)

    if args.out != '-':
        out.close()
        print(f"Saved → {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
